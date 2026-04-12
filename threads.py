"""Background threads for the Context Bleed Firewall.

Thread 1 - GraphBuilderThread:
    Periodically reads the graph, calls Llama (Ollama) and/or OpenAI to build a
    "mirror image" personality profile of the user, and stores it as a special
    mirror_profile node.

Thread 2 - SearchProcessorThread:
    Receives Chrome search events via a queue (populated by the /chrome_input
    FastAPI endpoint) and ensures every query is recorded and enriched in the
    graph.  Runs independently so the HTTP request returns immediately while
    enrichment happens asynchronously.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Any, Dict, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------------
# Thread 2 – Search Processor
# ---------------------------------------------------------------------------

class SearchProcessorThread(threading.Thread):
    """Dequeues Chrome search payloads and records them into the graph.

    The /chrome_input endpoint calls :meth:`enqueue` so the HTTP response is
    not blocked by LLM calls.  This thread does the actual record + optional
    Llama enrichment for search-type inputs.
    """

    def __init__(self, graph_store: Any) -> None:
        super().__init__(name="SearchProcessor", daemon=True)
        self._graph = graph_store
        self._queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, item: Dict[str, Any]) -> None:
        """Push a chrome-input payload onto the processing queue."""
        self._queue.put(item)

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def run(self) -> None:
        print("[SearchProcessor] Thread 2 started — watching for Chrome searches…")
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=1.0)
                self._process(item)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[SearchProcessor] Error processing item: {exc}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process(self, item: Dict[str, Any]) -> None:
        tab_id = item.get("tab_id", "unknown")
        text = item.get("text", "").strip()
        meta = item.get("meta", {})

        if not text:
            return

        print(f"[SearchProcessor] Recording search: '{text[:80]}' (tab={tab_id})")
        try:
            result = self._graph.record_chrome_input(
                tab_id=tab_id,
                element_id=item.get("element_id", "search"),
                text=text,
                element_meta=meta,
            )
            topic = result.get("topic", "unknown")
            node_id = result.get("graph_node", "")
            print(f"[SearchProcessor] Saved → topic={topic}  node={node_id}")

            # Optional: enrich the node with Llama intent classification
            intent = self._llama_search_intent(text)
            if intent and node_id and node_id in self._graph.nodes:
                self._graph.nodes[node_id]["llama_intent"] = intent
                self._graph._persist()
                print(f"[SearchProcessor] Llama intent: {intent[:120]}")
        except Exception as exc:  # noqa: BLE001
            print(f"[SearchProcessor] Failed to record '{text[:40]}': {exc}")

    def _llama_search_intent(self, query: str) -> Optional[str]:
        """Ask Ollama/Llama to infer the user's intent behind the search query."""
        enabled = os.getenv("ENABLE_OLLAMA", "1").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return None
        host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        model = os.getenv("TREE_ROUTE_MODEL", "gemma3:latest")
        prompt = (
            "In one sentence, what is the user's likely intent or goal behind this search query? "
            "Be concise and factual.\n\n"
            f'Query: "{query}"\n\nIntent:'
        )
        payload = json.dumps(
            {"model": model, "prompt": prompt, "stream": False,
             "options": {"num_predict": 60, "temperature": 0.2}}
        ).encode("utf-8")
        try:
            req = Request(
                f"{host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
        except (URLError, Exception):
            return None


# ---------------------------------------------------------------------------
# Thread 1 – Graph Builder (Mirror Profile)
# ---------------------------------------------------------------------------

class GraphBuilderThread(threading.Thread):
    """Builds a "mirror image" profile of the user from graph evidence.

    Every :attr:`INTERVAL` seconds the thread:
    1. Collects recent evidence nodes from the graph.
    2. Calls Llama (Ollama) to summarise the user's interests / patterns.
    3. Calls OpenAI (if key available) for a richer personality profile.
    4. Stores the combined profile as a ``mirror_profile`` node so the
       dashboard and graph view can display it.
    """

    INTERVAL: int = 30  # seconds between profile-build cycles
    MAX_EVIDENCE: int = 30  # evidence nodes fed to LLM

    def __init__(self, graph_store: Any) -> None:
        super().__init__(name="GraphBuilder", daemon=True)
        self._graph = graph_store
        self._stop = threading.Event()
        self._last_node_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def run(self) -> None:
        print("[GraphBuilder] Thread 1 started — building mirror user profile…")
        while not self._stop.is_set():
            try:
                self._build_cycle()
            except Exception as exc:  # noqa: BLE001
                print(f"[GraphBuilder] Cycle error: {exc}")
            self._stop.wait(self.INTERVAL)

    # ------------------------------------------------------------------
    # Build cycle
    # ------------------------------------------------------------------

    def _build_cycle(self) -> None:
        current_count = len(self._graph.nodes)
        if current_count == self._last_node_count:
            return  # nothing new – skip
        self._last_node_count = current_count

        evidence = self._graph._collect_evidence_nodes()
        if not evidence:
            return

        # Take the most recent N nodes
        recent = evidence[-self.MAX_EVIDENCE:]
        activity_lines = []
        for node in recent:
            label = (node.get("label") or "").strip()
            topic = (node.get("topic") or "general").strip()
            path_parts = node.get("memory_path") or []
            path_str = " > ".join(str(p) for p in path_parts[1:]) if len(path_parts) > 1 else topic
            if label:
                activity_lines.append(f"- [{path_str}] {label}")

        if not activity_lines:
            return

        activity_text = "\n".join(activity_lines)
        print(f"[GraphBuilder] Building mirror profile from {len(activity_lines)} evidence items…")

        llama_profile = self._call_llama(activity_text)
        openai_profile = self._call_openai(activity_text)

        combined = self._merge_profiles(llama_profile, openai_profile)
        if combined:
            source = (
                "llama+openai" if (llama_profile and openai_profile)
                else ("llama" if llama_profile else "openai")
            )
            self._store_profile(combined, source)

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _call_llama(self, activity_text: str) -> Optional[str]:
        """Call Ollama (local Llama) to generate a mirror profile summary."""
        enabled = os.getenv("ENABLE_OLLAMA", "1").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return None
        host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        model = os.getenv("TREE_ROUTE_MODEL", "gemma3:latest")
        prompt = (
            "You are a privacy-safe user analyst.  Based on the following browsing "
            "and search activity, write 3-5 bullet points summarising the user's "
            "interests, goals, and behavioural patterns.  Do NOT reproduce sensitive "
            "data verbatim.  Be concise and insightful.\n\n"
            f"Activity:\n{activity_text}\n\n"
            "Mirror Profile:"
        )
        payload = json.dumps(
            {"model": model, "prompt": prompt, "stream": False,
             "options": {"num_predict": 200, "temperature": 0.3}}
        ).encode("utf-8")
        try:
            req = Request(
                f"{host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                result = data.get("response", "").strip()
                if result:
                    print(f"[GraphBuilder] Llama profile: {result[:120]}…")
                return result or None
        except (URLError, Exception) as exc:
            print(f"[GraphBuilder] Llama unavailable: {exc}")
            return None

    def _call_openai(self, activity_text: str) -> Optional[str]:
        """Call OpenAI to generate a richer mirror profile."""
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
        max_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "250"))

        payload = json.dumps({
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are building a privacy-safe mirror profile of a user from their "
                        "search and browsing activity.  Identify their core interests, goals, "
                        "and behavioural patterns in 3-5 concise bullet points.  "
                        "Never reproduce sensitive data verbatim."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Recent activity:\n{activity_text}\n\n"
                        "Generate a concise mirror profile:"
                    ),
                },
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        try:
            req = Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            with urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                result = data["choices"][0]["message"]["content"].strip()
                if result:
                    print(f"[GraphBuilder] OpenAI profile: {result[:120]}…")
                return result or None
        except Exception as exc:
            print(f"[GraphBuilder] OpenAI unavailable: {exc}")
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_profiles(llama: Optional[str], openai: Optional[str]) -> Optional[str]:
        parts = [p for p in (llama, openai) if p]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return f"[Llama]\n{llama}\n\n[OpenAI]\n{openai}"

    def _store_profile(self, profile_text: str, source: str) -> None:
        node_id = f"mirror:profile:{int(time.time())}"
        self._graph.nodes[node_id] = {
            "id": node_id,
            "type": "mirror_profile",
            "label": "User Mirror Profile",
            "profile_text": profile_text,
            "source": source,
            "created_at": time.time(),
            "memory_path": ["You", "Mirror Profile"],
        }
        self._graph.add_edge("memory:you", node_id, "mirror_profile")
        self._graph._persist()
        print(f"[GraphBuilder] Mirror profile node stored → {node_id} (source={source})")
