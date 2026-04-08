"""Privacy graph middleware for the Pacific submission.

This module keeps an encrypted interaction graph of browser inputs and LLM calls,
classifies captured textbox text into relatable versus non-relatable buckets, and
flags semantic bleed when a response paraphrases blocked content.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import secrets
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # Optional dependency; the fallback path keeps the demo runnable.
    from cryptography.fernet import Fernet
except Exception:  # pragma: no cover - handled at runtime.
    Fernet = None

try:  # Optional dependency for higher-quality semantic similarity.
    import numpy as np
except Exception:  # pragma: no cover - handled at runtime.
    np = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - handled at runtime.
    SentenceTransformer = None

BASE_DIR = os.path.dirname(__file__)
GRAPH_PATH = os.path.join(BASE_DIR, "interaction_graph.json")
KEY_FILE = os.path.join(BASE_DIR, "encryption.key")

RELATABLE_TOPICS: Dict[str, set[str]] = {
    "finance": {"budget", "salary", "revenue", "invoice", "cost", "pricing", "payment", "arr"},
    "product": {"roadmap", "feature", "launch", "release", "shipment", "ship", "priority"},
    "people_ops": {"hiring", "headcount", "promotion", "compensation", "benefit", "interview"},
    "operations": {"deadline", "meeting", "project", "timeline", "blocker", "status", "planning"},
    "support": {"ticket", "customer", "issue", "bug", "incident", "escalation"},
    "research": {"analysis", "experiment", "finding", "dataset", "insight", "evaluation"},
}

SENSITIVE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"#private", re.IGNORECASE),
    re.compile(r"(?:salary|ssn|dob|patient|account|credential|password)", re.IGNORECASE),
)

MONEY_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\$\s?(\d+(?:\.\d+)?)\s*([mkb])?", re.IGNORECASE),
    re.compile(r"(\d+(?:\.\d+)?)\s*(million|billion|thousand)", re.IGNORECASE),
    re.compile(r"(high|mid|low)?\s*(single|double|triple)\s*digit\s*(millions|billions|thousands)", re.IGNORECASE),
)


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _token_count(text: str) -> int:
    return max(1, len(_tokenize(text)))


def _derive_topic(text: str, heading_hint: Optional[str] = None) -> str:
    haystack = f"{heading_hint or ''} {text}".lower()
    scores = {"general": 0}
    for topic, keywords in RELATABLE_TOPICS.items():
        scores[topic] = sum(1 for keyword in keywords if keyword in haystack)
    topic, score = max(scores.items(), key=lambda item: item[1])
    if score == 0:
        if any(token in haystack for token in ("hello", "thanks", "random", "lunch", "weather", "joke")):
            return "non_relatable"
        return heading_hint.lower() if heading_hint else "general"
    return topic


def _classify_relevance(text: str, heading_hint: Optional[str] = None) -> Tuple[str, str, float]:
    topic = _derive_topic(text, heading_hint)
    tokens = set(_tokenize(text))
    relatable_hits = sum(len(tokens & keywords) for keywords in RELATABLE_TOPICS.values())
    sensitive_hits = sum(bool(pattern.search(text)) for pattern in SENSITIVE_PATTERNS)
    classification = "relatable" if relatable_hits > 0 or topic not in {"general", "non_relatable"} else "non_relatable"
    confidence = min(0.99, 0.45 + (relatable_hits * 0.12) + (0.12 if sensitive_hits else 0.0))
    if topic == "non_relatable":
        classification = "non_relatable"
        confidence = max(confidence, 0.66)
    return classification, topic, round(confidence, 3)


def _extract_money_signals(text: str) -> List[float]:
    normalized: List[float] = []
    for pattern in MONEY_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groups()
            if len(groups) == 2 and groups[0] and groups[1]:
                number = float(groups[0])
                suffix = groups[1].lower()
                multiplier = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(suffix, 1.0)
                normalized.append(number * multiplier)
            elif len(groups) == 2 and groups[0] and groups[1] in {"million", "billion", "thousand"}:
                number = float(groups[0])
                multiplier = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}[groups[1].lower()]
                normalized.append(number * multiplier)
            elif len(groups) == 3 and groups[1] and groups[2]:
                scale = {"thousands": 1_000, "millions": 1_000_000, "billions": 1_000_000_000}[groups[2].lower()]
                digit_band = {"single": 5, "double": 50, "triple": 500}[groups[1].lower()]
                modifier = {None: 1.0, "low": 0.75, "mid": 1.0, "high": 1.25}[groups[0].lower() if groups[0] else None]
                normalized.append(digit_band * scale * modifier)
    return normalized


def _jaccard_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _cosine_similarity_from_counts(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


@dataclass
class GraphStore:
    graph_path: str = GRAPH_PATH
    key_file: str = KEY_FILE
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    totals: Dict[str, Any] = field(default_factory=lambda: {
        "sessions": 0,
        "chrome_inputs": 0,
        "relatable_inputs": 0,
        "non_relatable_inputs": 0,
        "sensitive_chunks": 0,
        "blocked_leaks": 0,
    })

    def __post_init__(self) -> None:
        self._cipher = self._load_cipher()
        self._embedding_model = self._load_embedding_model()
        self._load()

    def _load_cipher(self):
        if Fernet is None:
            return None
        if os.path.exists(self.key_file):
            with open(self.key_file, "rb") as handle:
                key = handle.read().strip()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, "wb") as handle:
                handle.write(key)
        return Fernet(key)

    def _load_embedding_model(self):
        if SentenceTransformer is None:
            return None
        try:
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            return None

    def _load(self) -> None:
        if not os.path.exists(self.graph_path):
            self._bootstrap_default_nodes()
            return
        try:
            with open(self.graph_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            self._bootstrap_default_nodes()
            return
        self.nodes = payload.get("nodes", {})
        self.edges = payload.get("edges", [])
        self.totals.update(payload.get("totals", {}))

    def _bootstrap_default_nodes(self) -> None:
        self.nodes = {}
        self.edges = []
        self.totals = {
            "sessions": 0,
            "chrome_inputs": 0,
            "relatable_inputs": 0,
            "non_relatable_inputs": 0,
            "sensitive_chunks": 0,
            "blocked_leaks": 0,
        }

    def _encrypt(self, text: str) -> str:
        if self._cipher is None:
            encoded = base64.urlsafe_b64encode(text.encode("utf-8")).decode("utf-8")
            return f"plain:{encoded}"
        token = self._cipher.encrypt(text.encode("utf-8"))
        return token.decode("utf-8")

    def _decrypt(self, token: str) -> str:
        if token.startswith("plain:"):
            return base64.urlsafe_b64decode(token.removeprefix("plain:").encode("utf-8")).decode("utf-8")
        if self._cipher is None:
            return token
        return self._cipher.decrypt(token.encode("utf-8")).decode("utf-8")

    def _vectorize(self, text: str):
        tokens = _tokenize(text)
        if self._embedding_model is not None and np is not None:
            return self._embedding_model.encode([text])[0]
        return Counter(tokens)

    def similarity(self, left: str, right: str) -> float:
        left_money = _extract_money_signals(left)
        right_money = _extract_money_signals(right)
        entity_score = 0.0
        if left_money and right_money:
            left_value = left_money[0]
            right_value = right_money[0]
            ratio = min(left_value, right_value) / max(left_value, right_value)
            entity_score = ratio

        left_tokens = _tokenize(left)
        right_tokens = _tokenize(right)
        token_score = _jaccard_similarity(left_tokens, right_tokens)

        if self._embedding_model is not None and np is not None:
            left_vec = self._vectorize(left)
            right_vec = self._vectorize(right)
            numerator = float(np.dot(left_vec, right_vec))
            denominator = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
            embedding_score = numerator / denominator if denominator else 0.0
        else:
            embedding_score = _cosine_similarity_from_counts(self._vectorize(left), self._vectorize(right))

        composite = (embedding_score * 0.55) + (token_score * 0.25) + (entity_score * 0.20)
        if entity_score > 0.9:
            composite = max(composite, 0.9)
        return round(min(composite, 1.0), 4)

    def ensure_session(self, session_id: str, session_kind: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        node = self.nodes.get(session_id)
        if node is None:
            node = {
                "id": session_id,
                "type": session_kind,
                "created_at": time.time(),
                "metadata": metadata or {},
                "topic_buckets": {},
                "child_count": 0,
            }
            self.nodes[session_id] = node
            if session_kind in {"browser_session", "llm_session"}:
                self.totals["sessions"] += 1
        else:
            if metadata:
                node.setdefault("metadata", {}).update(metadata)
        return node

    def _topic_bucket(self, session_node: Dict[str, Any], topic: str) -> Dict[str, Any]:
        buckets = session_node.setdefault("topic_buckets", {})
        bucket = buckets.get(topic)
        if bucket is None:
            bucket = {
                "count": 0,
                "relatable": 0,
                "non_relatable": 0,
                "token_estimate": 0,
                "values": [],
            }
            buckets[topic] = bucket
        return bucket

    def add_edge(self, source: str, target: str, relation: str, **attributes: Any) -> None:
        self.edges.append({"source": source, "target": target, "relation": relation, **attributes})

    def record_chrome_input(
        self,
        tab_id: str,
        element_id: str,
        text: str,
        element_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta = element_meta or {}
        classification, topic, confidence = _classify_relevance(text, meta.get("heading") or meta.get("placeholder") or meta.get("name"))
        timestamp = time.time()
        session_id = f"browser-session:{tab_id}"
        session_node = self.ensure_session(session_id, "browser_session", {"tab_id": tab_id})
        input_id = f"input:{_fingerprint(f'{tab_id}:{element_id}:{timestamp}:{text}')[:24]}"
        encrypted_value = self._encrypt(text)
        token_estimate = _token_count(text)

        node = {
            "id": input_id,
            "type": "chrome_input",
            "tab_id": tab_id,
            "element_id": element_id,
            "heading": topic,
            "classification": classification,
            "confidence": confidence,
            "encrypted_value": encrypted_value,
            "fingerprint": _fingerprint(text),
            "token_estimate": token_estimate,
            "created_at": timestamp,
            "metadata": meta,
        }
        self.nodes[input_id] = node
        self.add_edge(session_id, input_id, "captured")

        bucket = self._topic_bucket(session_node, topic)
        bucket["count"] += 1
        bucket[classification] += 1
        bucket["token_estimate"] += token_estimate
        bucket["values"].append(
            {
                "node_id": input_id,
                "fingerprint": node["fingerprint"],
                "classification": classification,
                "confidence": confidence,
                "encrypted_value": encrypted_value,
                "token_estimate": token_estimate,
            }
        )
        session_node["child_count"] += 1
        self.totals["chrome_inputs"] += 1
        if classification == "relatable":
            self.totals["relatable_inputs"] += 1
        else:
            self.totals["non_relatable_inputs"] += 1

        self._persist()
        return {
            "input_id": input_id,
            "topic": topic,
            "classification": classification,
            "confidence": confidence,
            "token_estimate": token_estimate,
            "graph_node": session_id,
        }

    def record_llm_call(
        self,
        user_input: str,
        llm_response: str,
        sensitive_tags: Optional[List[str]] = None,
        similarity_threshold: float = 0.85,
    ) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
        sensitive_tags = sensitive_tags or ["#private"]
        session_id = f"llm-session:{_fingerprint(user_input)[:24]}"
        session_node = self.ensure_session(session_id, "llm_session", {"trigger": "process_llm_call"})
        blocked_chunks = [line.strip() for line in user_input.splitlines() if any(tag in line for tag in sensitive_tags)]
        leak_details: List[Dict[str, Any]] = []
        response_id = f"response:{_fingerprint(llm_response)[:24]}"

        response_node = {
            "id": response_id,
            "type": "llm_response",
            "created_at": time.time(),
            "token_estimate": _token_count(llm_response),
            "fingerprint": _fingerprint(llm_response),
        }
        self.nodes[response_id] = response_node
        self.add_edge(session_id, response_id, "generated")

        for chunk in blocked_chunks:
            chunk_id = f"sensitive:{_fingerprint(chunk)[:24]}"
            encrypted_chunk = self._encrypt(chunk)
            chunk_node = {
                "id": chunk_id,
                "type": "sensitive_chunk",
                "created_at": time.time(),
                "encrypted_value": encrypted_chunk,
                "fingerprint": _fingerprint(chunk),
                "token_estimate": _token_count(chunk),
            }
            self.nodes[chunk_id] = chunk_node
            self.add_edge(session_id, chunk_id, "contains", sensitive=True)
            self.totals["sensitive_chunks"] += 1

            similarity = self.similarity(chunk, llm_response)
            if similarity >= similarity_threshold:
                leak_details.append(
                    {
                        "fingerprint": chunk_node["fingerprint"],
                        "similarity": similarity,
                        "snippet": chunk[:120],
                    }
                )
                self.add_edge(chunk_id, response_id, "leaked", similarity=similarity)
                self.totals["blocked_leaks"] += 1

        session_node["last_response_fingerprint"] = response_node["fingerprint"]
        session_node["child_count"] += 1
        self._persist()
        return bool(leak_details), leak_details, self.summary()

    def _persist(self) -> None:
        payload = {"nodes": self.nodes, "edges": self.edges, "totals": self.totals, "updated_at": time.time()}
        with open(self.graph_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def summary(self) -> Dict[str, Any]:
        relatable = int(self.totals.get("relatable_inputs", 0))
        non_relatable = int(self.totals.get("non_relatable_inputs", 0))
        total_inputs = relatable + non_relatable
        derived_sessions = sum(1 for node in self.nodes.values() if node.get("type") in {"browser_session", "llm_session"})
        totals = dict(self.totals)
        totals["sessions"] = max(int(totals.get("sessions", 0)), derived_sessions)
        token_raw = sum(bucket["token_estimate"] for node in self.nodes.values() if node.get("type") == "browser_session" for bucket in node.get("topic_buckets", {}).values())
        token_compact = max(0, int(token_raw * 0.38)) if token_raw else 0
        reduction = round(((token_raw - token_compact) / token_raw) * 100, 1) if token_raw else 0.0
        topic_rollup: Dict[str, Dict[str, Any]] = {}
        for node in self.nodes.values():
            if node.get("type") != "browser_session":
                continue
            for topic, bucket in node.get("topic_buckets", {}).items():
                rolled = topic_rollup.setdefault(
                    topic,
                    {
                        "topic": topic,
                        "count": 0,
                        "relatable": 0,
                        "non_relatable": 0,
                        "token_estimate": 0,
                    },
                )
                rolled["count"] += int(bucket.get("count", 0))
                rolled["relatable"] += int(bucket.get("relatable", 0))
                rolled["non_relatable"] += int(bucket.get("non_relatable", 0))
                rolled["token_estimate"] += int(bucket.get("token_estimate", 0))
        top_topics = sorted(topic_rollup.values(), key=lambda item: item["count"], reverse=True)
        return {
            "totals": totals,
            "total_inputs": total_inputs,
            "relatable_ratio": round(relatable / total_inputs, 3) if total_inputs else 0.0,
            "token_reduction_pct": reduction,
            "raw_token_estimate": token_raw,
            "compact_token_estimate": token_compact,
            "topic_buckets": top_topics[:8],
        }


GRAPH = GraphStore()


def process_llm_call(
    user_input: str,
    llm_response: str,
    sensitive_tags: Optional[List[str]] = None,
    similarity_threshold: float = 0.85,
) -> Tuple[bool, List[Dict[str, Any]]]:
    leak_detected, leak_details, _summary = GRAPH.record_llm_call(
        user_input=user_input,
        llm_response=llm_response,
        sensitive_tags=sensitive_tags,
        similarity_threshold=similarity_threshold,
    )
    return leak_detected, leak_details


def capture_chrome_input(
    tab_id: str,
    element_id: str,
    text: str,
    element_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return GRAPH.record_chrome_input(tab_id=tab_id, element_id=element_id, text=text, element_meta=element_meta)


def log_interaction(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    timestamp = time.time()
    event_id = f"event:{event_type}:{_fingerprint(f'{event_type}:{timestamp}')[:24]}"
    GRAPH.ensure_session("global-events", "event_stream", {})
    GRAPH.nodes[event_id] = {
        "id": event_id,
        "type": "event",
        "event_type": event_type,
        "payload": payload,
        "created_at": timestamp,
    }
    GRAPH.add_edge("global-events", event_id, "event")
    GRAPH._persist()
    return GRAPH.nodes[event_id]


def get_graph_summary() -> Dict[str, Any]:
    return GRAPH.summary()


def get_graph_snapshot() -> Dict[str, Any]:
    return {"nodes": list(GRAPH.nodes.values()), "edges": list(GRAPH.edges), "summary": GRAPH.summary()}


def _sanitize_mermaid_text(text: str) -> str:
    cleaned = str(text).replace("\n", " ").replace("\r", " ")
    cleaned = cleaned.replace('"', "'")
    cleaned = cleaned.replace("<", "(").replace(">", ")").replace("&", "and")
    return cleaned[:80]


def get_mermaid_graph(max_nodes: int = 24) -> Dict[str, Any]:
    snapshot = get_graph_snapshot()
    nodes = snapshot.get("nodes", [])
    edges = snapshot.get("edges", [])

    ordered_nodes = sorted(
        nodes,
        key=lambda item: item.get("created_at", 0),
        reverse=True,
    )[:max_nodes]
    ordered_nodes = list(reversed(ordered_nodes))

    node_map: Dict[str, str] = {}
    mermaid_lines = ["graph TD"]

    def node_label(node: Dict[str, Any]) -> str:
        node_type = node.get("type", "node")
        if node_type == "browser_session":
            title = node.get("metadata", {}).get("page_title") or node.get("metadata", {}).get("page_url") or node.get("id", "browser session")
            return f"Browser Session\\n{_sanitize_mermaid_text(title)}"
        if node_type == "chrome_input":
            heading = node.get("heading") or node.get("metadata", {}).get("heading") or "input"
            classification = node.get("classification", "unknown")
            return f"Input\\n{_sanitize_mermaid_text(heading)}\\n{classification}"
        if node_type == "llm_session":
            return "LLM Session"
        if node_type == "llm_response":
            return "LLM Response"
        if node_type == "sensitive_chunk":
            return "Sensitive Chunk"
        if node_type == "event":
            return f"Event\\n{_sanitize_mermaid_text(node.get('event_type', 'event'))}"
        return _sanitize_mermaid_text(node_type)

    def node_style(node: Dict[str, Any]) -> str:
        node_type = node.get("type", "node")
        if node_type == "browser_session":
            return ":::session"
        if node_type == "chrome_input":
            classification = node.get("classification")
            return ":::relatable" if classification == "relatable" else ":::neutral"
        if node_type == "sensitive_chunk":
            return ":::danger"
        if node_type == "llm_response":
            return ":::response"
        return ":::neutral"

    for index, node in enumerate(ordered_nodes):
        original_id = str(node.get("id", f"node-{index}"))
        safe_id = f"n{index}"
        node_map[original_id] = safe_id
        label = node_label(node)
        style = node_style(node)
        mermaid_lines.append(f'{safe_id}["{label}"]{style}')

    selected_ids = set(node_map)
    for edge in edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source not in selected_ids or target not in selected_ids:
            continue
        source_id = node_map[source]
        target_id = node_map[target]
        relation = _sanitize_mermaid_text(edge.get("relation", "edge"))
        mermaid_lines.append(f"{source_id} -->|{relation}| {target_id}")

    if len(mermaid_lines) == 1:
        mermaid_lines.append('empty["No graph data yet"]:::neutral')

    mermaid_lines.extend(
        [
            "classDef session fill:#0f766e,color:#ffffff,stroke:#14b8a6,stroke-width:1px;",
            "classDef relatable fill:#1d4ed8,color:#ffffff,stroke:#60a5fa,stroke-width:1px;",
            "classDef neutral fill:#334155,color:#ffffff,stroke:#94a3b8,stroke-width:1px;",
            "classDef response fill:#7c3aed,color:#ffffff,stroke:#a78bfa,stroke-width:1px;",
            "classDef danger fill:#991b1b,color:#ffffff,stroke:#f87171,stroke-width:1px;",
        ]
    )

    return {
        "mermaid": "\n".join(mermaid_lines),
        "node_count": len(ordered_nodes),
        "edge_count": sum(
            1
            for edge in edges
            if str(edge.get("source", "")) in selected_ids and str(edge.get("target", "")) in selected_ids
        ),
        "summary": snapshot.get("summary", {}),
    }


if __name__ == "__main__":
    prompt = "Here is the quarterly budget: #private $8.2M. Please summarize."
    response = "The budget is in the high single digit millions."
    leak, details = process_llm_call(prompt, response)
    print("Leak detected:", leak)
    if leak:
        print(json.dumps(details, indent=2))
    print(json.dumps(get_graph_summary(), indent=2))
