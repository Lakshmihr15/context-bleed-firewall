"""Privacy graph middleware for the Pacific submission.

This module keeps an encrypted interaction graph of browser inputs and LLM calls,
classifies captured textbox text into relatable versus non-relatable buckets, and
flags semantic bleed when a response paraphrases blocked content.
"""

from __future__ import annotations

import base64
import heapq
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
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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


def _load_local_env_file() -> None:
    env_path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        return


_load_local_env_file()

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

PERSONAL_ROOT_ID = "memory:you"
PERSONAL_ROOT_LABEL = "You"

TREE_RULES: Tuple[Dict[str, Any], ...] = (
    {
        "path": ("Identity", "Personal Details"),
        "signals": ("my name", "i am", "i'm", "age", "birthday", "dob", "email", "phone", "address", "born"),
    },
    {
        "path": ("Politics", "Political Topics", "Elected Officials & Public Office"),
        "signals": (
            "politics",
            "political",
            "election",
            "campaign",
            "government",
            "president",
            "white house",
            "congress",
            "senate",
            "democrat",
            "republican",
            "donald trump",
            "trump",
            "joe biden",
            "biden",
            "kamala harris",
            "harris",
        ),
    },
    {
        "path": ("People", "Named Individuals"),
        "signals": ("who is", "person", "people", "name", "named", "individual", "someone", "someone named"),
    },
    {
        "path": ("Entertainment", "Celebrities & Artists"),
        "signals": ("celebrity", "celebrities", "actor", "actress", "musician", "singer", "artist", "director", "comedian", "influencer"),
    },
    {
        "path": ("Entertainment", "Video Platforms", "YouTube"),
        "signals": ("youtube", "video", "channel", "watch", "upload", "stream", "shorts", "vlog", "tutorial"),
    },
    {
        "path": ("Interests", "General Interests"),
        "signals": ("like", "love", "enjoy", "interest", "favorite", "prefer", "into"),
    },
    {
        "path": ("People", "Family & Friends"),
        "signals": ("friend", "family", "mom", "dad", "sister", "brother", "partner", "roommate", "wife", "husband", "child"),
    },
    {
        "path": ("Entertainment", "Music", "Singers & Artists"),
        "signals": ("spotify", "music", "song", "songs", "singer", "artist", "album", "playlist", "band"),
    },
    {
        "path": ("Entertainment", "Movies & Shows"),
        "signals": ("movie", "film", "series", "show", "netflix", "prime video", "tv"),
    },
    {
        "path": ("Sports", "Teams & Players"),
        "signals": ("sport", "sports", "football", "soccer", "basketball", "cricket", "tennis", "baseball", "nfl", "nba", "player", "team", "match", "game"),
    },
    {
        "path": ("Shopping", "Brands & Products"),
        "signals": ("buy", "purchase", "shopping", "brand", "product", "amazon", "flipkart", "store", "price", "review"),
    },
    {
        "path": ("Travel", "Places & Trips"),
        "signals": ("travel", "trip", "flight", "hotel", "vacation", "tour", "place", "places", "city", "country", "visit"),
    },
    {
        "path": ("Food", "Recipes", "Home Cooking"),
        "signals": ("recipe", "recipes", "cook", "cooking", "bake", "baking", "how to make", "make", "ingredients"),
    },
    {
        "path": ("Food", "Meals & Taste"),
        "signals": ("food", "meal", "restaurant", "cuisine", "snack", "drink", "dinner", "lunch", "breakfast", "pancake", "pancakes", "dessert"),
    },
    {
        "path": ("Work", "Projects & Coding"),
        "signals": ("project", "code", "coding", "python", "javascript", "git", "api", "backend", "frontend", "bug", "intern", "resume"),
    },
    {
        "path": ("Learning", "AI & Study"),
        "signals": ("learn", "study", "course", "tutorial", "ai", "machine learning", "llm", "prompt", "context"),
    },
    {
        "path": ("Life", "Routines"),
        "signals": ("todo", "schedule", "calendar", "habit", "routine", "plan", "shopping", "meal"),
    },
    {
        "path": ("Finance", "Money & Budgeting"),
        "signals": ("budget", "salary", "income", "expense", "invoice", "payment", "cost", "pricing"),
    },
    {
        "path": ("Health", "Wellbeing"),
        "signals": ("health", "workout", "exercise", "sleep", "diet", "doctor", "medicine"),
    },
    {
        "path": ("News", "Current Affairs"),
        "signals": ("news", "headline", "breaking", "current affairs", "update", "politics", "election", "government"),
    },
    {
        "path": ("Arts", "Creativity"),
        "signals": ("art", "drawing", "painting", "design", "creative", "music", "lyrics", "video"),
    },
)

TREE_ROUTE_PROVIDER = os.getenv("TREE_ROUTE_PROVIDER", "heuristic").strip().lower()
TREE_ROUTE_MODEL = os.getenv("TREE_ROUTE_MODEL", "llama3.2:1b").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions").strip()
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "220"))

TREE_ROUTING_SCHEMA = {
    "path": ["Root", "Category", "Subcategory", "Leaf"],
    "required": ["path", "leaf_label", "category", "confidence", "reason"],
    "fields": {
        "path": "Ordered tree path for the new node. Use 2-4 segments.",
        "leaf_label": "Human-readable leaf node label.",
        "category": "Top-level category name like Food, People, Entertainment, Work, Travel.",
        "source_hint": "Optional source platform or origin, such as YouTube or browser search.",
        "intent": "Short classification like recipe_search, named_person, product_lookup, interest_note.",
        "confidence": "Number from 0 to 1.",
        "reason": "Short explanation of the chosen path.",
    },
}

TREE_ROUTING_EXAMPLES = [
    {
        "input": "elon musk",
        "output": {
            "path": ["People", "Named Individuals", "Elon Musk"],
            "leaf_label": "Elon Musk",
            "category": "People",
            "source_hint": "search",
            "intent": "named_person",
            "confidence": 0.97,
            "reason": "A proper name should be stored under named individuals.",
        },
    },
    {
        "input": "pancakes on youtube",
        "output": {
            "path": ["Food", "Recipes", "Pancakes"],
            "leaf_label": "Pancakes",
            "category": "Food",
            "source_hint": "YouTube",
            "intent": "recipe_search",
            "confidence": 0.93,
            "reason": "The main interest is a recipe topic; YouTube is just the source.",
        },
    },
]

ENTITY_STOPWORDS = {
    "a",
    "about",
    "and",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "show",
    "the",
    "to",
    "what",
    "who",
    "why",
    "with",
}

ENTITY_HINT_WORDS = {
    "ceo",
    "founder",
    "person",
    "people",
    "personality",
    "profile",
    "public",
    "actor",
    "actress",
    "musician",
    "singer",
    "creator",
    "artist",
    "director",
    "entrepreneur",
    "inventor",
}


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return cleaned or "node"


def _path_id(path: Sequence[str]) -> str:
    return f"memory:{'/'.join(_slugify(part) for part in path)}"


def _path_label(path: Sequence[str]) -> str:
    return " / ".join(path)


def _compact_text(text: str, limit: int = 80) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    return cleaned[:limit]


def _prompt_candidate(path: Sequence[str], score: float) -> str:
    return f"{_path_label(path)}={round(score, 2)}"


def _build_internal_route_prompt(query: str, candidates: Sequence[Tuple[Sequence[str], float]], limit: int = 4) -> str:
    compact_query = _compact_text(query, 72)
    compact_candidates = ";".join(_prompt_candidate(path, score) for path, score in candidates[:limit])
    return (
        "route|prefer=People / Named Individuals for proper names, "
        "Food / Recipes for cooking and recipe searches, "
        "Entertainment / Video Platforms / YouTube for source platform context, "
        "Politics / Political Topics / Elected Officials & Public Office only for explicitly political content|"
        f"schema={json.dumps(TREE_ROUTING_SCHEMA, ensure_ascii=False)}|"
        f"examples={json.dumps(TREE_ROUTING_EXAMPLES, ensure_ascii=False)}|"
        f"q={compact_query}|c={compact_candidates}"
    )


def _build_route_payload(query: str, candidates: Sequence[Tuple[Sequence[str], float]], limit: int = 4) -> Dict[str, Any]:
    return {
        "query": _compact_text(query, 120),
        "raw_query": str(query).strip(),
        "schema": TREE_ROUTING_SCHEMA,
        "examples": TREE_ROUTING_EXAMPLES,
        "hints": {
            "prefer_named_individuals_for_proper_names": True,
            "prefer_food_recipes_for_recipe_queries": True,
            "treat_source_platform_as_metadata_not_topic": True,
        },
        "candidates": [
            {"path": _path_label(path), "score": round(score, 3)}
            for path, score in candidates[:limit]
        ],
    }


def _coerce_route_path(route_value: Any) -> List[str]:
    if isinstance(route_value, (list, tuple)):
        return [str(part).strip() for part in route_value if str(part).strip()]
    if isinstance(route_value, str):
        parts = [part.strip() for part in route_value.split("/") if part.strip()]
        if parts:
            return parts
    return []


def _coerce_router_result(raw_result: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_result, dict):
        return None
    path = _coerce_route_path(
        raw_result.get("path")
        or raw_result.get("branch_path")
        or raw_result.get("tree_path")
        or raw_result.get("category_path")
    )
    if not path:
        return None
    leaf_label = str(raw_result.get("leaf_label") or raw_result.get("leaf") or path[-1]).strip()
    if len(path) >= 2 and path[-1].lower() != leaf_label.lower():
        path = [*path[:-1], leaf_label]
    elif len(path) < 3:
        path = [*path, leaf_label]
    return {
        "path": path,
        "leaf_label": leaf_label,
        "category": str(raw_result.get("category") or path[0]),
        "source_hint": str(raw_result.get("source_hint") or raw_result.get("source") or ""),
        "intent": str(raw_result.get("intent") or raw_result.get("query_type") or ""),
        "confidence": float(raw_result.get("confidence", 0.0)),
        "reason": str(raw_result.get("reason") or raw_result.get("explanation") or ""),
    }


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = cleaned[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _call_ollama_router(route_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if TREE_ROUTE_PROVIDER not in {"ollama", "open-source", "opensource"}:
        return None
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    prompt = (
        "You are building the memory tree. Return only valid JSON, no markdown, no code fences. "
        "Do not use heuristics or guesswork from the application. Use the provided schema and examples only. "
        "If the query mentions a source platform like YouTube, store it as source_hint or metadata, not the leaf topic. "
        "If the query is a proper name, create a named-person leaf. If it is a recipe or food term, create a food/recipe leaf. "
        "Return the deepest appropriate path from the schema and examples.\n"
        f"INPUT={json.dumps(route_payload, ensure_ascii=False)}"
    )
    body = {
        "model": TREE_ROUTE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 80},
    }
    request = Request(
        f"{host}/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=8) as response:
            raw = response.read().decode("utf-8")
        payload = json.loads(raw)
        text = payload.get("response", "")
        parsed = _extract_json_object(text) or {}
        coerced = _coerce_router_result(parsed)
        if coerced is not None:
            return coerced
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return None
    return None


def _call_ollama_tree_constructor(query: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    payload = {
        "query": _compact_text(query, 120),
        "raw_query": str(query).strip(),
        "metadata": {
            "page_title": (metadata or {}).get("page_title"),
            "page_url": (metadata or {}).get("page_url"),
            "heading": (metadata or {}).get("heading"),
            "field_type": (metadata or {}).get("field_type"),
        },
        "schema": TREE_ROUTING_SCHEMA,
        "examples": TREE_ROUTING_EXAMPLES,
        "instructions": [
            "Return JSON only.",
            "No heuristics from the app.",
            "Use a deep path of 2 to 4 segments.",
            "Source platforms are metadata, not the topic.",
            "Create a new leaf when the query names a specific person, recipe, product, show, or interest.",
        ],
    }
    route_payload = _build_route_payload(query, [])
    route_payload.update(payload)
    result = _call_ollama_router(route_payload)
    if result is not None:
        return result
    retry_payload = dict(route_payload)
    retry_payload["instructions"] = [
        "Retry: output valid JSON only.",
        "Do not include markdown, commentary, or code fences.",
        "Return a deep path with 2 to 4 segments.",
        "Ensure path is an array of strings, not a string.",
        "Treat source platforms as metadata, not the topic.",
    ]
    result = _call_ollama_router(retry_payload)
    if result is not None:
        return result
    return None


def _call_transformers_router(route_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if TREE_ROUTE_PROVIDER not in {"transformers", "hf", "huggingface"}:
        return None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
    except Exception:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(TREE_ROUTE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(TREE_ROUTE_MODEL)
        router = pipeline("text-generation", model=model, tokenizer=tokenizer)
        prompt = (
            "Return only JSON with keys path, leaf_label, category, source_hint, intent, confidence, reason. "
            f"Choose one from: {json.dumps(route_payload, ensure_ascii=False)}"
        )
        result = router(prompt, max_new_tokens=80, do_sample=False, temperature=0.0)
        text = result[0]["generated_text"] if isinstance(result, list) and result else ""
        parsed = _extract_json_object(text) or {}
        coerced = _coerce_router_result(parsed)
        if coerced is not None:
            return coerced
    except Exception:
        return None
    return None


def _route_with_open_source_llm(query: str, candidates: Sequence[Tuple[Sequence[str], float]]) -> Optional[Tuple[List[str], Dict[str, Any]]]:
    route_payload = _build_route_payload(query, candidates)
    result = _call_ollama_router(route_payload)
    if result is None:
        result = _call_transformers_router(route_payload)
    if result is None:
        return None
    requested_path = _coerce_route_path(result.get("path", []))
    if not requested_path:
        return None
    return requested_path, {
        "route_source": TREE_ROUTE_PROVIDER if TREE_ROUTE_PROVIDER != "heuristic" else "open_source_llm",
        "route_payload": route_payload,
        "route_decision": {
            "path": _path_label(requested_path),
            "leaf_label": result.get("leaf_label", requested_path[-1]),
            "category": result.get("category", requested_path[0]),
            "source_hint": result.get("source_hint", ""),
            "intent": result.get("intent", ""),
            "confidence": float(result.get("confidence", 0.0)),
            "reason": result.get("reason", ""),
            "score": 0.0,
        },
    }


def _text_haystack(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    metadata = metadata or {}
    pieces = [
        text or "",
        str(metadata.get("page_title") or ""),
        str(metadata.get("page_url") or ""),
        str(metadata.get("heading") or ""),
        str(metadata.get("field_type") or ""),
    ]
    return " ".join(pieces).lower()


def _entity_leaf_label(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip(" \t\n\r\f\v.,;:!?()[]{}")
    if not cleaned:
        return "Unknown Interest"
    words = [word for word in cleaned.split() if word]
    normalized_words: List[str] = []
    for word in words[:5]:
        if word.lower() in ENTITY_STOPWORDS:
            continue
        if len(word) <= 2 and word.isalpha():
            normalized_words.append(word.upper())
        elif word.isupper():
            normalized_words.append(word)
        else:
            normalized_words.append(word[:1].upper() + word[1:])
    if not normalized_words:
        normalized_words = [word[:1].upper() + word[1:] for word in words[:3]]
    return " ".join(normalized_words)[:64]


def _looks_like_named_entity(text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    haystack = _text_haystack(text, metadata)
    tokens = [token for token in _tokenize(text) if token not in ENTITY_STOPWORDS]
    if not tokens:
        return False
    if any(keyword in haystack for keyword in ENTITY_HINT_WORDS):
        return True
    if any(signal in haystack for signal in ("who is", "tell me about", "search", "look up")):
        return len(tokens) <= 6
    topic_tokens = {word for keywords in RELATABLE_TOPICS.values() for word in keywords}
    if len(tokens) <= 4 and all(token not in topic_tokens for token in tokens):
        return any(len(token) > 2 for token in tokens)
    if len(tokens) <= 2 and all(token.isalpha() for token in tokens):
        return True
    return False


def _infer_tree_path(text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
    selected_path, _routing_context = _select_tree_path(text, metadata)
    return selected_path


def _heuristic_tree_route(text: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[List[str], Dict[str, Any]]:
    metadata = metadata or {}
    haystack = _text_haystack(text, metadata)
    source_hint = str(metadata.get("heading") or metadata.get("page_title") or "").strip()
    intent = "general_note"
    reason = "Fallback heuristic route used because Ollama was unavailable."

    if any(signal in haystack for signal in ("politics", "political", "election", "campaign", "government", "president", "congress", "senate", "white house")):
        path = ["Politics", "Political Topics", "Elected Officials & Public Office"]
        intent = "political_content"
        reason = "Explicit political language routes to politics."
    elif any(signal in haystack for signal in ("recipe", "recipes", "cook", "cooking", "bake", "baking", "ingredients", "how to make")):
        path = ["Food", "Recipes", _entity_leaf_label(text)]
        intent = "recipe_search"
        reason = "Recipe or cooking language routes to food recipes."
    elif any(signal in haystack for signal in ("youtube", "video", "channel", "watch", "upload", "stream", "shorts", "vlog", "tutorial")):
        path = ["Entertainment", "Video Platforms", "YouTube"]
        intent = "video_source"
        reason = "Video-platform language routes to YouTube."
    elif _looks_like_named_entity(text, metadata):
        path = ["People", "Named Individuals", _entity_leaf_label(text)]
        intent = "named_person"
        reason = "Short proper-name style text routes to named individuals."
    else:
        topic = _derive_topic(text, str(metadata.get("heading") or metadata.get("page_title") or "") or None)
        if topic == "finance":
            path = ["Finance", "Money & Budgeting"]
            intent = "finance_topic"
        elif topic == "product":
            path = ["Shopping", "Brands & Products"]
            intent = "product_lookup"
        elif topic in {"people_ops", "operations", "support"}:
            path = ["Work", "Projects & Coding"]
            intent = f"{topic}_topic"
        elif topic == "research":
            path = ["Learning", "AI & Study"]
            intent = "study_topic"
        elif topic == "non_relatable":
            path = ["Life", "Routines"]
            intent = "non_relatable"
        else:
            path = ["Interests", "General Interests"]

    route_json = {
        "query": _compact_text(text, 120),
        "path": _path_label(path),
        "leaf_label": path[-1],
        "category": path[0],
        "source_hint": source_hint,
        "intent": intent,
        "reason": reason,
        "ranked_candidates": [],
        "route_source": "heuristic_fallback",
    }
    routing_context = {
        "prompt": _build_internal_route_prompt(text, []),
        "selected_path": _path_label(path),
        "candidates": [],
        "route_json": route_json,
        "route_source": "heuristic_fallback",
        "route_decision": {
            "path": _path_label(path),
            "leaf_label": path[-1],
            "category": path[0],
            "source_hint": source_hint,
            "intent": intent,
            "confidence": 0.58,
            "reason": reason,
            "score": 0.0,
        },
    }
    return path, routing_context


def _path_overlap_score(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = 0
    for left_part, right_part in zip(left, right):
        if _slugify(left_part) == _slugify(right_part):
            overlap += 1
        else:
            break
    return overlap / max(len(left), len(right))


def _first_sentence(text: str, limit: int = 120) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if not cleaned:
        return ""
    sentence = cleaned.split(".")[0].strip()
    return sentence[:limit]


def _time_bucket(timestamp: Optional[float] = None) -> str:
    tm = time.localtime(timestamp or time.time())
    hour = tm.tm_hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def _bucket_histogram_score(histogram: Dict[str, Any], bucket: str) -> float:
    if not histogram:
        return 0.0
    total = sum(int(value) for value in histogram.values())
    if total <= 0:
        return 0.0
    return int(histogram.get(bucket, 0)) / total


def _path_key(path: Sequence[str]) -> str:
    return _path_label(path)


def _select_tree_path(text: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[List[str], Dict[str, Any]]:
    llm_route = _call_ollama_tree_constructor(text, metadata)
    if llm_route is None:
        return _heuristic_tree_route(text, metadata)
    selected_path = list(llm_route[0])
    if len(selected_path) < 2:
        return _heuristic_tree_route(text, metadata)
    if len(selected_path) >= 2 and selected_path[0] == "People" and selected_path[1] == "Named Individuals" and len(selected_path) == 2:
        selected_path = [*selected_path, _entity_leaf_label(text)]
    if len(selected_path) >= 2 and selected_path[0] == "Food" and selected_path[1] == "Recipes" and len(selected_path) == 2:
        selected_path = [*selected_path, _entity_leaf_label(text)]
    route_json = {
        "query": _compact_text(text, 120),
        "path": _path_label(selected_path),
        "leaf_label": selected_path[-1] if selected_path else "node",
        "category": selected_path[0] if selected_path else "root",
        "source_hint": llm_route[1].get("route_decision", {}).get("source_hint", ""),
        "intent": llm_route[1].get("route_decision", {}).get("intent", ""),
        "reason": llm_route[1].get("route_decision", {}).get("reason", ""),
        "ranked_candidates": [],
        "route_source": llm_route[1].get("route_source", "ollama"),
    }
    routing_context = {
        "prompt": _build_internal_route_prompt(text, []),
        "selected_path": _path_label(selected_path),
        "candidates": [],
        "route_json": route_json,
        "route_source": route_json["route_source"],
        "route_decision": llm_route[1].get("route_decision", {"path": _path_label(selected_path), "score": 0.0}),
    }
    return selected_path, routing_context

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
    recency_heap: List[Tuple[float, int, str]] = field(default_factory=list)
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
        self._recency_counter = 0
        self._load()
        self._ensure_personal_root()
        self._migrate_legacy_inputs()
        self._persist()

    def _next_recency_stamp(self) -> int:
        self._recency_counter += 1
        return self._recency_counter

    def _touch_node(self, node_id: str, ts: Optional[float] = None) -> None:
        node = self.nodes.get(node_id)
        if node is None:
            return
        timestamp = ts if ts is not None else time.time()
        counter = self._next_recency_stamp()
        node["last_used_at"] = timestamp
        node["recency_counter"] = counter
        heapq.heappush(self.recency_heap, (timestamp, counter, node_id))

    def _recent_nodes(self, node_type_prefix: Optional[str] = None, limit: int = 24) -> List[Dict[str, Any]]:
        recency_items: List[Tuple[float, int, str]] = []
        for node_id, node in self.nodes.items():
            if node_type_prefix and not str(node.get("type", "")).startswith(node_type_prefix):
                continue
            ts = float(node.get("last_used_at") or node.get("created_at") or 0.0)
            counter = int(node.get("recency_counter") or 0)
            recency_items.append((ts, counter, node_id))

        while self.recency_heap:
            ts, counter, node_id = self.recency_heap[0]
            node = self.nodes.get(node_id)
            if node is None:
                heapq.heappop(self.recency_heap)
                continue
            current_ts = float(node.get("last_used_at") or node.get("created_at") or 0.0)
            current_counter = int(node.get("recency_counter") or 0)
            if current_ts != ts or current_counter != counter:
                heapq.heappop(self.recency_heap)
                continue
            break

        top_recent = heapq.nlargest(limit, recency_items, key=lambda item: (item[0], item[1]))
        return [self.nodes[node_id] for _ts, _counter, node_id in top_recent]

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

    def reset(self) -> None:
        self._bootstrap_default_nodes()
        self.recency_heap = []
        self._recency_counter = 0
        self._ensure_personal_root()
        self._persist()

    def _ensure_personal_root(self) -> Dict[str, Any]:
        root = self.nodes.get(PERSONAL_ROOT_ID)
        if root is None:
            root = {
                "id": PERSONAL_ROOT_ID,
                "type": "memory_root",
                "label": PERSONAL_ROOT_LABEL,
                "memory_path": [PERSONAL_ROOT_LABEL],
                "created_at": time.time(),
                "child_count": 0,
                "evidence_count": 0,
                "relatable_count": 0,
                "non_relatable_count": 0,
                "token_estimate": 0,
                "time_histogram": {},
                "query_histogram": {},
                "last_query_bucket": None,
                "last_query_at": None,
            }
            self.nodes[PERSONAL_ROOT_ID] = root
        else:
            root.setdefault("type", "memory_root")
            root.setdefault("label", PERSONAL_ROOT_LABEL)
            root.setdefault("memory_path", [PERSONAL_ROOT_LABEL])
            root.setdefault("time_histogram", {})
            root.setdefault("query_histogram", {})
            root.setdefault("last_query_bucket", None)
            root.setdefault("last_query_at", None)
        self._touch_node(PERSONAL_ROOT_ID)
        return root

    def _ensure_tree_node(self, path: Sequence[str], node_type: str = "memory_branch") -> Dict[str, Any]:
        self._ensure_personal_root()
        current_parent_id = PERSONAL_ROOT_ID
        current_path: List[str] = [PERSONAL_ROOT_LABEL]
        node = self.nodes[PERSONAL_ROOT_ID]

        for depth, segment in enumerate(path, start=1):
            current_path.append(segment)
            node_id = _path_id(current_path)
            created = False
            node = self.nodes.get(node_id)
            if node is None:
                node = {
                    "id": node_id,
                    "type": node_type if depth < len(path) else node_type,
                    "label": segment,
                    "memory_path": list(current_path),
                    "depth": depth,
                    "created_at": time.time(),
                    "child_count": 0,
                    "evidence_count": 0,
                    "relatable_count": 0,
                    "non_relatable_count": 0,
                    "token_estimate": 0,
                    "time_histogram": {},
                    "query_histogram": {},
                    "last_query_bucket": None,
                    "last_query_at": None,
                }
                self.nodes[node_id] = node
                created = True
            else:
                node.setdefault("label", segment)
                node.setdefault("memory_path", list(current_path))
                node.setdefault("depth", depth)
                node.setdefault("time_histogram", {})
                node.setdefault("query_histogram", {})
                node.setdefault("last_query_bucket", None)
                node.setdefault("last_query_at", None)
            if created or not any(edge.get("source") == current_parent_id and edge.get("target") == node_id and edge.get("relation") == "contains" for edge in self.edges):
                self.add_edge(current_parent_id, node_id, "contains")
            self.nodes[current_parent_id]["child_count"] = int(self.nodes[current_parent_id].get("child_count", 0)) + (1 if created else 0)
            self._touch_node(node_id)
            current_parent_id = node_id

        return node

    def _migrate_legacy_inputs(self) -> None:
        legacy_inputs = [node for node in self.nodes.values() if node.get("type") == "chrome_input" and not node.get("memory_path")]
        if not legacy_inputs:
            return
        for legacy_node in legacy_inputs:
            metadata = legacy_node.get("metadata") or {}
            legacy_text = self._decrypt(legacy_node.get("encrypted_value", "")) if legacy_node.get("encrypted_value") else ""
            path = _infer_tree_path(legacy_text, metadata)
            leaf = self._ensure_tree_node(path)
            legacy_node["legacy_type"] = "chrome_input"
            legacy_node["type"] = "memory_evidence"
            legacy_node["memory_path"] = [PERSONAL_ROOT_LABEL, *path]
            legacy_node["parent_id"] = leaf["id"]
            legacy_node["display_label"] = metadata.get("heading") or metadata.get("page_title") or legacy_node.get("heading") or "Captured memory"
            if not any(edge.get("source") == leaf["id"] and edge.get("target") == legacy_node["id"] and edge.get("relation") == "evidence" for edge in self.edges):
                self.add_edge(leaf["id"], legacy_node["id"], "evidence")
            self._touch_node(leaf["id"], float(legacy_node.get("created_at", time.time())))
            self._touch_node(legacy_node["id"], float(legacy_node.get("created_at", time.time())))

    def _update_tree_stats(self, path: Sequence[str], classification: str, token_estimate: int, timestamp: float) -> None:
        self._ensure_personal_root()
        bucket = _time_bucket(timestamp)
        self.nodes[PERSONAL_ROOT_ID]["child_count"] = int(self.nodes[PERSONAL_ROOT_ID].get("child_count", 0))
        current_path: List[str] = [PERSONAL_ROOT_LABEL]
        current_node = self.nodes[PERSONAL_ROOT_ID]
        current_node["evidence_count"] = int(current_node.get("evidence_count", 0)) + 1
        current_node["token_estimate"] = int(current_node.get("token_estimate", 0)) + token_estimate
        if classification == "relatable":
            current_node["relatable_count"] = int(current_node.get("relatable_count", 0)) + 1
        else:
            current_node["non_relatable_count"] = int(current_node.get("non_relatable_count", 0)) + 1
        current_node["last_seen"] = timestamp
        current_node.setdefault("time_histogram", {})[bucket] = int(current_node.get("time_histogram", {}).get(bucket, 0)) + 1
        self._touch_node(PERSONAL_ROOT_ID, timestamp)

        for segment in path:
            current_path.append(segment)
            node_id = _path_id(current_path)
            node = self.nodes.get(node_id)
            if node is None:
                continue
            node["evidence_count"] = int(node.get("evidence_count", 0)) + 1
            node["token_estimate"] = int(node.get("token_estimate", 0)) + token_estimate
            if classification == "relatable":
                node["relatable_count"] = int(node.get("relatable_count", 0)) + 1
            else:
                node["non_relatable_count"] = int(node.get("non_relatable_count", 0)) + 1
            node["last_seen"] = timestamp
            node.setdefault("time_histogram", {})[bucket] = int(node.get("time_histogram", {}).get(bucket, 0)) + 1
            self._touch_node(node_id, timestamp)

    def _record_query_usage(self, path: Sequence[str], query: str, timestamp: Optional[float] = None) -> None:
        if not path:
            return
        timestamp = timestamp or time.time()
        bucket = _time_bucket(timestamp)
        query_tokens = _tokenize(query)
        current_path: List[str] = [PERSONAL_ROOT_LABEL]

        root = self.nodes.get(PERSONAL_ROOT_ID)
        if root is not None:
            root.setdefault("query_histogram", {})[bucket] = int(root.get("query_histogram", {}).get(bucket, 0)) + 1
            root["last_query_bucket"] = bucket
            root["last_query_at"] = timestamp
            root["last_query_tokens"] = query_tokens[:12]

        for segment in path:
            current_path.append(segment)
            node_id = _path_id(current_path)
            node = self.nodes.get(node_id)
            if node is None:
                continue
            node.setdefault("query_histogram", {})[bucket] = int(node.get("query_histogram", {}).get(bucket, 0)) + 1
            node["last_query_bucket"] = bucket
            node["last_query_at"] = timestamp
            node["last_query_tokens"] = query_tokens[:12]
            self._touch_node(node_id, timestamp)

    def _collect_evidence_nodes(self) -> List[Dict[str, Any]]:
        return [node for node in self.nodes.values() if node.get("type") == "memory_evidence"]

    def _lookup_tree_node(self, path: Sequence[str]) -> Optional[Dict[str, Any]]:
        target = [PERSONAL_ROOT_LABEL, *path]
        for node in self.nodes.values():
            if node.get("memory_path") == target:
                return node
        return None

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
        memory_path, routing_context = _select_tree_path(text, meta)
        timestamp = time.time()
        leaf_node = self._ensure_tree_node(memory_path)
        input_id = f"memory:{_fingerprint(f'{tab_id}:{element_id}:{timestamp}:{text}')[:24]}"
        encrypted_value = self._encrypt(text)
        token_estimate = _token_count(text)

        node = {
            "id": input_id,
            "type": "memory_evidence",
            "legacy_type": "chrome_input",
            "tab_id": tab_id,
            "element_id": element_id,
            "heading": meta.get("heading") or meta.get("placeholder") or meta.get("name") or _path_label(memory_path),
            "label": meta.get("page_title") or meta.get("heading") or _first_sentence(text, 48) or _path_label(memory_path),
            "classification": classification,
            "confidence": confidence,
            "encrypted_value": encrypted_value,
            "fingerprint": _fingerprint(text),
            "token_estimate": token_estimate,
            "created_at": timestamp,
            "metadata": meta,
            "memory_path": [PERSONAL_ROOT_LABEL, *memory_path],
            "parent_id": leaf_node["id"],
            "routing_prompt": routing_context["prompt"],
            "routing_candidates": routing_context["candidates"],
            "route_json": routing_context["route_json"],
            "route_source": routing_context.get("route_source", "heuristic"),
            "route_decision": routing_context.get("route_decision", {}),
            "source": {
                "page_title": meta.get("page_title"),
                "page_url": meta.get("page_url"),
                "field_type": meta.get("field_type"),
            },
        }
        self.nodes[input_id] = node
        self.add_edge(leaf_node["id"], input_id, "evidence")
        self._update_tree_stats(memory_path, classification, token_estimate, timestamp)
        self.totals["chrome_inputs"] += 1
        if classification == "relatable":
            self.totals["relatable_inputs"] += 1
        else:
            self.totals["non_relatable_inputs"] += 1

        self._persist()
        return {
            "input_id": input_id,
            "topic": _path_label(memory_path),
            "classification": classification,
            "confidence": confidence,
            "token_estimate": token_estimate,
            "graph_node": leaf_node["id"],
            "routing_prompt": routing_context["prompt"],
            "route_json": routing_context["route_json"],
            "route_source": routing_context.get("route_source", "heuristic"),
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
        evidence_nodes = self._collect_evidence_nodes()
        relatable = sum(1 for node in evidence_nodes if node.get("classification") == "relatable")
        non_relatable = sum(1 for node in evidence_nodes if node.get("classification") != "relatable")
        total_inputs = len(evidence_nodes)
        totals = dict(self.totals)
        totals["sessions"] = max(int(totals.get("sessions", 0)), 1 if self.nodes.get(PERSONAL_ROOT_ID) else 0)
        totals["chrome_inputs"] = total_inputs
        totals["relatable_inputs"] = relatable
        totals["non_relatable_inputs"] = non_relatable
        token_raw = sum(int(node.get("token_estimate", 0)) for node in evidence_nodes)
        token_compact = max(0, int(token_raw * 0.38)) if token_raw else 0
        reduction = round(((token_raw - token_compact) / token_raw) * 100, 1) if token_raw else 0.0
        branch_rollup: Dict[str, Dict[str, Any]] = {}
        root = self.nodes.get(PERSONAL_ROOT_ID)
        if root:
            branch_rollup[PERSONAL_ROOT_LABEL] = {
                "topic": PERSONAL_ROOT_LABEL,
                "path": [PERSONAL_ROOT_LABEL],
                "count": total_inputs,
                "relatable": relatable,
                "non_relatable": non_relatable,
                "token_estimate": token_raw,
                "last_used_at": float(root.get("last_used_at") or root.get("created_at") or 0.0),
            }
        for node in evidence_nodes:
            path = node.get("memory_path") or [PERSONAL_ROOT_LABEL]
            if len(path) < 2:
                continue
            branch_key = _path_label(path[1:3] if len(path) > 2 else path[1:])
            rolled = branch_rollup.setdefault(
                branch_key,
                {
                    "topic": branch_key,
                    "path": [*path[: min(len(path), 3)]],
                    "count": 0,
                    "relatable": 0,
                    "non_relatable": 0,
                    "token_estimate": 0,
                    "last_used_at": 0.0,
                },
            )
            rolled["count"] += 1
            rolled["last_used_at"] = max(rolled["last_used_at"], float(node.get("last_used_at") or node.get("created_at") or 0.0))
            if node.get("classification") == "relatable":
                rolled["relatable"] += 1
            else:
                rolled["non_relatable"] += 1
            rolled["token_estimate"] += int(node.get("token_estimate", 0))
        top_topics = sorted(branch_rollup.values(), key=lambda item: (item["last_used_at"], item["count"]), reverse=True)
        return {
            "totals": totals,
            "total_inputs": total_inputs,
            "relatable_ratio": round(relatable / total_inputs, 3) if total_inputs else 0.0,
            "token_reduction_pct": reduction,
            "raw_token_estimate": token_raw,
            "compact_token_estimate": token_compact,
            "topic_buckets": top_topics[:8],
            "root_label": PERSONAL_ROOT_LABEL,
            "memory_depth": max((len(node.get("memory_path", [])) for node in evidence_nodes), default=1),
            "recent_bucket": max((node.get("last_used_at") or node.get("created_at") or 0.0 for node in evidence_nodes), default=0.0),
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
    nodes = GRAPH._recent_nodes(limit=len(GRAPH.nodes))
    return {"nodes": nodes, "edges": list(GRAPH.edges), "summary": GRAPH.summary()}


def get_personal_context(query: str, limit: int = 6) -> Dict[str, Any]:
    query = query or ""
    query_path = _infer_tree_path(query)
    current_time = time.time()
    current_bucket = _time_bucket(current_time)
    evidence_nodes = GRAPH._collect_evidence_nodes()

    ranked: List[Dict[str, Any]] = []
    query_tokens = set(_tokenize(query))
    for node in evidence_nodes:
        path = node.get("memory_path") or [PERSONAL_ROOT_LABEL]
        score = _path_overlap_score(query_path, path[1:] if len(path) > 1 else [])
        label_parts = [str(node.get("label") or ""), str(node.get("heading") or ""), str(node.get("metadata", {}).get("page_title") or ""), str(node.get("metadata", {}).get("page_url") or "")]
        label_tokens = set(_tokenize(" ".join(label_parts)))
        token_score = len(query_tokens & label_tokens) / max(len(query_tokens | label_tokens), 1)
        score = max(score, token_score)
        if any(part in query.lower() for part in [segment.lower() for segment in path]):
            score = max(score, 0.8)
        branch_node = GRAPH._lookup_tree_node(path[1:] if len(path) > 1 else [])
        if branch_node is not None:
            score += _bucket_histogram_score(branch_node.get("time_histogram", {}), current_bucket) * 1.15
            score += _bucket_histogram_score(branch_node.get("query_histogram", {}), current_bucket) * 0.9
            last_used_at = float(branch_node.get("last_used_at") or branch_node.get("created_at") or 0.0)
            score += 1.0 / (1.0 + max(0.0, current_time - last_used_at)) * 0.4
        ranked.append({"score": round(score, 3), "node": node})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    matched = [item for item in ranked[:limit] if item["score"] > 0]
    relevant_nodes = [item["node"] for item in matched]
    branch_label = _path_label(query_path)
    if query_path == ["Interests", "General Interests"] and relevant_nodes:
        inferred_paths = [node.get("memory_path") or [PERSONAL_ROOT_LABEL] for node in relevant_nodes]
        branch_label = _path_label(inferred_paths[0][1:]) if len(inferred_paths[0]) > 1 else branch_label

    if not relevant_nodes and not query.strip():
        relevant_nodes = evidence_nodes[:limit]

    evidence = []
    for item in relevant_nodes:
        decrypted_value = ""
        if item.get("encrypted_value"):
            try:
                decrypted_value = GRAPH._decrypt(item["encrypted_value"])
            except Exception:
                decrypted_value = ""
        GRAPH._touch_node(item["id"], time.time())
        evidence.append(
            {
                "id": item.get("id"),
                "path": item.get("memory_path", []),
                "label": item.get("label") or item.get("heading") or "memory",
                "classification": item.get("classification", "unknown"),
                "confidence": item.get("confidence", 0),
                "page_title": item.get("source", {}).get("page_title") or item.get("metadata", {}).get("page_title"),
                "page_url": item.get("source", {}).get("page_url") or item.get("metadata", {}).get("page_url"),
                "field_type": item.get("source", {}).get("field_type") or item.get("metadata", {}).get("field_type"),
                "preview": _first_sentence(decrypted_value, 140),
            }
        )

    top_branches: Dict[str, Dict[str, Any]] = {}
    for node in evidence_nodes:
        path = node.get("memory_path") or [PERSONAL_ROOT_LABEL]
        branch_key = _path_label(path[1:3] if len(path) > 2 else path[1:]) if len(path) > 1 else PERSONAL_ROOT_LABEL
        branch = top_branches.setdefault(
            branch_key,
            {"path": path[:3] if len(path) > 1 else [PERSONAL_ROOT_LABEL], "count": 0, "samples": [], "last_used_at": 0.0},
        )
        branch["count"] += 1
        branch["last_used_at"] = max(branch["last_used_at"], float(node.get("last_used_at") or node.get("created_at") or 0.0))
        if len(branch["samples"]) < 3:
            branch["samples"].append(node.get("label") or node.get("heading") or "memory")

    top_branch_list = sorted(top_branches.items(), key=lambda item: (item[1]["last_used_at"], item[1]["count"]), reverse=True)
    routed = bool(relevant_nodes and (matched[0]["score"] >= 0.2 or any(token in query.lower() for token in ("my ", "me", "mine", "what do i know", "remember", "favorite", "liked", "prefer"))))
    if routed:
        GRAPH._record_query_usage(query_path, query, current_time)
        if relevant_nodes:
            GRAPH._touch_node(PERSONAL_ROOT_ID, current_time)
    summary = GRAPH.summary()

    return {
        "route_json": {
            "query": query,
            "time_bucket": current_bucket,
            "path": query_path,
            "matched_branch": branch_label,
            "top_branch": top_branch_list[0][0] if top_branch_list else branch_label,
        },
        "query": query,
        "is_personal": routed,
        "query_path": query_path,
        "matched_branch": branch_label,
        "time_bucket": current_bucket,
        "evidence": evidence,
        "top_branches": [
            {"path": value["path"], "count": value["count"], "samples": value["samples"], "last_used_at": value["last_used_at"]}
            for _, value in top_branch_list[:8]
        ],
        "summary": summary,
    }


def _build_openai_payload(query: str, personal_context: Dict[str, Any]) -> Dict[str, Any]:
    compact_context = {
        "route": personal_context.get("route_json", {}),
        "evidence": personal_context.get("evidence", [])[:6],
        "top_branches": personal_context.get("top_branches", [])[:6],
        "summary": {
            "total_inputs": personal_context.get("summary", {}).get("total_inputs", 0),
            "relatable_ratio": personal_context.get("summary", {}).get("relatable_ratio", 0),
            "memory_depth": personal_context.get("summary", {}).get("memory_depth", 0),
            "recent_bucket": personal_context.get("summary", {}).get("recent_bucket", 0),
        },
    }
    system_prompt = (
        "You are a personalized assistant. Use the JSON context only. "
        "Answer briefly, naturally, and grounded in the tree. "
        "Do not invent memory that is not present."
    )
    user_prompt = json.dumps({"query": query, "context": compact_context}, ensure_ascii=False)
    return {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": OPENAI_MAX_OUTPUT_TOKENS,
    }


def generate_personal_answer(query: str, limit: int = 6) -> Dict[str, Any]:
    personal_context = get_personal_context(query, limit=limit)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    payload = _build_openai_payload(query, personal_context)

    if api_key:
        request = Request(
            OPENAI_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=20) as response:
                raw = response.read().decode("utf-8")
            api_payload = json.loads(raw)
            answer = ""
            if api_payload.get("choices"):
                answer = api_payload["choices"][0].get("message", {}).get("content", "")
            return {
                "answer": answer.strip() or _build_personal_fallback_answer(query, personal_context),
                "source": "openai",
                "model": OPENAI_MODEL,
                "route_json": personal_context.get("route_json", {}),
                "personal_context": personal_context,
            }
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
            pass

    return {
        "answer": _build_personal_fallback_answer(query, personal_context),
        "source": "fallback",
        "model": OPENAI_MODEL,
        "route_json": personal_context.get("route_json", {}),
        "personal_context": personal_context,
    }


def _build_personal_fallback_answer(query: str, personal_context: Dict[str, Any]) -> str:
    evidence = personal_context.get("evidence", []) or []
    top_branches = personal_context.get("top_branches", []) or []
    branch = personal_context.get("matched_branch", "your memory tree")
    evidence_bits = []
    for item in evidence[:3]:
        path = item.get("path") or []
        label = item.get("label") or "memory"
        preview = item.get("preview") or label
        evidence_bits.append(f"{_path_label(path[1:]) if len(path) > 1 else label}: {preview}")
    branch_bits = []
    for item in top_branches[:3]:
        path = item.get("path") or []
        branch_bits.append(_path_label(path[1:]) if len(path) > 1 else branch)
    branch_text = "; ".join(branch_bits) if branch_bits else branch
    evidence_text = " | ".join(evidence_bits) if evidence_bits else "No direct evidence yet."
    return (
        f"I routed this through your personal tree first. Matched branch: {branch}. "
        f"Strongest branches: {branch_text}. "
        f"Grounding evidence: {evidence_text}. "
        f"Answer this query using the tree instead of generic context: {query.strip() or 'this question'}."
    )


def _sanitize_mermaid_text(text: str) -> str:
    cleaned = str(text).replace("\n", " ").replace("\r", " ")
    cleaned = cleaned.replace('"', "'")
    cleaned = cleaned.replace("<", "(").replace(">", ")").replace("&", "and")
    return cleaned[:80]


def get_mermaid_graph(max_nodes: int = 80) -> Dict[str, Any]:
    snapshot = get_graph_snapshot()
    nodes = snapshot.get("nodes", [])
    edges = snapshot.get("edges", [])

    node_lookup = {str(node.get("id")): node for node in nodes if node.get("id")}
    root_node = node_lookup.get(PERSONAL_ROOT_ID)
    memory_nodes = [node for node in nodes if str(node.get("type", "")).startswith("memory_") and node.get("id") != PERSONAL_ROOT_ID]
    evidence_nodes = [node for node in memory_nodes if node.get("type") == "memory_evidence"]
    branch_nodes = [node for node in memory_nodes if node.get("type") != "memory_evidence"]

    def recency_key(node: Dict[str, Any]) -> Tuple[float, int]:
        return (
            float(node.get("last_used_at") or node.get("created_at") or 0.0),
            int(node.get("recency_counter") or 0),
        )

    selected_nodes: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add_node(node: Optional[Dict[str, Any]]) -> None:
        if not node:
            return
        node_id = str(node.get("id", ""))
        if not node_id or node_id in selected_ids:
            return
        selected_ids.add(node_id)
        selected_nodes.append(node)

    def add_path_ancestors(path: Sequence[str]) -> None:
        if not path:
            return
        for depth in range(1, len(path) + 1):
            ancestor = node_lookup.get(_path_id(path[:depth]))
            if ancestor is not None:
                add_node(ancestor)

    add_node(root_node)

    ordered_evidence = sorted(evidence_nodes, key=recency_key, reverse=True)
    for evidence_node in ordered_evidence:
        path = evidence_node.get("memory_path") or []
        add_path_ancestors(path)
        add_node(evidence_node)
        if len(selected_ids) >= max_nodes:
            break

    if len(selected_ids) < max_nodes:
        remaining_branches = [node for node in branch_nodes if str(node.get("id")) not in selected_ids]
        for branch_node in sorted(remaining_branches, key=recency_key, reverse=True):
            add_node(branch_node)
            if len(selected_ids) >= max_nodes:
                break

    ordered_nodes = selected_nodes

    node_map: Dict[str, str] = {}
    mermaid_lines = ["graph TD"]

    def node_label(node: Dict[str, Any]) -> str:
        node_type = node.get("type", "node")
        if node_type == "memory_root":
            return "You\\n(personal memory root)"
        if node_type == "memory_evidence":
            label = node.get("label") or node.get("heading") or node.get("metadata", {}).get("page_title") or "evidence"
            classification = node.get("classification", "unknown")
            return f"Evidence\\n{_sanitize_mermaid_text(label)}\\n{classification}"
        if node_type.startswith("memory_"):
            path = node.get("memory_path") or []
            if len(path) >= 2:
                return _sanitize_mermaid_text(path[-1])
            return _sanitize_mermaid_text(node.get("label") or node_type)
        return _sanitize_mermaid_text(node_type)

    def node_style(node: Dict[str, Any]) -> str:
        node_type = node.get("type", "node")
        if node_type == "memory_root":
            return ":::session"
        if node_type == "memory_evidence":
            classification = node.get("classification")
            return ":::relatable" if classification == "relatable" else ":::neutral"
        if node_type.startswith("memory_"):
            return ":::neutral"
        return ":::neutral"

    for index, node in enumerate(ordered_nodes):
        original_id = str(node.get("id", f"node-{index}"))
        safe_id = f"n{index}"
        node_map[original_id] = safe_id
        label = node_label(node)
        style = node_style(node)
        mermaid_lines.append(f'{safe_id}["{label}"]{style}')

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
        mermaid_lines.append('empty["No personal memory yet"]:::neutral')

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
