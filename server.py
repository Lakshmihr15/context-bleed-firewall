"""FastAPI server for the privacy graph middleware."""

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from middleware import capture_chrome_input, get_mermaid_graph, get_graph_snapshot, get_graph_summary, process_llm_call

app = FastAPI(title="Context Bleed Firewall Server", version="0.1.0")
BASE_DIR = Path(__file__).resolve().parent


@app.get("/", response_class=HTMLResponse)
async def index():
    template = (BASE_DIR / "index.html").read_text(encoding="utf-8")
    initial_summary = json.dumps(get_graph_summary())
    initial_snapshot = json.dumps(get_graph_snapshot())
    bootstrap = (
        "<script>"
        f"window.__INITIAL_GRAPH_SUMMARY__ = {initial_summary};"
        f"window.__INITIAL_GRAPH_SNAPSHOT__ = {initial_snapshot};"
        "</script>"
    )
    rendered = template.replace("</body>", f"{bootstrap}</body>")
    return HTMLResponse(rendered)


@app.get("/style.css")
async def style_css():
    return FileResponse(BASE_DIR / "style.css")


@app.get("/app.js")
async def app_js():
    return FileResponse(BASE_DIR / "app.js")


class ProcessRequest(BaseModel):
    user_input: str
    llm_response: str
    sensitive_tags: List[str] = Field(default_factory=lambda: ["#private"])
    similarity_threshold: float = 0.85


class ProcessResponse(BaseModel):
    leak_detected: bool
    leaked_chunks: List[Dict[str, Any]]


class ChromeInputRequest(BaseModel):
    tab_id: str
    element_id: str
    text: str
    page_title: Optional[str] = None
    page_url: Optional[str] = None
    heading: Optional[str] = None
    field_type: Optional[str] = None


class ChromeInputResponse(BaseModel):
    status: str
    input_id: str
    topic: str
    classification: str
    confidence: float
    token_estimate: int
    graph_node: str


class GraphSummaryResponse(BaseModel):
    totals: Dict[str, Any]
    total_inputs: int
    relatable_ratio: float
    token_reduction_pct: float
    raw_token_estimate: int
    compact_token_estimate: int
    topic_buckets: List[Dict[str, Any]]


class MermaidGraphResponse(BaseModel):
    mermaid: str
    node_count: int
    edge_count: int
    summary: Dict[str, Any]


@app.post("/process", response_model=ProcessResponse)
async def process(request: ProcessRequest):
    try:
        leak, details = process_llm_call(
            user_input=request.user_input,
            llm_response=request.llm_response,
            sensitive_tags=request.sensitive_tags,
            similarity_threshold=request.similarity_threshold,
        )
        return ProcessResponse(leak_detected=leak, leaked_chunks=details)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chrome_input", response_model=ChromeInputResponse)
async def chrome_input(request: ChromeInputRequest):
    try:
        result = capture_chrome_input(
            request.tab_id,
            request.element_id,
            request.text,
            {
                "page_title": request.page_title,
                "page_url": request.page_url,
                "heading": request.heading,
                "field_type": request.field_type,
            },
        )
        return ChromeInputResponse(status="captured", **result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/summary", response_model=GraphSummaryResponse)
async def graph_summary():
    try:
        return GraphSummaryResponse(**get_graph_summary())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/snapshot")
async def graph_snapshot():
    try:
        return get_graph_snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/mermaid", response_model=MermaidGraphResponse)
async def graph_mermaid():
    try:
        return MermaidGraphResponse(**get_mermaid_graph())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/view", response_class=HTMLResponse)
async def graph_view():
    try:
        graph = get_mermaid_graph()
        mermaid = graph["mermaid"].replace("`", "\\`")
        mermaid_pre = html.escape(graph["mermaid"])
        return HTMLResponse(
            f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Pacific Graph View</title>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: Inter, system-ui, sans-serif;
      background: #0f1115;
      color: #f8fafc;
      padding: 24px;
    }}
    .wrap {{ max-width: 1400px; margin: 0 auto; }}
    .card {{
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 20px;
      margin-top: 16px;
    }}
    pre {{ white-space: pre-wrap; word-break: break-word; color: #cbd5e1; overflow-x: auto; }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 12px; }}
    .pill {{ padding: 12px 14px; border-radius: 12px; background: rgba(15, 118, 110, 0.16); border: 1px solid rgba(20, 184, 166, 0.2); }}
    .graph {{ background: #fff; border-radius: 16px; overflow: auto; padding: 16px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Pacific Graph View</h1>
    <div class="meta">
      <div class="pill">Nodes: {graph['node_count']}</div>
      <div class="pill">Edges: {graph['edge_count']}</div>
      <div class="pill">Relatable ratio: {round(graph['summary'].get('relatable_ratio', 0) * 100)}%</div>
      <div class="pill">Token reduction: {graph['summary'].get('token_reduction_pct', 0)}%</div>
    </div>
    <div class="card graph">
      <div class="mermaid">
{mermaid}
      </div>
    </div>
    <div class="card">
      <h2>Mermaid Source</h2>
      <pre>{mermaid_pre}</pre>
    </div>
  </div>
  <script>
    mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
  </script>
</body>
</html>
"""
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
