from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.batch import process_folder
from app.paths import describe_path_help, normalize_user_path
from app.pipeline import (
    OLMOCR_MODEL_DEFAULT,
    ROOT,
    UPLOADS,
    WORKSPACE,
    parse_file_async,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pdfmd")

UPLOADS.mkdir(parents=True, exist_ok=True)
WORKSPACE.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PDF → Markdown", version="1.1.0")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

MAX_BYTES = int(os.environ.get("PDFMD_MAX_BYTES", 80 * 1024 * 1024))
JOBS: dict[str, dict[str, Any]] = {}
_batch_lock = asyncio.Lock()


class BatchRequest(BaseModel):
    folder: str = Field(..., description="Absolute path to a local folder")
    out_subdir: str = "markdown"
    mode: str = "auto"
    recursive: bool = True
    skip_existing: bool = True
    model: str = ""
    server: str = ""
    api_key: str = ""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    poppler = (ROOT / "mamba" / "env" / "bin" / "pdftoppm").is_file()
    torch_ok = False
    cuda_ok = False
    torch_ver = None
    try:
        import torch

        torch_ok = True
        torch_ver = torch.__version__
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        pass
    return {
        "ok": True,
        "poppler": poppler,
        "torch": torch_ok,
        "torch_version": torch_ver,
        "cuda": cuda_ok,
        "olmocr_model": os.environ.get("OLMOCR_MODEL", OLMOCR_MODEL_DEFAULT),
        "olmocr_server": os.environ.get("OLMOCR_SERVER") or None,
        "gpu_hint": "RTX 3080 10GB is snug for olmOCR-2-7B-FP8; free VRAM first.",
    }


@app.post("/api/parse")
async def parse(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    model: str = Form(""),
    server: str = Form(""),
    api_key: str = Form(""),
) -> JSONResponse:
    if mode not in ("auto", "native", "olmocr", "ppocr", "anydoc"):
        raise HTTPException(400, f"invalid mode: {mode}")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"file exceeds {MAX_BYTES} bytes")

    job_id = uuid.uuid4().hex[:16]
    orig = file.filename or "document.pdf"
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(orig).name)[:120]
    dest = UPLOADS / f"{job_id}_{safe}"
    dest.write_bytes(raw)

    JOBS[job_id] = {"status": "running", "filename": orig}

    try:
        result = await parse_file_async(
            dest,
            mode=mode,  # type: ignore[arg-type]
            model=model.strip() or None,
            server=server.strip() or None,
            api_key=api_key.strip() or None,
        )
        result.job_id = job_id
        out_md = RESULTS / f"{job_id}.md"
        out_json = RESULTS / f"{job_id}.json"
        out_md.write_text(result.markdown, encoding="utf-8")
        payload = result.to_dict()
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        JOBS[job_id] = {"status": "done", "filename": orig, "result": payload}
        return JSONResponse(payload)
    except Exception as e:
        log.exception("parse failed")
        JOBS[job_id] = {"status": "error", "filename": orig, "error": str(e)}
        raise HTTPException(500, str(e)) from e


@app.post("/api/batch")
async def batch(req: BatchRequest) -> JSONResponse:
    """Process every supported file under a local folder → folder/<out_subdir>/."""
    if req.mode not in ("auto", "native", "olmocr", "ppocr", "anydoc"):
        raise HTTPException(400, f"invalid mode: {req.mode}")

    try:
        folder = normalize_user_path(req.folder)
    except Exception as e:
        raise HTTPException(400, f"bad path: {e}") from e

    if not folder.is_dir():
        raise HTTPException(400, describe_path_help(folder))

    job_id = uuid.uuid4().hex[:16]
    JOBS[job_id] = {
        "status": "running",
        "kind": "batch",
        "folder": str(folder),
        "progress": [],
    }

    def on_progress(ev: dict[str, Any]) -> None:
        job = JOBS.get(job_id)
        if not job:
            return
        prog = list(job.get("progress") or [])
        prog.append(ev)
        # keep last 200 events
        job["progress"] = prog[-200:]
        job["last"] = ev

    async with _batch_lock:
        try:
            report = await asyncio.to_thread(
                process_folder,
                folder,
                out_subdir=req.out_subdir or "markdown",
                mode=req.mode,  # type: ignore[arg-type]
                recursive=req.recursive,
                skip_existing=req.skip_existing,
                model=req.model.strip() or None,
                server=req.server.strip() or None,
                api_key=req.api_key.strip() or None,
                on_progress=on_progress,
            )
            payload = report.to_dict()
            payload["job_id"] = job_id
            JOBS[job_id] = {"status": "done", "kind": "batch", "result": payload}
            (RESULTS / f"{job_id}.batch.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            return JSONResponse(payload)
        except Exception as e:
            log.exception("batch failed")
            JOBS[job_id] = {"status": "error", "kind": "batch", "error": str(e)}
            raise HTTPException(500, str(e)) from e


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        jp = RESULTS / f"{job_id}.json"
        bp = RESULTS / f"{job_id}.batch.json"
        if jp.is_file():
            return {"status": "done", "result": json.loads(jp.read_text(encoding="utf-8"))}
        if bp.is_file():
            return {"status": "done", "kind": "batch", "result": json.loads(bp.read_text(encoding="utf-8"))}
        raise HTTPException(404, "job not found")
    return job


@app.get("/api/download/{job_id}.md")
async def download_md(job_id: str) -> FileResponse:
    path = RESULTS / f"{job_id}.md"
    if not path.is_file():
        raise HTTPException(404, "not found")
    name = JOBS.get(job_id, {}).get("filename", job_id)
    stem = Path(str(name)).stem or job_id
    return FileResponse(path, filename=f"{stem}.md", media_type="text/markdown")


@app.get("/api/raw/{job_id}.md", response_class=PlainTextResponse)
async def raw_md(job_id: str) -> PlainTextResponse:
    path = RESULTS / f"{job_id}.md"
    if not path.is_file():
        raise HTTPException(404, "not found")
    return PlainTextResponse(path.read_text(encoding="utf-8"))


def create_app() -> FastAPI:
    return app
