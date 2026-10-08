"""Hybrid PDF/document → Markdown pipeline.

Routing (Firecrawl pdf-inspector):
  - text_based pages → native markdown (fast, no GPU)
  - scanned / image / mixed OCR pages → olmOCR 2 (VLM) when enabled
  - optional pdf-inspector PP-OCR as lightweight local OCR
Non-PDF office docs → Firecrawl AnyDoc.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import pdf_inspector

log = logging.getLogger("pdfmd.pipeline")

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "uploads"
WORKSPACE = ROOT / "workspace"
POPPLER_BIN = ROOT / "mamba" / "env" / "bin"

Mode = Literal["auto", "native", "olmocr", "ppocr", "anydoc"]

OLMOCR_MODEL_DEFAULT = "allenai/olmOCR-2-7B-1025-FP8"


@dataclass
class ParseResult:
    job_id: str
    filename: str
    mode_used: str
    pdf_type: str | None
    confidence: float | None
    page_count: int
    pages_native: list[int] = field(default_factory=list)
    pages_ocr: list[int] = field(default_factory=list)
    markdown: str = ""
    processing_ms: int = 0
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ensure_dirs() -> None:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    WORKSPACE.mkdir(parents=True, exist_ok=True)


def _env_path() -> dict[str, str]:
    env = os.environ.copy()
    if POPPLER_BIN.is_dir():
        env["PATH"] = f"{POPPLER_BIN}:{env.get('PATH', '')}"
    return env


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w.\-]+", "_", stem).strip("._") or "doc"
    return stem[:80]


def _is_pdf(filename: str, data: bytes) -> bool:
    if filename.lower().endswith(".pdf"):
        return True
    return data[:5] == b"%PDF-"


def _page_markdown_map(path: Path) -> dict[int, tuple[str, bool]]:
    """1-indexed page → (markdown, needs_ocr)."""
    pages = pdf_inspector.extract_pages_markdown(str(path))
    out: dict[int, tuple[str, bool]] = {}
    for p in pages.pages:
        # PageMarkdown.page is 0-indexed
        out[int(p.page) + 1] = (p.markdown or "", bool(p.needs_ocr))
    return out


def _join_pages(page_md: dict[int, str], page_count: int) -> str:
    parts: list[str] = []
    for n in range(1, page_count + 1):
        body = (page_md.get(n) or "").strip()
        parts.append(f"<!-- Page {n} -->\n\n{body}" if body else f"<!-- Page {n} -->\n")
    return "\n\n".join(parts).strip() + "\n"


def native_parse(path: Path) -> ParseResult:
    t0 = time.perf_counter()
    result = pdf_inspector.process_pdf(str(path))
    page_map = _page_markdown_map(path)
    native_pages = [n for n, (_, need) in page_map.items() if not need]
    ocr_pages = [n for n, (_, need) in page_map.items() if need]
    md = result.markdown or _join_pages({n: m for n, (m, _) in page_map.items()}, result.page_count)
    warnings: list[str] = []
    if ocr_pages:
        warnings.append(
            f"{len(ocr_pages)} page(s) flagged for OCR by pdf-inspector "
            f"({ocr_pages[:20]}{'…' if len(ocr_pages) > 20 else ''}); "
            "native mode left them as-is / empty."
        )
    if result.has_encoding_issues:
        warnings.append("Font encoding issues detected — consider olmOCR mode.")
    return ParseResult(
        job_id="",
        filename=path.name,
        mode_used="native",
        pdf_type=result.pdf_type,
        confidence=float(result.confidence),
        page_count=int(result.page_count),
        pages_native=native_pages,
        pages_ocr=ocr_pages,
        markdown=md,
        processing_ms=int((time.perf_counter() - t0) * 1000),
        warnings=warnings,
        meta={
            "title": result.title,
            "author": result.author,
            "is_complex_layout": result.is_complex_layout,
            "pages_with_tables": list(result.pages_with_tables or []),
            "has_encoding_issues": result.has_encoding_issues,
            "inspector_ms": result.processing_time_ms,
        },
    )


def ppocr_parse(path: Path) -> ParseResult:
    t0 = time.perf_counter()
    detect = pdf_inspector.detect_pdf(str(path))
    try:
        ocr = pdf_inspector.process_pdf_with_ocr(str(path))
    except Exception as e:
        # Fall back to native if OCR runtime missing
        base = native_parse(path)
        base.mode_used = "ppocr→native"
        base.warnings.append(f"pdf-inspector PP-OCR failed ({e}); used native extraction.")
        return base

    pages_native = [
        p.page_number
        for p in ocr.pages
        if p.provenance and p.provenance.source == "native"
    ]
    pages_ocr = list(ocr.pages_routed_to_ocr or [])
    return ParseResult(
        job_id="",
        filename=path.name,
        mode_used="ppocr",
        pdf_type=detect.pdf_type,
        confidence=float(detect.confidence),
        page_count=int(ocr.page_count),
        pages_native=pages_native,
        pages_ocr=pages_ocr,
        markdown=ocr.markdown or "",
        processing_ms=int((time.perf_counter() - t0) * 1000),
        warnings=[],
        meta={
            "pages_recommending_hosted": list(ocr.pages_recommending_hosted or []),
            "ocr_time_ms": ocr.ocr_time_ms,
            "render_time_ms": ocr.render_time_ms,
        },
    )


def anydoc_parse(path: Path) -> ParseResult:
    import anydoc

    t0 = time.perf_counter()
    md = anydoc.to_markdown(str(path))
    if not isinstance(md, str):
        md = str(md)
    return ParseResult(
        job_id="",
        filename=path.name,
        mode_used="anydoc",
        pdf_type=None,
        confidence=None,
        page_count=0,
        markdown=md,
        processing_ms=int((time.perf_counter() - t0) * 1000),
        meta={"format": path.suffix.lower()},
    )


def _find_olmocr_markdown(work: Path, pdf_path: Path) -> str | None:
    md_dir = work / "markdown"
    if not md_dir.is_dir():
        # newer layouts
        candidates = list(work.rglob("*.md"))
    else:
        candidates = list(md_dir.rglob("*.md"))
    if not candidates:
        return None
    stem = pdf_path.stem.lower()
    for c in candidates:
        if c.stem.lower() == stem or stem in c.stem.lower():
            return c.read_text(encoding="utf-8", errors="replace")
    # single file
    if len(candidates) == 1:
        return candidates[0].read_text(encoding="utf-8", errors="replace")
    # concatenate sorted
    parts = []
    for c in sorted(candidates):
        parts.append(c.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(parts)


def _olmocr_cmd(
    work: Path,
    pdf_path: Path,
    *,
    model: str,
    server: str | None,
    api_key: str | None,
    pages: list[int] | None,
) -> list[str]:
    # Prefer module invocation from the project venv
    py = str(ROOT / ".venv" / "bin" / "python")
    if not Path(py).exists():
        py = shutil.which("python3") or "python3"
    cmd = [
        py,
        "-m",
        "olmocr.pipeline",
        str(work),
        "--markdown",
        "--pdfs",
        str(pdf_path),
        "--model",
        model,
    ]
    if server:
        cmd.extend(["--server", server])
        if api_key:
            cmd.extend(["--api_key", api_key])
        cmd.extend(["--workers", "1"])
    else:
        # 3080 10GB is tight; leave headroom
        cmd.extend(["--gpu-memory-utilization", os.environ.get("OLMOCR_GPU_MEM", "0.85")])
    return cmd


def run_olmocr(
    path: Path,
    *,
    model: str = OLMOCR_MODEL_DEFAULT,
    server: str | None = None,
    api_key: str | None = None,
    timeout_s: int = 3600,
) -> tuple[str, dict[str, Any]]:
    work = WORKSPACE / f"olmocr-{uuid.uuid4().hex[:12]}"
    work.mkdir(parents=True, exist_ok=True)
    env = _env_path()
    cmd = _olmocr_cmd(work, path, model=model, server=server, api_key=api_key, pages=None)
    log.info("Running olmOCR: %s", " ".join(cmd[:8]) + " …")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"olmOCR timed out after {timeout_s}s") from e
    except FileNotFoundError as e:
        raise RuntimeError(
            "olmOCR not runnable. Install with: pip install 'olmocr[gpu]' "
            "(local GPU) or pip install olmocr + set OLMOCR_SERVER for remote vLLM."
        ) from e

    meta = {
        "returncode": proc.returncode,
        "workspace": str(work),
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
        "model": model,
        "server": server,
    }
    if proc.returncode != 0:
        raise RuntimeError(
            "olmOCR failed (exit %s). stderr tail:\n%s"
            % (proc.returncode, (proc.stderr or proc.stdout or "")[-2000:])
        )
    md = _find_olmocr_markdown(work, path)
    if not md:
        raise RuntimeError(
            "olmOCR finished but no markdown found under %s. stderr:\n%s"
            % (work, (proc.stderr or "")[-1500:])
        )
    return md, meta


def hybrid_auto(
    path: Path,
    *,
    model: str,
    server: str | None,
    api_key: str | None,
    force_olmocr_on_mixed: bool = True,
) -> ParseResult:
    """pdf-inspector native for clean pages; olmOCR for OCR-flagged pages / full scan."""
    t0 = time.perf_counter()
    detect = pdf_inspector.detect_pdf(str(path))
    page_map = _page_markdown_map(path)
    page_count = int(detect.page_count)
    need_ocr = sorted(
        {n for n, (_, flag) in page_map.items() if flag}
        | set(detect.pages_needing_ocr or [])
    )
    # detect.pages_needing_ocr is 1-indexed per docs for detect_pdf
    # Also force full olmOCR for pure scanned/image docs
    fully_scan = detect.pdf_type in ("scanned", "image_based") or (
        detect.pdf_type == "mixed" and force_olmocr_on_mixed and len(need_ocr) >= max(1, page_count // 2)
    )

    warnings: list[str] = []
    pages_native = [n for n in range(1, page_count + 1) if n not in need_ocr]
    pages_ocr: list[int] = []
    meta: dict[str, Any] = {
        "pdf_type": detect.pdf_type,
        "confidence": detect.confidence,
    }

    if not need_ocr and detect.pdf_type == "text_based" and not detect.has_encoding_issues:
        md = pdf_inspector.process_pdf(str(path)).markdown or _join_pages(
            {n: m for n, (m, _) in page_map.items()}, page_count
        )
        mode = "auto:native"
    elif fully_scan or detect.has_encoding_issues:
        try:
            md, ometa = run_olmocr(path, model=model, server=server, api_key=api_key)
            pages_ocr = list(range(1, page_count + 1))
            pages_native = []
            mode = "auto:olmocr-full"
            meta["olmocr"] = {k: ometa[k] for k in ("model", "server", "workspace") if k in ometa}
        except Exception as e:
            warnings.append(f"olmOCR unavailable ({e}); falling back to native/pp-ocr path.")
            try:
                fb = ppocr_parse(path)
                md = fb.markdown
                pages_ocr = fb.pages_ocr
                pages_native = fb.pages_native
                mode = "auto:ppocr-fallback"
                meta["fallback"] = str(e)[:500]
            except Exception as e2:
                base = native_parse(path)
                md = base.markdown
                mode = "auto:native-fallback"
                warnings.append(f"PP-OCR also failed ({e2}).")
    else:
        # Selective: native pages kept; OCR only needed pages via temp PDF subset if possible,
        # else full olmOCR and stitch.
        native_md = {n: page_map[n][0] for n in pages_native if n in page_map}
        try:
            # For mixed docs, run olmOCR on whole PDF (pipeline is page-group based)
            # then prefer native text on clean pages for fidelity/speed provenance.
            ocr_md, ometa = run_olmocr(path, model=model, server=server, api_key=api_key)
            pages_ocr = need_ocr
            # If olmOCR returns one blob, use it for OCR pages only when we can split by markers
            split = _split_olmocr_pages(ocr_md, page_count)
            merged: dict[int, str] = {}
            for n in range(1, page_count + 1):
                if n in need_ocr and n in split and split[n].strip():
                    merged[n] = split[n]
                elif n in native_md and native_md[n].strip():
                    merged[n] = native_md[n]
                elif n in split:
                    merged[n] = split[n]
                else:
                    merged[n] = native_md.get(n, "")
            md = _join_pages(merged, page_count)
            mode = "auto:hybrid"
            meta["olmocr"] = {k: ometa[k] for k in ("model", "server", "workspace") if k in ometa}
        except Exception as e:
            warnings.append(f"olmOCR selective path failed ({e}); native only.")
            md = _join_pages(native_md, page_count)
            mode = "auto:native-partial"
            pages_ocr = need_ocr

    return ParseResult(
        job_id="",
        filename=path.name,
        mode_used=mode,
        pdf_type=detect.pdf_type,
        confidence=float(detect.confidence),
        page_count=page_count,
        pages_native=pages_native,
        pages_ocr=pages_ocr,
        markdown=md,
        processing_ms=int((time.perf_counter() - t0) * 1000),
        warnings=warnings,
        meta=meta,
    )


def _split_olmocr_pages(md: str, page_count: int) -> dict[int, str]:
    """Best-effort split on common page markers."""
    # olmOCR sometimes emits form-feed or "Page N" style breaks
    if "\f" in md:
        chunks = md.split("\f")
        return {i + 1: c.strip() for i, c in enumerate(chunks) if i < page_count}
    # <!-- Page N --> markers (our hybrid) or similar
    parts = re.split(r"(?m)^(?:<!--\s*Page\s+(\d+)\s*-->|#{1,3}\s*Page\s+(\d+)\s*)$", md)
    if len(parts) > 1:
        out: dict[int, str] = {}
        # re.split keeps groups: text, g1, g2, text, ...
        i = 0
        preamble = parts[0]
        i = 1
        current = None
        buf: list[str] = []
        if preamble.strip() and page_count == 1:
            return {1: md.strip()}
        while i < len(parts):
            g1 = parts[i] if i < len(parts) else None
            g2 = parts[i + 1] if i + 1 < len(parts) else None
            body = parts[i + 2] if i + 2 < len(parts) else ""
            num_s = g1 or g2
            if current is not None:
                out[current] = "\n".join(buf).strip()
            try:
                current = int(num_s) if num_s else None
            except ValueError:
                current = None
            buf = [body] if body is not None else []
            i += 3
        if current is not None:
            out[current] = "\n".join(buf).strip()
        if out:
            return out
    # no markers — single blob
    return {1: md.strip()} if page_count >= 1 else {}


def parse_file(
    path: Path,
    *,
    mode: Mode = "auto",
    model: str | None = None,
    server: str | None = None,
    api_key: str | None = None,
) -> ParseResult:
    _ensure_dirs()
    model = model or os.environ.get("OLMOCR_MODEL", OLMOCR_MODEL_DEFAULT)
    server = server if server is not None else os.environ.get("OLMOCR_SERVER") or None
    api_key = api_key if api_key is not None else os.environ.get("OLMOCR_API_KEY") or None

    suffix = path.suffix.lower()
    head = path.read_bytes()[:8]
    is_pdf = _is_pdf(path.name, head)

    if mode == "anydoc" or (
        not is_pdf
        and suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
    ):
        return anydoc_parse(path)

    if not is_pdf and suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
        # image → olmOCR
        t0 = time.perf_counter()
        md, meta = run_olmocr(path, model=model, server=server, api_key=api_key)
        return ParseResult(
            job_id="",
            filename=path.name,
            mode_used="olmocr-image",
            pdf_type="image",
            confidence=None,
            page_count=1,
            pages_ocr=[1],
            markdown=md,
            processing_ms=int((time.perf_counter() - t0) * 1000),
            meta=meta,
        )

    if mode == "native":
        return native_parse(path)
    if mode == "ppocr":
        return ppocr_parse(path)
    if mode == "olmocr":
        t0 = time.perf_counter()
        detect = pdf_inspector.detect_pdf(str(path))
        md, meta = run_olmocr(path, model=model, server=server, api_key=api_key)
        return ParseResult(
            job_id="",
            filename=path.name,
            mode_used="olmocr",
            pdf_type=detect.pdf_type,
            confidence=float(detect.confidence),
            page_count=int(detect.page_count),
            pages_ocr=list(range(1, int(detect.page_count) + 1)),
            markdown=md,
            processing_ms=int((time.perf_counter() - t0) * 1000),
            meta=meta,
        )
    # auto
    return hybrid_auto(path, model=model, server=server, api_key=api_key)


async def parse_file_async(path: Path, **kwargs: Any) -> ParseResult:
    return await asyncio.to_thread(parse_file, path, **kwargs)
