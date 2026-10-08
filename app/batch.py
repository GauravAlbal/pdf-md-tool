"""Batch-convert every supported file in a folder → <folder>/<out_subdir>/."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from app.paths import normalize_user_path
from app.pipeline import Mode, parse_file

log = logging.getLogger("pdfmd.batch")

DOC_EXTS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".tif",
    ".tiff",
    ".docx",
    ".doc",
    ".docm",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".xlsm",
    ".odt",
    ".ods",
    ".odp",
    ".rtf",
    ".epub",
    ".csv",
}

DEFAULT_OUT = "markdown"


@dataclass
class FileOutcome:
    source: str
    output: str | None
    ok: bool
    mode_used: str | None = None
    pdf_type: str | None = None
    page_count: int = 0
    processing_ms: int = 0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class BatchReport:
    folder: str
    out_dir: str
    total: int
    ok: int
    failed: int
    skipped: int
    processing_ms: int
    files: list[FileOutcome] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def iter_docs(folder: Path, *, recursive: bool, out_subdir: str) -> Iterable[Path]:
    out_name = out_subdir.strip("/\\") or DEFAULT_OUT
    if recursive:
        paths = sorted(p for p in folder.rglob("*") if p.is_file())
    else:
        paths = sorted(p for p in folder.iterdir() if p.is_file())
    for p in paths:
        # never re-process our own output tree
        try:
            rel = p.relative_to(folder)
        except ValueError:
            continue
        if out_name in rel.parts:
            continue
        if p.suffix.lower() in DOC_EXTS:
            yield p


def output_path_for(src: Path, folder: Path, out_dir: Path, *, recursive: bool) -> Path:
    if recursive:
        rel = src.relative_to(folder)
        dest = out_dir / rel.with_suffix(".md")
    else:
        dest = out_dir / f"{src.stem}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def process_folder(
    folder: str | Path,
    *,
    out_subdir: str = DEFAULT_OUT,
    mode: Mode = "auto",
    recursive: bool = True,
    skip_existing: bool = True,
    model: str | None = None,
    server: str | None = None,
    api_key: str | None = None,
    ocr_engine: str | None = None,
    on_progress: Any = None,
) -> BatchReport:
    root = normalize_user_path(str(folder))
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    out_name = (out_subdir or DEFAULT_OUT).strip("/\\") or DEFAULT_OUT
    out_dir = root / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = list(iter_docs(root, recursive=recursive, out_subdir=out_name))
    t0 = time.perf_counter()
    outcomes: list[FileOutcome] = []
    ok_n = fail_n = skip_n = 0

    for i, src in enumerate(docs, 1):
        dest = output_path_for(src, root, out_dir, recursive=recursive)
        if on_progress:
            on_progress(
                {
                    "index": i,
                    "total": len(docs),
                    "source": str(src),
                    "output": str(dest),
                    "phase": "start",
                }
            )
        if skip_existing and dest.is_file() and dest.stat().st_size > 0:
            skip_n += 1
            outcomes.append(
                FileOutcome(
                    source=str(src),
                    output=str(dest),
                    ok=True,
                    mode_used="skipped",
                    error=None,
                )
            )
            if on_progress:
                on_progress({"index": i, "total": len(docs), "phase": "skipped", "source": str(src)})
            continue
        try:
            result = parse_file(
                src,
                mode=mode,
                model=model,
                server=server,
                api_key=api_key,
                ocr_engine=ocr_engine,  # type: ignore[arg-type]
            )
            dest.write_text(result.markdown or "", encoding="utf-8")
            # sidecar meta optional
            meta_path = dest.with_suffix(".json")
            meta_path.write_text(
                json.dumps(
                    {
                        "source": str(src),
                        "mode_used": result.mode_used,
                        "pdf_type": result.pdf_type,
                        "page_count": result.page_count,
                        "processing_ms": result.processing_ms,
                        "warnings": result.warnings,
                        "meta": result.meta,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            ok_n += 1
            outcomes.append(
                FileOutcome(
                    source=str(src),
                    output=str(dest),
                    ok=True,
                    mode_used=result.mode_used,
                    pdf_type=result.pdf_type,
                    page_count=result.page_count,
                    processing_ms=result.processing_ms,
                    warnings=list(result.warnings or []),
                )
            )
            if on_progress:
                on_progress(
                    {
                        "index": i,
                        "total": len(docs),
                        "phase": "done",
                        "source": str(src),
                        "output": str(dest),
                        "ok": True,
                    }
                )
        except Exception as e:
            log.exception("batch failed for %s", src)
            fail_n += 1
            outcomes.append(
                FileOutcome(
                    source=str(src),
                    output=None,
                    ok=False,
                    error=str(e),
                )
            )
            if on_progress:
                on_progress(
                    {
                        "index": i,
                        "total": len(docs),
                        "phase": "error",
                        "source": str(src),
                        "error": str(e),
                    }
                )

    report = BatchReport(
        folder=str(root),
        out_dir=str(out_dir),
        total=len(docs),
        ok=ok_n,
        failed=fail_n,
        skipped=skip_n,
        processing_ms=int((time.perf_counter() - t0) * 1000),
        files=outcomes,
    )
    (out_dir / "_batch_report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Batch PDF/docs → markdown subfolder")
    p.add_argument("folder", help="Folder of documents")
    p.add_argument(
        "-o",
        "--out-subdir",
        default=DEFAULT_OUT,
        help=f"Subfolder name inside the source folder (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "-m",
        "--mode",
        default="auto",
        choices=["auto", "native", "lighton", "olmocr", "ppocr", "anydoc"],
    )
    p.add_argument(
        "--ocr-engine",
        default=None,
        choices=["lighton", "olmocr"],
        help="VLM backend for auto mode (default: lighton)",
    )
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--server", default=None)
    p.add_argument("--api-key", default=None)
    args = p.parse_args(argv)

    def prog(ev: dict[str, Any]) -> None:
        if ev.get("phase") == "start":
            print(f"[{ev['index']}/{ev['total']}] {ev['source']}")
        elif ev.get("phase") == "done":
            print(f"  → {ev.get('output')}")
        elif ev.get("phase") == "skipped":
            print(f"  skip existing")
        elif ev.get("phase") == "error":
            print(f"  ERROR: {ev.get('error')}")

    report = process_folder(
        args.folder,
        out_subdir=args.out_subdir,
        mode=args.mode,  # type: ignore[arg-type]
        recursive=not args.no_recursive,
        skip_existing=not args.no_skip_existing,
        model=args.model,
        server=args.server,
        api_key=args.api_key,
        ocr_engine=args.ocr_engine,
        on_progress=prog,
    )
    print(
        f"\nDone: {report.ok} ok, {report.failed} failed, {report.skipped} skipped "
        f"→ {report.out_dir} ({report.processing_ms} ms)"
    )
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
