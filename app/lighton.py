"""LightOnOCR-3 local OCR (default VLM backend).

Supports:
  - in-process transformers (default)
  - OpenAI-compatible vLLM HTTP (LIGHTONOCR_BASE_URL)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger("pdfmd.lighton")

DEFAULT_MODEL = os.environ.get("LIGHTONOCR_MODEL", "lightonai/LightOnOCR-3-0.8B")
DEFAULT_LONGEST = int(os.environ.get("LIGHTONOCR_LONGEST_EDGE", "2048"))
DEFAULT_DPI = int(os.environ.get("LIGHTONOCR_DPI", "400"))
DEFAULT_MODE: Literal["plain", "grounding"] = (
    "grounding" if os.environ.get("LIGHTONOCR_MODE", "plain").lower() == "grounding" else "plain"
)

# ![label](x1,y1,x2,y2) markers from grounding mode
_GROUND_RE = re.compile(
    r"!\[([a-z0-9_+]+)\]\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*",
    re.I,
)

_lock = threading.Lock()
_tf_model = None
_tf_processor = None
_tf_model_id: str | None = None


def strip_grounding_markers(text: str) -> str:
    """Remove LightOnOCR-3 box markers; keep block text (and HTML tables)."""
    return _GROUND_RE.sub("", text or "").strip()


def render_pdf_pages(
    path: Path,
    *,
    longest_edge: int = DEFAULT_LONGEST,
    dpi: int = DEFAULT_DPI,
    pages: list[int] | None = None,
) -> list[tuple[int, Any]]:
    """Return list of (1-indexed page, PIL.Image)."""
    import pypdfium2 as pdfium
    from PIL import Image

    doc = pdfium.PdfDocument(str(path))
    n = len(doc)
    want = set(pages) if pages else None
    scale = dpi / 72.0
    out: list[tuple[int, Any]] = []
    for i in range(n):
        pnum = i + 1
        if want is not None and pnum not in want:
            continue
        pil = doc[i].render(scale=scale).to_pil()
        if not isinstance(pil, Image.Image):
            pil = Image.fromarray(pil)
        if max(pil.size) > longest_edge:
            pil = pil.copy()
            pil.thumbnail((longest_edge, longest_edge))
        out.append((pnum, pil.convert("RGB")))
    return out


def render_image(path: Path, *, longest_edge: int = DEFAULT_LONGEST) -> Any:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    if max(im.size) > longest_edge:
        im = im.copy()
        im.thumbnail((longest_edge, longest_edge))
    return im


def _pil_to_data_url(pil: Any) -> str:
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _ensure_transformers(model_id: str) -> tuple[Any, Any]:
    global _tf_model, _tf_processor, _tf_model_id
    with _lock:
        if _tf_model is not None and _tf_model_id == model_id:
            return _tf_model, _tf_processor
        import torch

        log.info("Loading LightOnOCR model %s …", model_id)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 0.8B / 4B: Qwen3.5; 1B / v2: LightOnOcr classes
        is_qwen = any(x in model_id for x in ("0.8B", "4B", "Qwen3"))
        if is_qwen:
            try:
                from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
            except ImportError as e:
                raise RuntimeError(
                    f"Need transformers with Qwen3.5 for {model_id} "
                    f'(uv pip install "transformers>=5.5.4"): {e}'
                ) from e
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_id, dtype=dtype, device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(model_id)
        else:
            try:
                from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
            except ImportError as e:
                raise RuntimeError(
                    f"Need transformers LightOnOcr classes for {model_id} "
                    f"(transformers>=5): {e}"
                ) from e
            model = LightOnOcrForConditionalGeneration.from_pretrained(
                model_id, dtype=dtype
            ).to(device)
            processor = LightOnOcrProcessor.from_pretrained(model_id)

        model.eval()
        _tf_model, _tf_processor, _tf_model_id = model, processor, model_id
        dev = getattr(model, "device", None) or next(model.parameters()).device
        log.info("LightOnOCR ready on %s", dev)
        return model, processor


def ocr_pil_transformers(
    pil: Any,
    *,
    model_id: str = DEFAULT_MODEL,
    mode: Literal["plain", "grounding"] = "plain",
    max_new_tokens: int = 4096,
    temperature: float = 0.2,
) -> str:
    import torch

    model, processor = _ensure_transformers(model_id)
    content: list[dict[str, Any]] = [{"type": "image", "image": pil}]
    if mode == "grounding":
        content.append({"type": "text", "text": "grounding"})
    conversation = [{"role": "user", "content": content}]

    try:
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    except TypeError:
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    # Resolve device
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.bfloat16)

    moved = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            if getattr(v, "is_floating_point", lambda: False)():
                moved[k] = v.to(device=device, dtype=dtype)
            else:
                moved[k] = v.to(device)
        else:
            moved[k] = v
    inputs = moved

    with torch.inference_mode():
        # 1B card uses greedy; 0.8B recommends sample — try sample then greedy
        try:
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
            )
        except Exception:
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)

    prompt_len = inputs["input_ids"].shape[1]
    gen = out[0, prompt_len:]
    return processor.decode(gen, skip_special_tokens=True).strip()


def ocr_pil_http(
    pil: Any,
    *,
    base_url: str,
    model_id: str,
    mode: Literal["plain", "grounding"] = "plain",
    api_key: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    timeout_s: float = 600.0,
) -> str:
    import requests

    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        if url.endswith("/v1"):
            url = url + "/chat/completions"
        else:
            url = url + "/v1/chat/completions"

    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": _pil_to_data_url(pil)}}
    ]
    if mode == "grounding":
        content.append({"type": "text", "text": "grounding"})

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
    if r.status_code >= 400:
        raise RuntimeError(f"LightOnOCR HTTP {r.status_code}: {r.text[:800]}")
    data = r.json()
    return (data["choices"][0]["message"]["content"] or "").strip()


def ocr_pil(
    pil: Any,
    *,
    model_id: str | None = None,
    mode: Literal["plain", "grounding"] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    model_id = model_id or DEFAULT_MODEL
    mode = mode or DEFAULT_MODE
    base_url = base_url if base_url is not None else (os.environ.get("LIGHTONOCR_BASE_URL") or "")
    api_key = api_key if api_key is not None else os.environ.get("LIGHTONOCR_API_KEY")

    if base_url.strip():
        text = ocr_pil_http(
            pil, base_url=base_url.strip(), model_id=model_id, mode=mode, api_key=api_key
        )
    else:
        text = ocr_pil_transformers(pil, model_id=model_id, mode=mode)

    if mode == "grounding" and os.environ.get("LIGHTONOCR_KEEP_MARKERS", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        # Keep tables/text; drop box markers for cleaner markdown default
        text = strip_grounding_markers(text)
    return text


def run_lighton_pdf(
    path: Path,
    *,
    model_id: str | None = None,
    mode: Literal["plain", "grounding"] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    pages: list[int] | None = None,
) -> tuple[str, dict[str, Any]]:
    model_id = model_id or DEFAULT_MODEL
    mode = mode or DEFAULT_MODE
    page_imgs = render_pdf_pages(path, pages=pages)
    if not page_imgs:
        raise RuntimeError(f"No pages rendered from {path}")

    parts: dict[int, str] = {}
    for pnum, pil in page_imgs:
        log.info("LightOnOCR page %s/%s (%s)", pnum, page_imgs[-1][0], path.name)
        parts[pnum] = ocr_pil(
            pil, model_id=model_id, mode=mode, base_url=base_url, api_key=api_key
        )

    # Join with page markers for hybrid stitch
    max_p = max(parts)
    joined_parts = []
    for n in range(1, max_p + 1):
        body = (parts.get(n) or "").strip()
        joined_parts.append(f"<!-- Page {n} -->\n\n{body}" if body else f"<!-- Page {n} -->\n")
    md = "\n\n".join(joined_parts).strip() + "\n"
    meta = {
        "engine": "lightonocr",
        "model": model_id,
        "mode": mode,
        "pages": list(parts.keys()),
        "base_url": base_url or None,
        "backend": "http" if (base_url or os.environ.get("LIGHTONOCR_BASE_URL")) else "transformers",
    }
    return md, meta


def run_lighton_image(
    path: Path,
    *,
    model_id: str | None = None,
    mode: Literal["plain", "grounding"] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> tuple[str, dict[str, Any]]:
    pil = render_image(path)
    text = ocr_pil(pil, model_id=model_id, mode=mode, base_url=base_url, api_key=api_key)
    meta = {
        "engine": "lightonocr",
        "model": model_id or DEFAULT_MODEL,
        "mode": mode or DEFAULT_MODE,
        "backend": "http" if (base_url or os.environ.get("LIGHTONOCR_BASE_URL")) else "transformers",
    }
    return text + ("\n" if not text.endswith("\n") else ""), meta
