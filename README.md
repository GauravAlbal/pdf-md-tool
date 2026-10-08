# PDF → Markdown (local web UI)

| Layer | Role |
|--------|------|
| **pdf-inspector** | Per-page classify + native text → Markdown |
| **LightOnOCR-3** (default) | VLM OCR for scans (`LightOnOCR-3-0.8B`) |
| **olmOCR 2** (optional) | Ai2 7B VLM path |
| **AnyDoc** | docx/pptx/xlsx/… → Markdown |

## Quick start

```bash
cd /home/dev/pdf-md-tool
./run.sh
```

Open **http://127.0.0.1:8787**

Needs: `transformers>=5.5.4`, CUDA torch, poppler under `mamba/env` (bundled).

## Modes

- **Auto** — native text via pdf-inspector; **LightOnOCR-3** on OCR pages
- **Native** — pdf-inspector only
- **Full LightOnOCR-3** — every page through LightOn
- **Full olmOCR 2** — Ai2 pipeline
- **PP-OCR** / **AnyDoc**

Advanced: switch Auto’s engine to olmOCR, set model, or point at a vLLM URL (`LIGHTONOCR_BASE_URL` / server field).

## Folder batch

UI **Folder batch** or:

```bash
python -m app.batch /path/to/folder
python -m app.batch 'C:\Users\hello\Downloads\docs' -m auto
python -m app.batch /path/to/folder --ocr-engine lighton
```

Writes `<folder>/markdown/…`.

## Env

```bash
# default engine is lighton
export PDFMD_OCR_ENGINE=lighton   # or olmocr
export LIGHTONOCR_MODEL=lightonai/LightOnOCR-3-0.8B
# export LIGHTONOCR_BASE_URL=http://127.0.0.1:8010/v1
# export LIGHTONOCR_MODE=plain    # or grounding
```

## GPU

LightOnOCR-3-0.8B is the default for **~10 GB** cards. First run downloads weights from Hugging Face.
