# PDF → Markdown (local web UI)

Firecrawl **pdf-inspector** + **AnyDoc** + AllenAI **olmOCR 2**.

| Layer | Role |
|--------|------|
| **pdf-inspector** | Per-page classify + native text → Markdown |
| **olmOCR 2** | VLM OCR for scanned / image pages |
| **AnyDoc** | docx/pptx/xlsx/… → Markdown |

## Quick start

```bash
cd /home/dev/pdf-md-tool
./run.sh
```

Open **http://127.0.0.1:8787**

## Single file

Drop a PDF (or office doc) in the UI.

## Folder batch

**UI → Folder batch**: paste a local path (e.g. `/mnt/c/Users/hello/Documents/invoices`).

Output:

```text
<your-folder>/markdown/
  report.md
  report.json
  nested/scan.md
  _batch_report.json
```

CLI:

```bash
cd /home/dev/pdf-md-tool
source .venv/bin/activate
export PYTHONPATH=.
export PATH="$PWD/mamba/env/bin:$PATH"

python -m app.batch /path/to/folder
python -m app.batch /path/to/folder -o markdown -m native
python -m app.batch /path/to/folder --no-recursive --no-skip-existing
```

## Modes

- **Auto** — native when clean; olmOCR when needed  
- **Native** — pdf-inspector only  
- **Full olmOCR 2** — whole doc through VLM  
- **PP-OCR** — pdf-inspector PP-OCRv6  
- **AnyDoc** — office formats  

## GPU

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Torch **2.9+cu128** and **vllm** are in `.venv`. Free ~9 GB+ VRAM before local olmOCR, or set:

```bash
export OLMOCR_SERVER=http://127.0.0.1:8000/v1
export OLMOCR_MODEL=allenai/olmOCR-2-7B-1025-FP8
```

Poppler is under `mamba/env` (no root).
