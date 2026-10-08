(() => {
  const $ = (id) => document.getElementById(id);
  const drop = $("dropzone");
  const fileInput = $("file");
  const fileLabel = $("file-label");
  const go = $("go");
  const status = $("status");
  const statusTitle = $("status-title");
  const statusSub = $("status-sub");
  const result = $("result");
  const batchResult = $("batch-result");
  const meta = $("meta");
  const preview = $("preview");
  const raw = $("raw");
  const warn = $("warn");
  const download = $("download");
  const copyBtn = $("copy");
  const healthEl = $("health");
  const folderInput = $("folder");

  let selected = null;
  let lastMarkdown = "";
  let seg = "file";

  function updateGo() {
    if (seg === "file") go.disabled = !selected;
    else go.disabled = !(folderInput.value || "").trim();
    go.textContent = seg === "folder" ? "Process folder" : "Convert to Markdown";
  }

  fetch("/api/health")
    .then((r) => r.json())
    .then((h) => {
      const bits = [];
      if (h.torch) bits.push(`torch ${h.torch_version || "ok"}`);
      else bits.push("torch missing");
      bits.push(h.cuda ? "cuda" : "no-cuda");
      if (h.poppler) bits.push("poppler");
      if (h.olmocr_server) bits.push("remote olmOCR");
      healthEl.textContent = bits.join(" · ");
    })
    .catch(() => {
      healthEl.textContent = "server unreachable";
    });

  document.querySelectorAll(".seg").forEach((btn) => {
    btn.addEventListener("click", () => {
      seg = btn.dataset.seg;
      document.querySelectorAll(".seg").forEach((b) => b.classList.toggle("active", b === btn));
      $("seg-file").hidden = seg !== "file";
      $("seg-folder").hidden = seg !== "folder";
      result.hidden = true;
      batchResult.hidden = true;
      updateGo();
    });
  });

  function setFile(f) {
    selected = f || null;
    fileLabel.textContent = selected ? selected.name : "Drop a PDF or document";
    updateGo();
  }

  drop.addEventListener("click", () => fileInput.click());
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") fileInput.click();
  });
  fileInput.addEventListener("change", () => setFile(fileInput.files?.[0]));
  folderInput.addEventListener("input", updateGo);

  ["dragenter", "dragover"].forEach((ev) => {
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.add("drag");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.remove("drag");
    });
  });
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (f) setFile(f);
  });

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const name = tab.dataset.tab;
      preview.hidden = name !== "preview";
      raw.hidden = name !== "raw";
    });
  });

  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function mdToHtml(md) {
    let s = escapeHtml(md);
    s = s.replace(/^```[\w]*\n([\s\S]*?)```/gm, (_, code) => `<pre><code>${code}</code></pre>`);
    s = s.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    s = s.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    s = s.replace(/^# (.+)$/gm, "<h1>$1</h1>");
    s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/\*(.+?)\*/g, "<em>$1</em>");
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/^\s*[-*] (.+)$/gm, "<li>$1</li>");
    s = s.replace(/(<li>.*<\/li>\n?)+/g, (m) => `<ul>${m}</ul>`);
    s = s.replace(/(?!<[hul]|<pre|<li)(.+)$/gm, (line) => {
      if (!line.trim()) return "";
      if (line.startsWith("<")) return line;
      return `<p>${line}</p>`;
    });
    return s;
  }

  function chip(text, cls) {
    const c = document.createElement("span");
    c.className = "chip" + (cls ? " " + cls : "");
    c.textContent = text;
    return c;
  }

  function sharedOpts() {
    return {
      mode: $("mode").value,
      model: $("model").value.trim(),
      server: $("server").value.trim(),
      api_key: $("api_key").value.trim(),
      ocr_engine: ($("ocr_engine") && $("ocr_engine").value) || "lighton",
    };
  }

  function showResult(data) {
    batchResult.hidden = true;
    result.hidden = false;
    lastMarkdown = data.markdown || "";
    raw.textContent = lastMarkdown;
    preview.innerHTML = mdToHtml(lastMarkdown);
    download.href = `/api/download/${data.job_id}.md`;
    download.hidden = false;
    copyBtn.hidden = false;
    download.setAttribute("download", `${(data.filename || "doc").replace(/\.[^.]+$/, "")}.md`);

    meta.innerHTML = "";
    meta.appendChild(chip(data.mode_used || "?", "ok"));
    if (data.pdf_type) meta.appendChild(chip(`type: ${data.pdf_type}`));
    if (data.page_count) meta.appendChild(chip(`${data.page_count} pages`));
    if (data.confidence != null) meta.appendChild(chip(`conf ${Number(data.confidence).toFixed(2)}`));
    meta.appendChild(chip(`${data.processing_ms} ms`));
    if (data.pages_ocr?.length) meta.appendChild(chip(`OCR pages: ${data.pages_ocr.length}`, "warn"));
    if (data.pages_native?.length) meta.appendChild(chip(`native: ${data.pages_native.length}`));

    if (data.warnings?.length) {
      warn.hidden = false;
      warn.textContent = data.warnings.join("\n");
    } else {
      warn.hidden = true;
      warn.textContent = "";
    }
  }

  function showBatch(data) {
    result.hidden = true;
    batchResult.hidden = false;
    download.hidden = true;
    copyBtn.hidden = true;
    const bm = $("batch-meta");
    bm.innerHTML = "";
    bm.appendChild(chip("batch", "ok"));
    bm.appendChild(chip(`${data.ok} ok`));
    if (data.failed) bm.appendChild(chip(`${data.failed} failed`, "warn"));
    if (data.skipped) bm.appendChild(chip(`${data.skipped} skipped`));
    bm.appendChild(chip(`${data.total} files`));
    bm.appendChild(chip(`${data.processing_ms} ms`));
    bm.appendChild(chip(data.out_dir || ""));

    const lines = [];
    lines.push(`folder: ${data.folder}`);
    lines.push(`out:    ${data.out_dir}`);
    lines.push("");
    for (const f of data.files || []) {
      if (f.ok) {
        const tag = f.mode_used === "skipped" ? "SKIP" : "OK";
        lines.push(`${tag}  ${f.source}`);
        if (f.output) lines.push(`   → ${f.output}`);
        if (f.warnings?.length) lines.push(`   ! ${f.warnings.join("; ")}`);
      } else {
        lines.push(`FAIL ${f.source}`);
        lines.push(`   × ${f.error || "unknown"}`);
      }
    }
    $("batch-log").textContent = lines.join("\n");
    const bw = $("batch-warn");
    if (data.failed) {
      bw.hidden = false;
      bw.textContent = `${data.failed} file(s) failed — see log. Report: ${data.out_dir}/_batch_report.json`;
    } else {
      bw.hidden = true;
    }
  }

  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(lastMarkdown);
      copyBtn.textContent = "Copied";
      setTimeout(() => (copyBtn.textContent = "Copy"), 1200);
    } catch {
      copyBtn.textContent = "Failed";
      setTimeout(() => (copyBtn.textContent = "Copy"), 1200);
    }
  });

  go.addEventListener("click", async () => {
    go.disabled = true;
    result.hidden = true;
    batchResult.hidden = true;
    status.hidden = false;

    try {
      if (seg === "folder") {
        statusTitle.textContent = "Batch converting…";
        statusSub.textContent = "Writing markdown into the folder’s subfolder";
        const opts = sharedOpts();
        const res = await fetch("/api/batch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            folder: folderInput.value.trim(),
            out_subdir: $("out_subdir").value.trim() || "markdown",
            recursive: $("recursive").checked,
            skip_existing: $("skip_existing").checked,
            ...opts,
          }),
        });
        const text = await res.text();
        let data;
        try {
          data = JSON.parse(text);
        } catch {
          throw new Error(text || res.statusText);
        }
        if (!res.ok) throw new Error(data.detail || data.message || text);
        status.hidden = true;
        showBatch(data);
      } else {
        if (!selected) return;
        const fd = new FormData();
        fd.append("file", selected);
        const opts = sharedOpts();
        fd.append("mode", opts.mode);
        fd.append("model", opts.model);
        fd.append("server", opts.server);
        fd.append("api_key", opts.api_key);
        fd.append("ocr_engine", opts.ocr_engine);
        const res = await fetch("/api/parse", { method: "POST", body: fd });
        const text = await res.text();
        let data;
        try {
          data = JSON.parse(text);
        } catch {
          throw new Error(text || res.statusText);
        }
        if (!res.ok) throw new Error(data.detail || data.message || text);
        status.hidden = true;
        showResult(data);
      }
    } catch (err) {
      status.hidden = true;
      if (seg === "folder") {
        batchResult.hidden = false;
        $("batch-meta").innerHTML = "";
        $("batch-meta").appendChild(chip("error", "warn"));
        $("batch-log").textContent = "";
        $("batch-warn").hidden = false;
        $("batch-warn").textContent = String(err.message || err);
      } else {
        result.hidden = false;
        meta.innerHTML = "";
        meta.appendChild(chip("error", "warn"));
        preview.innerHTML = "";
        raw.textContent = "";
        warn.hidden = false;
        warn.textContent = String(err.message || err);
      }
    } finally {
      updateGo();
    }
  });

  updateGo();
})();
