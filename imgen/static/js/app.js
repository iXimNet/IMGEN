(() => {
  const state = {
    lang: "zh",
    bootstrap: null,
    mode: "generate",
    modelKey: "qwen-image-2.1",
    hub: "huggingface",
    scale: "2k",
    aspect: "1:1",
    refs: [],
    jobId: null,
    currentImage: null,
    history: [],
    selected: null,
    wizardStep: 0,
    startedAt: 0,
    timer: null,
    presetCat: null,
  };

  const $ = (id) => document.getElementById(id);
  const on = (el, ev, fn) => el && el.addEventListener(ev, fn);

  function tr(key) {
    return t(state.lang, key);
  }

  function applyI18n() {
    document.documentElement.lang = state.lang === "zh" ? "zh-Hans" : "en";
    document.querySelectorAll("[data-i]").forEach((el) => {
      el.textContent = tr(el.getAttribute("data-i"));
    });
    document.querySelectorAll("[data-i-placeholder]").forEach((el) => {
      el.placeholder = tr(el.getAttribute("data-i-placeholder"));
    });
    $("prompt").placeholder = state.mode === "edit" ? tr("promptPhEdit") : tr("promptPhGen");
    $("promptHint").textContent = state.mode === "edit" ? tr("promptPhEdit") : tr("promptPhGen");
    $("langToggle").textContent = state.lang === "zh" ? "EN" : "中文";
    $("followRow").classList.toggle("hidden", state.mode !== "edit");
    $("refSection").classList.toggle("hidden", state.mode !== "edit");
    renderPresets();
    renderHistory();
    const closeLabel = tr("close");
    ["closeSettingsX", "closeHistoryMoreX"].forEach((id) => {
      if ($(id)) $(id).setAttribute("aria-label", closeLabel);
    });
  }

  function esc(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function toast(message, err) {
    const el = document.createElement("div");
    el.className = "toast" + (err ? " err" : "");
    el.textContent = message;
    $("toasts").appendChild(el);
    setTimeout(() => el.remove(), 4200);
  }

  async function api(path, options) {
    const res = await fetch(path, options);
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const data = await res.json();
        detail = data.detail || data.message || JSON.stringify(data);
      } catch (_) {
        detail = await res.text();
      }
      throw new Error(detail);
    }
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) return res.json();
    return res;
  }

  function connectWs() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      handleEvent(msg);
    };
    ws.onclose = () => setTimeout(connectWs, 1500);
  }

  function handleEvent(msg) {
    if (msg.type === "generate_progress") {
      const pct = msg.total ? Math.round((msg.step / msg.total) * 100) : 0;
      $("progress").classList.remove("hidden");
      $("progressBar").style.width = pct + "%";
      const elapsed = ((Date.now() - state.startedAt) / 1000).toFixed(1);
      $("progressText").textContent = `${tr("stepOf")} ${msg.step} / ${msg.total} · ${elapsed}s`;
    }
    if (msg.type === "download_progress") {
      const n = msg.n || 0;
      const total = msg.total;
      const mb = (n / (1024 * 1024)).toFixed(1);
      const body = $("wizardBody");
      const log = body.querySelector(".dl-log");
      if (log) {
        log.textContent = total
          ? `${msg.file || ""}  ${mb} / ${(total / (1024 * 1024)).toFixed(1)} MB`
          : `${msg.file || ""}  ${mb} MB`;
      }
    }
    if (msg.type === "download_complete" || msg.type === "download_ok") {
      toast(tr("downloaded"));
      refreshBootstrap().then(() => {
        renderWizard();
        const btn = $("settingsDownload");
        if (btn) {
          btn.disabled = false;
          btn.textContent = tr("download");
        }
      });
    }
    if (msg.type === "job_complete") {
      state.jobId = msg.id;
      showResult(msg.image + "?t=" + Date.now());
      setBusy(false);
      loadHistory();
    }
    if (msg.type === "error") {
      setBusy(false);
      toast(msg.message || tr("failed"), true);
      const btn = $("settingsDownload");
      if (btn) {
        btn.disabled = false;
        btn.textContent = tr("download");
      }
    }
    if (msg.type === "job_cancelled") {
      setBusy(false);
      toast(tr("cancelled"));
    }
  }

  function setBusy(busy) {
    $("run").disabled = busy;
    $("run").textContent = busy ? tr("running") : tr("run");
    $("cancel").classList.toggle("hidden", !busy);
    if (!busy) {
      clearInterval(state.timer);
      setTimeout(() => $("progress").classList.add("hidden"), 800);
    }
  }

  function showResult(url) {
    state.currentImage = url;
    $("empty").classList.add("hidden");
    $("result").classList.remove("hidden");
    $("result").src = url;
  }

  function sizesForScale() {
    return state.bootstrap.sizes.scales[state.scale].sizes;
  }

  function renderScale() {
    $("scaleSeg").innerHTML = "";
    Object.entries(state.bootstrap.sizes.scales).forEach(([key, spec]) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = state.lang === "zh" ? spec.label_zh : spec.label_en;
      btn.className = key === state.scale ? "active" : "";
      btn.onclick = () => {
        state.scale = key;
        $("outRes").value = spec.output_resolution;
        const [w, h] = spec.sizes[state.aspect] || spec.sizes["1:1"];
        $("width").value = w;
        $("height").value = h;
        renderScale();
        renderAspect();
      };
      $("scaleSeg").appendChild(btn);
    });
  }

  function renderAspect() {
    const sizes = sizesForScale();
    $("aspectSeg").innerHTML = "";
    Object.keys(sizes).forEach((ratio) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = ratio;
      btn.className = ratio === state.aspect ? "active" : "";
      btn.onclick = () => {
        state.aspect = ratio;
        const [w, h] = sizes[ratio];
        $("width").value = w;
        $("height").value = h;
        renderAspect();
      };
      $("aspectSeg").appendChild(btn);
    });
    const current = sizes[state.aspect] || sizes["1:1"];
    if (!$("width").value) {
      $("width").value = current[0];
      $("height").value = current[1];
    }
  }

  function currentPresets() {
    const pack = state.bootstrap.prompts[state.mode === "edit" ? "edit" : "generate"] || [];
    return pack;
  }

  function renderPresets() {
    const cats = currentPresets();
    if (!cats.length) return;
    if (!state.presetCat || !cats.find((c) => c.id === state.presetCat)) {
      state.presetCat = cats[0].id;
    }
    $("presetCats").innerHTML = "";
    cats.forEach((cat) => {
      const btn = document.createElement("button");
      btn.className = "pill" + (cat.id === state.presetCat ? " active" : "");
      btn.textContent = `${cat.icon || ""} ${state.lang === "zh" ? cat.zh : cat.en}`;
      btn.onclick = () => {
        state.presetCat = cat.id;
        renderPresets();
      };
      $("presetCats").appendChild(btn);
    });
    const cat = cats.find((c) => c.id === state.presetCat) || cats[0];
    $("presetList").innerHTML = "";
    (cat.prompts || []).forEach((p) => {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "preset-card";
      const title = state.lang === "zh" ? p.title_zh : p.title_en;
      card.innerHTML = `<b>${title}</b><span>${p.prompt.slice(0, 90)}${p.prompt.length > 90 ? "…" : ""}</span>`;
      card.onclick = () => {
        $("prompt").value = p.prompt;
      };
      $("presetList").appendChild(card);
    });
  }

  function lucky() {
    const cats = currentPresets();
    const all = cats.flatMap((c) => c.prompts || []);
    if (!all.length) return;
    const p = all[Math.floor(Math.random() * all.length)];
    $("prompt").value = p.prompt;
  }

  function renderRefs() {
    $("refCount").textContent = `${state.refs.length} / 10`;
    $("refGrid").innerHTML = "";
    state.refs.forEach((item, i) => {
      const div = document.createElement("div");
      div.className = "ref-item";
      div.innerHTML = `<img src="${item.url}" alt="ref ${i + 1}" /><button type="button">×</button><span class="ref-index">${i + 1}</span>`;
      div.querySelector("button").onclick = () => {
        URL.revokeObjectURL(item.url);
        state.refs.splice(i, 1);
        renderRefs();
      };
      $("refGrid").appendChild(div);
    });
  }

  function addFiles(fileList) {
    const files = Array.from(fileList || []);
    files.forEach((file) => {
      if (!file.type.startsWith("image/")) return;
      if (state.refs.length >= 10) return;
      state.refs.push({ file, url: URL.createObjectURL(file) });
    });
    renderRefs();
  }

  function chips() {
    const d = state.bootstrap.device;
    $("deviceChip").innerHTML = `<strong>${d.device_name}</strong>`;
    const model = (state.bootstrap.models || []).find((m) => m.key === state.modelKey);
    const loaded = state.bootstrap.engine && state.bootstrap.engine.loaded;
    $("modelChip").innerHTML = `<strong>${model ? model.label : state.modelKey}</strong> · ${state.hub}`;
    $("modelChip").classList.toggle("warn", !!(model && model.incomplete));
    $("modelChip").classList.toggle("ok", !!(loaded && loaded.model_key === state.modelKey));
    $("demoChip").classList.toggle("hidden", !state.bootstrap.demo);
  }

  function fmt(value) {
    if (value == null || value === "") return "—";
    if (typeof value === "boolean") return value ? tr("on") : tr("off");
    return String(value);
  }

  function historyBrief(item) {
    const mode = item.mode === "edit" ? tr("edit") : tr("generate");
    const size = item.width && item.height ? `${item.width}×${item.height}` : "";
    const seed = item.seed != null ? `${tr("seed")} ${item.seed}` : "";
    return [mode, size, seed].filter(Boolean).join(" · ");
  }

  function fmtTime(iso) {
    if (!iso) return "";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return iso;
    const locale = state.lang === "zh" ? "zh-CN" : "en-US";
    return date.toLocaleString(locale, {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  }

  function historyParamRows(item) {
    const p = item.params || {};
    const size = item.width && item.height ? `${item.width} × ${item.height}` : "";
    return [
      [tr("status"), item.status],
      [tr("createdAt"), fmtTime(item.created_at)],
      [tr("mode"), item.mode === "edit" ? tr("edit") : tr("generate")],
      [tr("duration"), item.duration_ms ? (item.duration_ms / 1000).toFixed(1) + " s" : ""],
      [tr("model"), item.model_key],
      [tr("hub"), item.hub],
      [`${tr("width")} × ${tr("height")}`, size],
      [tr("aspect"), p.aspect],
      [tr("scale"), p.scale],
      [tr("outputRes"), p.output_resolution],
      [tr("steps"), p.steps],
      [tr("cfg"), p.true_cfg_scale],
      [tr("seed"), item.seed],
      [tr("nImages"), p.num_images],
      [tr("kvCache"), p.use_kv_cache],
      [tr("transparent"), p.transparent],
      [tr("followRef"), p.follow_ref_aspect],
      [tr("vaeTiling"), p.vae_tiling],
    ];
  }

  function renderHistory() {
    const list = $("historyList");
    if (!list || !state.history) return;
    const q = ($("historySearch").value || "").toLowerCase();
    list.innerHTML = "";
    const items = state.history.filter((h) => !q || (h.prompt || "").toLowerCase().includes(q));
    if (!items.length) {
      list.innerHTML = `<p class="hint">${tr("emptyHistory")}</p>`;
      return;
    }
    items.forEach((item) => {
      const row = document.createElement("div");
      row.className = "h-item" + (state.selected === item.id ? " active" : "");
      const thumb = item.thumb_url || item.image_url;
      const title = (item.prompt || "").trim() || item.id;
      const card = document.createElement("button");
      card.type = "button";
      card.className = "h-card";
      card.innerHTML = `
        ${thumb ? `<img src="${esc(thumb)}" alt="" />` : `<div class="h-ph"></div>`}
        <div class="meta">
          <b title="${esc(title)}">${esc(title)}</b>
          <div class="brief">${esc(historyBrief(item))}</div>
        </div>`;
      card.onclick = () => selectHistory(item);
      const more = document.createElement("button");
      more.type = "button";
      more.className = "ghost h-more";
      more.textContent = tr("more");
      more.onclick = (event) => {
        event.stopPropagation();
        openHistoryMore(item);
      };
      row.appendChild(card);
      row.appendChild(more);
      list.appendChild(row);
    });
  }

  function selectHistory(item) {
    state.selected = item.id;
    if (item.image_url) showResult(item.image_url);
    renderHistory();
  }

  function openHistoryMore(item) {
    selectHistory(item);
    const thumb = item.thumb_url || item.image_url;
    const cells = historyParamRows(item)
      .map(
        ([k, v]) =>
          `<div class="kv-cell"><span class="k">${esc(k)}</span><span class="v">${esc(fmt(v))}</span></div>`
      )
      .join("");
    const negative = item.negative_prompt || (item.params || {}).negative_prompt || "";
    const negHtml = negative
      ? `<div class="kv-cell span-2"><span class="k">${esc(tr("negative"))}</span><span class="v">${esc(fmt(negative))}</span></div>`
      : "";
    $("historyMoreBody").innerHTML = `
      ${thumb ? `<img class="sheet-preview" src="${esc(item.image_url || thumb)}" alt="" />` : ""}
      <div class="kv-grid">${cells}${negHtml}</div>
      <div class="prompt-label">${esc(tr("prompt"))}</div>
      <div class="prompt-block">${esc(item.prompt || "")}</div>`;
    $("histReuse").onclick = () => {
      reuse(item);
      $("historyMore").close();
    };
    $("histDel").onclick = async () => {
      if (!confirm(tr("confirmDelete"))) return;
      await api("/api/jobs/" + item.id, { method: "DELETE" });
      $("historyMore").close();
      if (state.selected === item.id) state.selected = null;
      loadHistory();
    };
    $("historyMore").showModal();
  }

  function reuse(item) {
    const p = item.params || {};
    $("prompt").value = item.prompt || p.prompt || "";
    $("negative").value = item.negative_prompt || "";
    $("steps").value = p.steps || 40;
    $("cfg").value = p.true_cfg_scale || 1;
    $("seed").value = item.seed ?? -1;
    $("nImages").value = p.num_images || 1;
    if (p.scale) state.scale = p.scale;
    if (p.aspect) state.aspect = p.aspect;
    $("outRes").value = p.output_resolution || 2048;
    if (p.width) $("width").value = p.width;
    if (p.height) $("height").value = p.height;
    $("kv").classList.toggle("on", p.use_kv_cache !== false);
    $("rgba").classList.toggle("on", !!p.transparent);
    if (item.mode === "edit") {
      state.mode = "edit";
      $("modeEdit").classList.add("active");
      $("modeGen").classList.remove("active");
    }
    $("stepsVal").textContent = $("steps").value;
    $("cfgVal").textContent = Number($("cfg").value).toFixed(1);
    applyI18n();
    renderScale();
    renderAspect();
  }

  async function loadHistory() {
    const data = await api("/api/jobs?limit=80");
    state.history = data.items || [];
    renderHistory();
  }

  async function run() {
    const prompt = $("prompt").value.trim();
    if (!prompt) {
      $("prompt").focus();
      return;
    }
    if (state.mode === "edit" && !state.refs.length) {
      toast(tr("needRefs"), true);
      return;
    }
    const fd = new FormData();
    fd.set("mode", state.mode);
    fd.set("prompt", prompt);
    fd.set("negative_prompt", $("negative").value);
    fd.set("model_key", state.modelKey);
    fd.set("hub", state.hub);
    fd.set("scale", state.scale);
    fd.set("aspect", state.aspect);
    fd.set("width", $("width").value || "0");
    fd.set("height", $("height").value || "0");
    fd.set("output_resolution", $("outRes").value || "0");
    fd.set("steps", $("steps").value);
    fd.set("true_cfg_scale", $("cfg").value);
    fd.set("seed", $("seed").value);
    fd.set("use_kv_cache", $("kv").classList.contains("on") ? "true" : "false");
    fd.set("num_images", $("nImages").value);
    fd.set("transparent", $("rgba").classList.contains("on") ? "true" : "false");
    fd.set("follow_ref_aspect", $("follow").classList.contains("on") ? "true" : "false");
    fd.set("vae_tiling", $("vaeTiling").value);
    state.refs.forEach((r) => fd.append("files", r.file, r.file.name));
    setBusy(true);
    state.startedAt = Date.now();
    $("progress").classList.remove("hidden");
    $("progressBar").style.width = "2%";
    try {
      const job = await api("/api/jobs", { method: "POST", body: fd });
      state.jobId = job.id;
    } catch (err) {
      setBusy(false);
      toast(err.message, true);
    }
  }

  function renderWizard() {
    const step = state.wizardStep;
    const dots = [...$("wizardSteps").children];
    dots.forEach((d, i) => d.classList.toggle("on", i <= step));
    const body = $("wizardBody");
    const cfg = state.bootstrap.config;
    if (step === 0) {
      body.innerHTML = `
        <h1>${tr("setupTitle")}</h1>
        <p class="hint">${tr("setupWelcome")}</p>
        <div class="field" style="margin-top:16px">
          <label>${tr("language")}</label>
          <div class="seg">
            <button type="button" class="${state.lang === "zh" ? "active" : ""}" data-lang="zh">中文</button>
            <button type="button" class="${state.lang === "en" ? "active" : ""}" data-lang="en">English</button>
          </div>
        </div>`;
      body.querySelectorAll("[data-lang]").forEach((b) => {
        b.onclick = () => {
          state.lang = b.getAttribute("data-lang");
          applyI18n();
          renderWizard();
        };
      });
    }
    if (step === 1) {
      body.innerHTML = `<h1>${tr("hub")}</h1><div class="choice-grid"></div>`;
      const grid = body.querySelector(".choice-grid");
      Object.values(state.bootstrap.hubs).forEach((hub) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "choice" + (state.hub === hub.key ? " active" : "");
        btn.innerHTML = `<b>${hub.label}</b><p>${state.lang === "zh" ? hub.hint_zh : hub.hint_en}</p>`;
        btn.onclick = () => {
          state.hub = hub.key;
          renderWizard();
        };
        grid.appendChild(btn);
      });
    }
    if (step === 2) {
      body.innerHTML = `<h1>${tr("model")}</h1><div class="choice-grid"></div>`;
      const grid = body.querySelector(".choice-grid");
      (state.bootstrap.models || []).forEach((m) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "choice" + (state.modelKey === m.key ? " active" : "");
        const note = state.lang === "zh" ? m.notes_zh : m.notes_en;
        const status = m.downloaded ? tr("downloaded") : m.incomplete ? tr("incomplete") : m.repo;
        btn.innerHTML = `<b>${m.label} · ${m.precision}</b>
          <p>${tr("approx")} ${m.approx_gb} ${tr("gb")} · ${status}</p>
          <p>${note}</p>`;
        btn.onclick = () => {
          state.modelKey = m.key;
          renderWizard();
        };
        grid.appendChild(btn);
      });
    }
    if (step === 3) {
      const model = (state.bootstrap.models || []).find((m) => m.key === state.modelKey);
      body.innerHTML = `
        <h1>${tr("download")}</h1>
        <p class="hint">${model ? model.label : ""} · ${state.hub} · ${tr("approx")} ${model ? model.approx_gb : "?"} ${tr("gb")}</p>
        <div class="field"><label>${tr("hfToken")}</label><input id="wizHf" type="password" value="" /></div>
        <div class="field"><label>${tr("msToken")}</label><input id="wizMs" type="password" value="" /></div>
        <p class="dl-log hint">${model && model.downloaded ? tr("downloaded") : model && model.incomplete ? tr("incomplete") : ""}</p>
        <div class="seg" style="margin-top:12px">
          <button type="button" class="btn-gold" id="wizDl">${model && model.downloaded ? tr("skip") : model && model.incomplete ? tr("resumeDownload") : tr("download")}</button>
        </div>
        <p class="hint">${tr("licenseNote")}</p>`;
      $("wizDl").onclick = async () => {
        await api("/api/config", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            language: state.lang,
            hub: state.hub,
            model_key: state.modelKey,
            hf_token: $("wizHf").value,
            ms_token: $("wizMs").value,
          }),
        });
        if (model && model.downloaded) {
          finishWizard();
          return;
        }
        $("wizDl").disabled = true;
        $("wizDl").textContent = tr("downloading");
        await api("/api/models/download", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_key: state.modelKey, hub: state.hub }),
        });
      };
    }
    $("wizBack").style.visibility = step === 0 ? "hidden" : "visible";
    $("wizNext").textContent = step === 3 ? tr("done") : tr("next");
  }

  async function finishWizard() {
    await api("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        language: state.lang,
        hub: state.hub,
        model_key: state.modelKey,
        setup_completed: true,
      }),
    });
    $("wizard").close();
    chips();
  }

  async function refreshBootstrap() {
    state.bootstrap = await api("/api/bootstrap");
    const cfg = state.bootstrap.config;
    state.lang = cfg.language || state.lang;
    state.hub = cfg.hub || state.hub;
    state.modelKey = cfg.model_key || state.modelKey;
    chips();
    return state.bootstrap;
  }

  function bind() {
    on($("modeGen"), "click", () => {
      state.mode = "generate";
      $("modeGen").classList.add("active");
      $("modeEdit").classList.remove("active");
      applyI18n();
    });
    on($("modeEdit"), "click", () => {
      state.mode = "edit";
      $("modeEdit").classList.add("active");
      $("modeGen").classList.remove("active");
      applyI18n();
    });
    on($("langToggle"), "click", async () => {
      state.lang = state.lang === "zh" ? "en" : "zh";
      applyI18n();
      await api("/api/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ language: state.lang }),
      });
    });
    on($("lucky"), "click", lucky);
    on($("steps"), "input", () => ($("stepsVal").textContent = $("steps").value));
    on($("cfg"), "input", () => ($("cfgVal").textContent = Number($("cfg").value).toFixed(1)));
    ["kv", "rgba", "follow"].forEach((id) => {
      on($(id), "click", () => $(id).classList.toggle("on"));
    });
    on($("run"), "click", run);
    on($("cancel"), "click", async () => {
      if (state.jobId) await api("/api/jobs/" + state.jobId + "/cancel", { method: "POST" });
    });
    on($("downloadBtn"), "click", () => {
      if (!state.currentImage) return;
      const a = document.createElement("a");
      a.href = state.currentImage;
      a.download = "imgen.png";
      a.click();
    });
    on($("reuseBtn"), "click", () => {
      const item = state.history.find((h) => h.id === state.selected);
      if (item) reuse(item);
    });
    on($("sendEditBtn"), "click", async () => {
      if (!state.currentImage) return;
      const res = await fetch(state.currentImage);
      const blob = await res.blob();
      const file = new File([blob], "reference.png", { type: "image/png" });
      state.mode = "edit";
      $("modeEdit").classList.add("active");
      $("modeGen").classList.remove("active");
      addFiles([file]);
      applyI18n();
    });
    on($("dropzone"), "click", () => $("fileInput").click());
    on($("fileInput"), "change", (e) => addFiles(e.target.files));
    ["dragenter", "dragover"].forEach((ev) =>
      on($("dropzone"), ev, (e) => {
        e.preventDefault();
        $("dropzone").classList.add("drag");
      })
    );
    on($("dropzone"), "dragleave", () => $("dropzone").classList.remove("drag"));
    on($("dropzone"), "drop", (e) => {
      e.preventDefault();
      $("dropzone").classList.remove("drag");
      addFiles(e.dataTransfer.files);
    });
    on(document, "paste", (e) => {
      if (state.mode !== "edit") return;
      const items = [...(e.clipboardData?.items || [])];
      const files = items.filter((i) => i.type.startsWith("image/")).map((i) => i.getAsFile());
      addFiles(files);
    });
    on($("historySearch"), "input", renderHistory);
    on($("toggleHistory"), "click", () => $("historyPane").classList.toggle("open"));
    on($("toggleComposer"), "click", () => $("composer").classList.toggle("open"));
    on($("openSettings"), "click", () => {
      $("setHub").value = state.hub;
      $("setModel").value = state.modelKey;
      const model = (state.bootstrap.models || []).find((m) => m.key === state.modelKey);
      const status = $("settingsModelStatus");
      if (status) {
        if (model && model.incomplete) {
          status.textContent = tr("incomplete") + (model.missing_files || []).slice(0, 3).join(" · ");
        } else if (model && model.downloaded) {
          status.textContent = tr("downloaded");
        } else {
          status.textContent = "";
        }
      }
      $("settings").showModal();
    });
    on($("settingsDownload"), "click", async () => {
      const btn = $("settingsDownload");
      btn.disabled = true;
      btn.textContent = tr("downloading");
      try {
        await api("/api/models/download", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_key: $("setModel").value || state.modelKey,
            hub: $("setHub").value || state.hub,
          }),
        });
      } catch (err) {
        btn.disabled = false;
        btn.textContent = tr("download");
        toast(err.message, true);
      }
    });
    const closeSettings = () => $("settings").close();
    on($("closeSettings"), "click", closeSettings);
    on($("closeSettingsX"), "click", closeSettings);
    on($("closeHistoryMore"), "click", () => $("historyMore").close());
    on($("closeHistoryMoreX"), "click", () => $("historyMore").close());
    ["settings", "historyMore"].forEach((id) => {
      on($(id), "click", (event) => {
        if (event.target === $(id)) $(id).close();
      });
    });
    on($("saveSettings"), "click", async (e) => {
      e.preventDefault();
      state.hub = $("setHub").value;
      state.modelKey = $("setModel").value;
      await api("/api/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          hub: state.hub,
          model_key: state.modelKey,
          hf_token: $("setHf").value,
          ms_token: $("setMs").value,
        }),
      });
      $("settings").close();
      await refreshBootstrap();
    });
    on($("revealOut"), "click", async (e) => {
      e.preventDefault();
      await api("/api/reveal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
    });
    on($("wizBack"), "click", () => {
      state.wizardStep = Math.max(0, state.wizardStep - 1);
      renderWizard();
    });
    on($("wizNext"), "click", async () => {
      if (state.wizardStep < 3) {
        state.wizardStep += 1;
        renderWizard();
        return;
      }
      await finishWizard();
    });
    on(document, "keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        run();
      }
    });
  }

  async function init() {
    bind();
    await refreshBootstrap();
    applyI18n();
    renderScale();
    renderAspect();
    renderPresets();
    connectWs();
    await loadHistory();
    if (!state.bootstrap.config.setup_completed && !state.bootstrap.demo) {
      $("wizard").showModal();
      renderWizard();
    } else if (state.bootstrap.demo && !state.bootstrap.config.setup_completed) {
      // Still show a short welcome in demo so the flow is visible.
      $("wizard").showModal();
      renderWizard();
    }
  }

  init().catch((err) => toast(err.message, true));
})();
