/* IMGEN — local studio for Qwen-Image-2.1.
   Layers: state → renderers → API. Renderers never fetch; fetchers never render. */

(() => {
  "use strict";

  /* ======================================================================
     State
     ====================================================================== */
  const S = {
    lang: "zh",
    bootstrap: null,
    mode: "generate",
    /* One parameter set per mode. Switching modes swaps the whole panel, so
       tuning a generation never disturbs an edit in progress. */
    params: { generate: null, edit: null },
    modelKey: "qwen-image-2.1",
    hub: "huggingface",
    scale: "1k",
    aspect: "1:1",
    width: 1024,
    height: 1024,
    steps: 40,
    cfg: 1,
    out: 1024,
    refs: [],
    jobId: null,
    startedAt: 0,
    /* The stage the engine reports it is in. The socket announces it and the
       poll repeats it, so a reload can re-attach to a run it never saw start. */
    phaseName: null,
    phaseSlow: false,
    phaseAt: 0,
    phaseTimer: 0,
    /* Liveness: when the last job event arrived, and whether the server is
       still answering. A bar that never moves is not progress. */
    lastEventAt: 0,
    watchdog: 0,
    offline: false,
    pollTimer: 0,
    currentImage: null,
    /* Every frame the last run produced, and which one the canvas is showing.
       A run with num_images > 1 makes several pictures in one go; the stage
       strip switches between them and the toolbar follows the selection. */
    outputFrames: [],
    outputIndex: 0,
    hasResult: false,
    busy: false,
    history: [],
    historyQuery: "",
    historyTotal: 0,
    historyLoading: false,
    presetCat: null,
    detailId: null,
    /* Which picture the open detail overlay is showing. The footer's actions
       follow it, so a download saves what is on screen. */
    detailCurrent: { url: null, index: 0 },
    /* Set while the viewer shows a reference instead of a frame. */
    detailRefIndex: null,
    /* Whether the detail viewer has taken the panel's column as well. A way of
       looking at one record, not a mode: closing puts it back. */
    detailWide: false,
    download: { key: null, pct: 0, bytes: 0, total: null },
    /* Freeing VRAM: the confirmation being asked (`ask`), the wait for the
       running stage to return (`stopping`), which host shows it, and the stage
       the wait is on. Kept apart from `busy` because a release also applies
       when nothing is running. */
    releasing: false,
    releaseStage: null,
    releaseHost: null,
    releasePhase: null,
    /* The run has gone quiet past the stall threshold. One of the two states
       that put the release action in the progress block: it is the thing you
       reach for once you have stopped waiting, not part of every run. */
    stalled: false,
  };

  const $ = (id) => document.getElementById(id);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const tr = (key) => t(S.lang, key);
  const tfx = (key, vars) => tf(S.lang, key, vars);

  const narrow = window.matchMedia("(max-width: 900px)");
  const compact = window.matchMedia("(max-width: 1180px)");
  /* Debounce handle for the history search box. */
  let searchTimer = null;
  const isNarrow = () => narrow.matches;

  const cfg = () => (S.bootstrap && S.bootstrap.config) || {};
  const defaults = () => (S.bootstrap && S.bootstrap.defaults) || {};
  const modelList = () => (S.bootstrap && S.bootstrap.models) || [];
  const modelByKey = (key) => modelList().find((m) => m.key === key) || null;
  const modelLabel = (key) => (modelByKey(key) || {}).label || key;
  const hubLabel = (key) => (key === "modelscope" ? tr("hubMs") : tr("hubHf"));
  const maxRefs = () => defaults().max_reference_images || 10;
  const defaultSteps = () => defaults().steps || 40;
  const maxImages = () => defaults().max_images || 4;
  /* Tiling is opt-in now. "auto" is a legacy value meaning "resolution decided"
     — it resolves to off, so the select must show it that way. */
  const vaeTilingValue = (raw) => {
    const v = String(raw == null ? "" : raw).trim().toLowerCase();
    return ["1", "true", "yes", "on"].includes(v) ? "true" : "false";
  };
  const scaleSpec = (key) => ((S.bootstrap && S.bootstrap.sizes && S.bootstrap.sizes.scales) || {})[key || S.scale] || null;
  const sizesForScale = () => (scaleSpec() && scaleSpec().sizes) || {};
  const aspectList = () => Object.keys(sizesForScale());

  function esc(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
  function icon(id, size) {
    const s = size || 14;
    return `<svg class="ico" width="${s}" height="${s}"><use href="#${id}"/></svg>`;
  }
  const floor32 = (value) => {
    const n = Number(value);
    if (!Number.isFinite(n) || n <= 0) return 0;
    return Math.max(32, Math.floor(n / 32) * 32);
  };
  /* Byte counts for download readouts, e.g. 1.4 GB / 512 MB. */
  function fmtBytes(bytes) {
    const n = Number(bytes) || 0;
    if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GB`;
    if (n >= 1024 ** 2) return `${Math.round(n / 1024 ** 2)} MB`;
    if (n >= 1024) return `${Math.round(n / 1024)} KB`;
    return `${n} B`;
  }
  function areaSize(side, ratio) {
    const w = Math.sqrt(side * side * ratio);
    const h = w / ratio;
    const snap = (n) => Math.max(32, Math.round(n / 32) * 32);
    return [snap(w), snap(h)];
  }
  /* The proportion actually on screen, for a width/height pair that is not one
     of the presets. Reducing by the gcd gives the honest label (`3:2`, not
     `1:1`); when the pair shares no useful divisor, a decimal says it better
     than a ratio nobody can read. */
  function ratioLabel(w, h) {
    if (!w || !h) return null;
    const gcd = (a, b) => (b ? gcd(b, a % b) : a);
    const g = gcd(w, h);
    const rw = w / g;
    const rh = h / g;
    if (g > 1 && rw <= 40 && rh <= 40) return `${rw}:${rh}`;
    return `${(w / h).toFixed(2)}:1`;
  }
  const followsRef = () => S.mode === "edit" && $("follow").getAttribute("aria-checked") === "true";
  /* The shortcut hint names the key the user actually has. The handler accepts
     either modifier, but only one of the two glyphs is on their keyboard. */
  const isMac = () => /Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent);
  const runKey = () => (isMac() ? "⌘↵" : "Ctrl↵");

  /* ======================================================================
     API
     ====================================================================== */
  async function api(path, options) {
    let res;
    try {
      res = await fetch(path, options);
    } catch (_) {
      const err = new Error(tr("errOffline"));
      err.offline = true;
      throw err;
    }
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const data = await res.json();
        detail = data.detail || data.message || JSON.stringify(data);
      } catch (_) {
        detail = await res.text().catch(() => detail);
      }
      const err = new Error(detail || `HTTP ${res.status}`);
      err.status = res.status;
      throw err;
    }
    const type = res.headers.get("content-type") || "";
    return type.includes("application/json") ? res.json() : res;
  }

  const putConfig = (patch) =>
    api("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });

  /* ======================================================================
     Toasts — a human line first, backend detail underneath
     ====================================================================== */
  function toast(kind, title, detail) {
    const box = $("toasts");
    const el = document.createElement("div");
    el.className = `toast ${kind === "err" ? "err" : "ok"}`;
    el.innerHTML =
      icon(kind === "err" ? "i-alert" : "i-check", 16).replace('class="ico"', 'class="ico ti"') +
      `<div><div class="tt">${esc(title)}</div>${detail ? `<div class="td">${esc(detail)}</div>` : ""}</div>` +
      `<button type="button" class="tx" aria-label="${esc(tr("close"))}">${icon("i-x", 14)}</button>`;
    el.querySelector(".tx").onclick = () => el.remove();
    box.appendChild(el);
    while (box.children.length > 3) box.removeChild(box.firstChild);
    if (kind !== "err") setTimeout(() => el.remove(), 3200);
  }

  function reportError(err) {
    const offline = err && err.offline;
    toast("err", offline ? tr("errOffline") : tr("errFailed"), offline ? tr("errOfflineDetail") : (err && err.message) || "");
  }

  /* ======================================================================
     Zoom — one implementation, instantiated twice
     ====================================================================== */
  function makeZoom(spec) {
    const { viewport, wrap, img, view, pct, fitBtn, oneBtn } = spec;
    let nat = [0, 0];
    let scale = 1;
    let mode = "fit";
    let drag = null;

    const pad = (prop) => parseInt(getComputedStyle(viewport)[prop], 10) || 0;
    function fitScale() {
      if (!nat[0] || !nat[1]) return 1;
      const w = Math.max(40, viewport.clientWidth - pad("paddingLeft") * 2);
      const h = Math.max(40, viewport.clientHeight - pad("paddingTop") * 2);
      return Math.min(w / nat[0], h / nat[1]);
    }
    const maxScale = () => Math.max(2, fitScale() * 8);
    const pannable = () => viewport.scrollWidth > viewport.clientWidth + 1 || viewport.scrollHeight > viewport.clientHeight + 1;
    const syncView = () => view.classList.toggle("pannable", pannable());

    function setSize(next) {
      // Nothing loaded: keep the cleared state instead of sizing a phantom image.
      // The ResizeObserver fires after clear() and would otherwise restore a size.
      if (!nat[0] || !nat[1]) return;
      scale = next;
      wrap.style.width = `${Math.max(1, Math.round(nat[0] * next))}px`;
      wrap.style.height = `${Math.max(1, Math.round(nat[1] * next))}px`;
      const fit = fitScale();
      if (Math.abs(next - 1) < 0.005) pct.textContent = "100%";
      else if (Math.abs(next - fit) < 0.005) pct.textContent = `${tr("fit")} · ${Math.round(next * 100)}%`;
      else pct.textContent = `${Math.round(next * 100)}%`;
      fitBtn.classList.toggle("on", Math.abs(next - fit) < 0.005);
      oneBtn.classList.toggle("on", Math.abs(next - 1) < 0.005);
      syncView();
    }
    const fit = () => { mode = "fit"; setSize(fitScale()); };
    const one = () => { mode = "one"; setSize(1); };
    function step(dir) {
      const f = fitScale();
      const next = clamp(scale * (dir > 0 ? 1.25 : 0.8), f, maxScale());
      mode = Math.abs(next - 1) < 0.005 ? "one" : Math.abs(next - f) < 0.005 ? "fit" : "free";
      setSize(next);
    }
    function zoomAt(next, cx, cy) {
      const f = fitScale();
      const target = clamp(next, f, maxScale());
      if (Math.abs(target - scale) < 0.0005) return;
      const rect = viewport.getBoundingClientRect();
      const px = cx - rect.left + viewport.scrollLeft;
      const py = cy - rect.top + viewport.scrollTop;
      const k = target / scale;
      mode = Math.abs(target - 1) < 0.005 ? "one" : Math.abs(target - f) < 0.005 ? "fit" : "free";
      setSize(target);
      viewport.scrollLeft = px * k - (cx - rect.left);
      viewport.scrollTop = py * k - (cy - rect.top);
      syncView();
    }
    function load(w, h, url) {
      nat = [w, h];
      img.src = url;
      mode = "fit";
      // A cached picture can finish decoding before the listener is reached,
      // in which case no `load` event arrives for this src.
      if (img.complete && img.naturalWidth) adoptNatural();
      requestAnimationFrame(fit);
      setTimeout(fit, 70);
    }
    /* Trust the decoded bitmap over the caller's numbers. A reference image is
       stored at its own size, not the output's, so sizing the box from the
       record's width/height stretched it to the wrong aspect ratio. */
    function adoptNatural() {
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      if (!w || !h) return;
      if (w === nat[0] && h === nat[1]) return;
      nat = [w, h];
      if (mode === "fit") setSize(fitScale());
    }
    img.addEventListener("load", adoptNatural);
    /* Drop the picture entirely. Without this a failed record would keep
       showing whichever image was loaded last, because `load` bails on a
       missing url and nothing else clears the stage. */
    function clear() {
      nat = [0, 0];
      scale = 1;
      mode = "fit";
      img.removeAttribute("src");
      wrap.style.width = "0px";
      wrap.style.height = "0px";
      pct.textContent = "—";
      fitBtn.classList.remove("on");
      oneBtn.classList.remove("on");
      // `fit`/`one` bail out on an empty natural size now, so the readout stays
      // cleared even if a resize observer fires afterwards.
      syncView();
    }
    function setNatural(w, h) { nat = [w, h]; }

    viewport.addEventListener("wheel", (event) => {
      if (!(event.ctrlKey || event.metaKey)) return;
      event.preventDefault();
      zoomAt(scale * (event.deltaY < 0 ? 1.12 : 0.89), event.clientX, event.clientY);
    }, { passive: false });
    viewport.addEventListener("dblclick", (event) => {
      event.preventDefault();
      if (Math.abs(scale - 1) < 0.005) fit(); else one();
    });
    viewport.addEventListener("mousedown", (event) => {
      if (event.button !== 0 || !pannable()) return;
      drag = { x: event.clientX, y: event.clientY, sl: viewport.scrollLeft, st: viewport.scrollTop };
      view.classList.add("panning");
      event.preventDefault();
    });
    window.addEventListener("mousemove", (event) => {
      if (!drag) return;
      viewport.scrollLeft = drag.sl - (event.clientX - drag.x);
      viewport.scrollTop = drag.st - (event.clientY - drag.y);
      syncView();
    });
    window.addEventListener("mouseup", () => {
      if (!drag) return;
      drag = null;
      view.classList.remove("panning");
    });
    viewport.addEventListener("scroll", syncView);
    if (window.ResizeObserver) {
      new ResizeObserver(() => { if (mode === "fit") setSize(fitScale()); else syncView(); }).observe(viewport);
    }
    return {
      fit, one, step, load, clear, setNatural, syncView,
      resize() { if (mode === "fit") setSize(fitScale()); },
    };
  }

  const canvasZoom = makeZoom({
    viewport: $("viewport"), wrap: $("zoomwrap"), img: $("result"), view: $("view"),
    pct: $("zPct"), fitBtn: $("zFit"), oneBtn: $("zoombar").querySelector('[data-z="one"]'),
  });
  const detailZoom = makeZoom({
    viewport: $("dvp"), wrap: $("dwrap"), img: $("dimg"), view: $("dview"),
    pct: $("dzPct"), fitBtn: $("dzFit"), oneBtn: $("detail").querySelector('[data-dz="one"]'),
  });

  /* ======================================================================
     i18n
     ====================================================================== */
  /* Re-render the open detail overlay in the current language. It needs the
     record, so `applyI18n` cannot call the renderers directly. The panel's own
     title is a `data-i` node and is handled by `applyI18n` itself. */
  function refreshOpenDetail() {
    if ($("detail").classList.contains("hidden") || !S.detailId) return;
    const item = S.history.find((row) => row.id === S.detailId);
    if (!item) return;
    renderDetailMeta(item);
    renderDetailRefs(item);
    renderDetailFooter(item);
    // The blocks are rebuilt, so their fold state has to be re-established.
    wirePromptFolds();
  }

  /* Translate a freshly built fragment. Markup that is thrown away and rebuilt
     can be interpolated at paint time, but markup that must survive a repaint —
     the release confirmation owns its host until it is answered — has to carry
     `data-i` and be painted here instead. */
  function paintI18n(root) {
    $$("[data-i]", root).forEach((el) => { el.textContent = tr(el.getAttribute("data-i")); });
  }

  function applyI18n() {
    document.documentElement.lang = S.lang === "zh" ? "zh-Hans" : "en";
    $$("[data-i]").forEach((el) => { el.textContent = tr(el.getAttribute("data-i")); });
    $$("[data-i-placeholder]").forEach((el) => { el.placeholder = tr(el.getAttribute("data-i-placeholder")); });
    // Some nodes carry their text in an attribute rather than as content — the
    // brand wordmark is an image, so its accessible name is the `alt`.
    $$("[data-i-alt]").forEach((el) => { el.alt = tr(el.getAttribute("data-i-alt")); });
    $("negative").placeholder = defaults().negative_placeholder || tr("negativePh");
    $("historySearch").placeholder = tr("searchHistory");
    $("presetSearch").placeholder = tr("presetsSearch");
    $("langToggle").title = tr("switchLang");
    $("btnHistory").title = tr("toggleHistory");
    $("btnHistory").setAttribute("aria-label", tr("toggleHistory"));
    $("openSettings").title = tr("settingsTitle");
    $("openSettings").setAttribute("aria-label", tr("settingsTitle"));
    $("langToggle").setAttribute("aria-label", tr("switchLang"));
    $("githubLink").title = tr("githubRepo");
    $("githubLink").setAttribute("aria-label", tr("githubRepo"));
    $("statusBtn").setAttribute("aria-label", tr("envTitle"));
    ["dClose", "settingsClose"].forEach((id) => $(id).setAttribute("aria-label", tr("close")));
    $("dPrev").title = tr("prevItem");
    $("dPrev").setAttribute("aria-label", tr("prevItem"));
    $("dNext").title = tr("nextItem");
    $("dNext").setAttribute("aria-label", tr("nextItem"));
    $("catPrev").setAttribute("aria-label", tr("catPrev"));
    $("catNext").setAttribute("aria-label", tr("catNext"));
    $$('[data-z="out"],[data-dz="out"]').forEach((b) => b.setAttribute("aria-label", tr("zoomOut")));
    $$('[data-z="in"],[data-dz="in"]').forEach((b) => b.setAttribute("aria-label", tr("zoomIn")));
    $$('[data-z="one"],[data-dz="one"]').forEach((b) => b.setAttribute("aria-label", tr("actualSize")));
    // The fill toggle's label names the next click rather than the thing, so it
    // is state-dependent and has to be repainted like any other readout.
    renderDetailExpand();
    renderMode();
    // Anything that reads `S.lang` at render time must be re-run here, or it
    // keeps the previous language until the next unrelated repaint.
    renderScale();
    renderRatios();
    renderValues();
    renderFollowState();
    renderRefs();
    renderRun();
    // The count stepper's spoken labels are translated, so they follow too.
    renderCount();
    renderSeed();
    // The stage readout is language-sensitive too; a switch mid-render must not
    // leave "VAE 解码中…" on screen in English mode.
    renderPhase();
    // The release row is drawn by script in two places, so the language switch
    // reaches it from here rather than from a `[data-i]` pass.
    renderReleaseSlots();
    renderTools();
    // The frame strip's labels are spoken, not drawn — they still have to
    // follow the language, or a screen reader announces the old one.
    renderFrameStrip();
    renderHistory();
    renderPresets();
    // The negative library is bilingual in its term lists, so a language switch
    // has to repaint which bundles read as applied and what they say.
    renderNegPresets();
    renderModels();
    renderEnv();
    renderWeightsList();
    renderStorage();
    renderExtraDirs();
    refreshOpenDetail();
  }

  /* ======================================================================
     Composer
     ====================================================================== */
  function renderScale() {
    const scales = (S.bootstrap && S.bootstrap.sizes && S.bootstrap.sizes.scales) || {};
    const box = $("scaleSeg");
    box.innerHTML = "";
    Object.entries(scales).forEach(([key, spec]) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = key === S.scale ? "on" : "";
      const label = S.lang === "zh" ? spec.label_zh : spec.label_en;
      const note = (S.lang === "zh" ? spec.note_zh : spec.note_en) || "";
      btn.innerHTML = `${esc(label)}${note ? `<small>${esc(note)}</small>` : ""}`;
      btn.onclick = () => {
        S.scale = key;
        $("outRes").value = spec.output_resolution;
        S.out = spec.output_resolution;
        const pair = sizesForScale()[S.aspect] || sizesForScale()["1:1"];
        if (pair) { S.width = pair[0]; S.height = pair[1]; }
        $("width").value = S.width;
        $("height").value = S.height;
        renderScale(); renderRatios(); renderValues();
      };
      box.appendChild(btn);
    });
  }

  function renderRatios() {
    const box = $("ratios");
    const sizes = sizesForScale();
    box.innerHTML = "";
    aspectList().forEach((aspect) => {
      const pair = sizes[aspect] || [0, 0];
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `ratio${aspect === S.aspect ? " on" : ""}`;
      btn.style.setProperty("--ar", aspect.replace(":", "/"));
      btn.setAttribute("aria-pressed", aspect === S.aspect ? "true" : "false");
      btn.setAttribute("aria-label", `${aspect} · ${pair[0]} × ${pair[1]}`);
      btn.innerHTML = `<i></i><b>${esc(aspect)}</b>`;
      btn.onclick = () => setAspect(aspect);
      box.appendChild(btn);
    });
  }

  function setAspect(aspect) {
    S.aspect = aspect;
    const pair = sizesForScale()[aspect];
    if (pair) { S.width = pair[0]; S.height = pair[1]; }
    $("width").value = S.width;
    $("height").value = S.height;
    renderRatios(); renderValues();
  }

  function renderValues() {
    /* A preset hit is authoritative. Otherwise the label is derived from the
       numbers on screen, so the header never claims a proportion the two fields
       contradict. */
    const shown = ratioLabel(S.width, S.height);
    $("sizeVal").textContent = shown
      ? `${S.width} × ${S.height} · ${shown}`
      : `${S.width} × ${S.height}`;
    if (followsRef()) {
      const last = S.refs[S.refs.length - 1];
      const side = Number($("outRes").value) || 1024;
      $("sizeVal").textContent = last && last.width
        ? tfx("sizeFollow", {
            w: last.width, h: last.height,
            ow: areaSize(side, last.width / last.height)[0],
            oh: areaSize(side, last.width / last.height)[1],
          })
        : tr("followRef");
    }

    const typedW = Number($("width").value);
    const typedH = Number($("height").value);
    const snapped = (typedW && typedW !== floor32(typedW)) || (typedH && typedH !== floor32(typedH));
    $("alignTip").textContent = snapped
      ? tfx("alignTipSnapped", { w: floor32(typedW), h: floor32(typedH) })
      : tr("alignTip");
    $("alignTip").classList.toggle("warn", !!snapped);

    $("sampVal").textContent = `${S.steps} · CFG ${S.cfg.toFixed(1)}`;
    // Only speak up when the value has left the default; next to a live number,
    // a permanent "defaults to 40" reads as a description of that number.
    $("stepsNote").textContent = S.steps === defaultSteps() ? "" : tfx("stepsNote", { n: defaultSteps() });
    $("cfgNote").textContent = S.cfg <= 1 ? tr("cfgOff") : tfx("cfgOn", { v: S.cfg.toFixed(1) });
    $("cfgTip").textContent = S.cfg <= 1 ? tr("cfgTipOff") : tfx("cfgTipOn", { v: S.cfg.toFixed(1) });
    $("outResVal").textContent = `${tr("outputRes")} ${$("outRes").value}`;
    $("outHint").textContent = followsRef() ? tr("followFollow") : tr("followFixed");
    $("refCount").textContent = `${S.refs.length} / ${maxRefs()}`;
    // The cap is a server value, so the sentence is filled rather than written.
    $("refsHint").textContent = tfx("refsHint", { n: maxRefs() });
    $("pCount").textContent = tfx("charUnit", { n: $("prompt").value.length });
    $("sendEditBtn").title = tr("sendEdit");

    $$("#stepQuick button").forEach((btn) => {
      btn.classList.toggle("on", Number(btn.getAttribute("data-steps")) === S.steps);
    });
    renderCount();
    renderPins();
  }

  /* The seed well has two states, and which one is on screen is the only thing
     that says whether the run is pinned. Empty is not "missing" — it is the
     instruction to pick one — so the hint stands in for the value, and a number
     replaces it with the way back. The toggle lives on the wrapper rather than
     on either half, so one rule owns both: there is no state in which the hint
     lies under a number, or a clear button sits over an empty field. */
  function renderSeed() {
    $("seedWrap").classList.toggle("has-value", $("seed").value !== "");
    // Spoken label follows the language, so it is set here, not in the HTML.
    $("seedClear").setAttribute("aria-label", tr("seedClear"));
  }

  /* The stepper: one to four pictures per run. The buttons carry the state at
     the ends of the range (a dimmed button is a reason, not decoration), and a
     hand-typed value is clamped to the same bound the engine enforces.
     `min`/`max` already publish the range, so no ARIA restates it. */
  function renderCount() {
    const input = $("nImages");
    const n = clamp(Number(input.value) || 1, 1, maxImages());
    input.value = n;
    input.max = maxImages();
    $("nImagesMinus").disabled = n <= 1;
    $("nImagesPlus").disabled = n >= maxImages();
    // Spoken labels follow the language, so they are set here, not in the HTML.
    $("nImagesMinus").setAttribute("aria-label", tr("nImagesMinus"));
    $("nImagesPlus").setAttribute("aria-label", tr("nImagesPlus"));
    $("nImagesRange").textContent = tfx("nImagesRange", { max: maxImages() });
  }

  /* Follow-the-reference takes over the output geometry entirely: the engine
     then ignores width/height/aspect, so those controls are disabled while it
     is on. The reference area size still applies — it is what the reference is
     resized to — so that row stays live. */
  function renderFollowState() {
    const on = followsRef();
    ["width", "height"].forEach((id) => { $(id).disabled = on; });
    $("scaleSeg").classList.toggle("is-disabled", on);
    $("ratios").classList.toggle("is-disabled", on);
    $("canvasFields").classList.toggle("is-disabled", on);
    $$("#scaleSeg button").forEach((btn) => { btn.disabled = on; });
    $$("#ratios button").forEach((btn) => { btn.disabled = on; });
    $("outHint").textContent = on ? tr("followFollow") : tr("followFixed");
  }

  function renderMode() {
    ["modeGen", "modeGen2"].forEach((id) => $(id).classList.toggle("on", S.mode === "generate"));
    ["modeEdit", "modeEdit2"].forEach((id) => $(id).classList.toggle("on", S.mode === "edit"));
    $("modeGen").setAttribute("aria-selected", S.mode === "generate" ? "true" : "false");
    $("modeEdit").setAttribute("aria-selected", S.mode === "edit" ? "true" : "false");
    $("refSection").classList.toggle("hidden", S.mode !== "edit");
    $("refSizeBlock").classList.toggle("hidden", S.mode !== "edit");
    $("followToggle").classList.toggle("hidden", S.mode !== "edit");
    renderFollowState();
    $("prompt").placeholder = S.mode === "edit" ? tr("promptPhEdit") : tr("promptPhGen");
    $("promptTip").textContent = S.mode === "edit" ? tr("promptTipEdit") : tr("promptTipGen");
    $("sendEditBtn").classList.toggle("hidden", S.mode === "edit");
    $("genShortcut").innerHTML = `${runKey()} <span>${esc(tr("run"))}</span>`;
    $("negative").placeholder = defaults().negative_placeholder || tr("negativePh");
  }

  function renderTools() {
    // The stage toolbar has no "reuse settings" action: the composer already
    // holds the parameters for the current run. Reusing a *past* record's
    // settings lives in the history detail overlay, where it applies to a
    // record that is not the one on the canvas.
    const enabled = S.hasResult && !S.busy;
    ["downloadBtn", "sendEditBtn"].forEach((id) => {
      const btn = $(id);
      btn.disabled = !enabled;
      btn.classList.toggle("dim", !enabled);
      btn.title = enabled ? "" : tr("needFirstImage");
    });
  }

  function renderRun() {
    $("run").disabled = S.busy;
    $("run2").disabled = S.busy;
    $("runLabel").textContent = S.busy ? tr("running") : tr("run");
    $("runLabel2").textContent = S.busy ? tr("running") : tr("run");
    $("cancel").classList.toggle("show", S.busy);
  }

  function renderStage() {
    $("empty").classList.toggle("hidden", S.hasResult);
    $("viewport").classList.toggle("hidden", !S.hasResult);
    $("zoombar").classList.toggle("hidden", !S.hasResult);
    // One picture needs no chooser; the strip would only repeat what the
    // canvas already shows.
    $("frameStrip").classList.toggle("hidden", !S.hasResult || S.outputFrames.length < 2);
  }

  /* The frames of the last run, as a row of thumbnails under the canvas.
     A separate row rather than a floating overlay: the pictures are the point
     of the strip, and covering the canvas to show them would be a poor trade. */
  function renderFrameStrip() {
    const box = $("frameList");
    if (!box) return;
    box.innerHTML = "";
    // Nothing to choose between: leave the strip empty rather than hidden with
    // stale thumbnails of a previous run inside it.
    if (S.outputFrames.length < 2) return;
    S.outputFrames.forEach((frame, index) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `reftile${index === S.outputIndex ? " on" : ""}`;
      btn.dataset.pos = String(index);
      btn.innerHTML = `<img src="${esc(frame.thumb || frame.url)}" alt="" loading="lazy" /><i>${index + 1}</i>`;
      btn.setAttribute("aria-label", tfx("tileFrame", { n: index + 1 }));
      if (index === S.outputIndex) btn.setAttribute("aria-current", "true");
      btn.onclick = () => selectFrame(index);
      box.appendChild(btn);
    });
  }

  function selectFrame(index) {
    const frame = S.outputFrames[index];
    if (!frame) return;
    S.outputIndex = index;
    S.currentImage = frame.url;
    const wrap = $("zoomwrap");
    wrap.classList.remove("enter");
    void wrap.offsetWidth;
    wrap.classList.add("enter");
    canvasZoom.load(frame.w || S.width, frame.h || S.height, frame.url);
    renderFrameStrip();
  }

  /* `frames` is every picture of the run; the first one opens on the canvas.
     Passing one frame is the ordinary case and behaves exactly as before. */
  function showResult(frames, w, h) {
    const list = (Array.isArray(frames) ? frames : [frames]).filter(Boolean);
    S.outputFrames = list.map((entry) => (typeof entry === "string" ? { url: entry } : entry));
    S.outputIndex = 0;
    S.currentImage = (S.outputFrames[0] || {}).url || null;
    S.hasResult = !!S.currentImage;
    const wrap = $("zoomwrap");
    wrap.classList.remove("enter");
    void wrap.offsetWidth;
    wrap.classList.add("enter");
    if (S.currentImage) canvasZoom.load(w, h, S.currentImage);
    renderStage();
    renderFrameStrip();
    renderTools();
  }

  function renderPins() {
    // The pill names the device and the model being worked with. It deliberately
    // leaves out the weight source: `S.hub` is the *preference*, not where the
    // weights came from — `resolve_local_hub` on the server silently falls back
    // to whichever source actually holds a copy, and the job log says so while
    // the pill would still have shown the preference. It was wrong on exactly
    // the configuration it claimed to describe. The model sheet behind this pill
    // reports `local_hub`, the real location, and every history record stores the
    // resolved value, so nothing is lost by dropping it here.
    $("pillDevice").textContent = ((S.bootstrap && S.bootstrap.device) || {}).device_name || "—";
    $("pillModel").textContent = `· ${modelLabel(S.modelKey)}`;
  }

  /* ---------- References ---------- */
  function renderRefs() {
    const box = $("refGrid");
    box.innerHTML = "";
    S.refs.forEach((item, index) => {
      const el = document.createElement("div");
      el.className = "ref-item";
      el.innerHTML = `<img src="${item.url}" alt="" /><button type="button" aria-label="${esc(tr("delete"))}">${icon("i-x", 11)}</button><span>${index + 1}</span>`;
      el.querySelector("button").onclick = () => {
        URL.revokeObjectURL(item.url);
        S.refs.splice(index, 1);
        renderRefs(); renderValues();
      };
      box.appendChild(el);
    });
  }

  function addFiles(fileList) {
    const limit = maxRefs();
    let rejected = 0;
    Array.from(fileList || []).forEach((file) => {
      if (!file.type.startsWith("image/") || S.refs.length >= limit) { rejected += 1; return; }
      const item = { file, url: URL.createObjectURL(file), width: 0, height: 0 };
      const probe = new Image();
      probe.onload = () => {
        item.width = probe.naturalWidth;
        item.height = probe.naturalHeight;
        if (followsRef()) renderValues();
      };
      probe.src = item.url;
      S.refs.push(item);
    });
    if (rejected) toast("err", tfx("maxRefs", { n: limit }), "");
    renderRefs(); renderValues();
  }

  /* ======================================================================
     History
     ====================================================================== */
  function dayLabel(iso) {
    const date = new Date(iso);
    if (!iso || Number.isNaN(date.getTime())) return tr("unknown");
    const today = new Date();
    const same = (a, b) => a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
    if (same(date, today)) return tr("today");
    if (same(date, new Date(today.getTime() - 86400000))) return tr("yesterday");
    return date.toLocaleDateString(S.lang === "zh" ? "zh-CN" : "en-US", { month: "short", day: "numeric" });
  }
  function clockOf(iso) {
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "";
    return date.toLocaleTimeString(S.lang === "zh" ? "zh-CN" : "en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
  }
  function fullTime(iso) {
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return iso || tr("unknown");
    return date.toLocaleString(S.lang === "zh" ? "zh-CN" : "en-US", {
      year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
    });
  }
  /* The server does the filtering now, so this is just the current page set.
     Keeping a second local filter here would disagree with it: the server also
     matches mode and model name, and it reaches records past the first page. */
  const filteredHistory = () => S.history;

  function renderHistory() {
    const box = $("historyList");
    if (!box) return;
    box.innerHTML = "";
    const items = filteredHistory();
    if (!items.length) {
      // A first load (or a search in flight) has no records yet — say so
      // instead of claiming the list is empty.
      if (S.historyLoading) {
        box.innerHTML = `<p class="hint-row">${esc(tr("loadingMore"))}</p>`;
      } else {
        box.innerHTML = S.historyQuery
          ? `<p class="hint-row">${esc(tfx("historyNoMatch", { q: S.historyQuery }))}</p>`
          : `<p class="hint-row">${esc(tr("historyEmpty"))}<br>${esc(tr("historyHint"))}</p>`;
      }
      return;
    }
    let lastDay = "";
    items.forEach((item) => {
      const day = dayLabel(item.created_at);
      if (day !== lastDay) {
        lastDay = day;
        const head = document.createElement("div");
        head.className = "hday";
        head.textContent = day;
        box.appendChild(head);
      }
      const thumb = item.thumb_url || item.image_url;
      const title = (item.prompt || "").trim() || item.id;
      // Second line: mode · size · time. Time is pushed right by CSS and its
      // tooltip carries the full date, since the row only shows the clock.
      const meta = [
        item.mode === "edit" ? tr("edit") : tr("generate"),
        item.width && item.height ? `${item.width}×${item.height}` : "",
      ].filter(Boolean);
      // A run that made several pictures says so on the thumbnail: the row can
      // only show one of them, and "there are three more" is the useful fact.
      const count = item.output_count || 1;

      const row = document.createElement("div");
      row.className = `hrow${S.detailId === item.id ? " on" : ""}`;
      row.dataset.id = item.id;
      row.tabIndex = 0;
      row.setAttribute("role", "button");
      row.innerHTML =
        `<div class="th">${thumb ? `<img src="${esc(thumb)}" alt="" loading="lazy" />` : ""}` +
        `${count > 1 ? `<i class="thcount">×${count}</i>` : ""}</div>` +
        `<div class="mt"><b title="${esc(title)}">${esc(title)}</b>` +
        `<span class="mrow">${meta.map((part) => `<i>${esc(part)}</i>`).join("")}` +
        `<i class="when" title="${esc(fullTime(item.created_at))}">${esc(clockOf(item.created_at))}</i></span></div>`;
      row.onclick = () => openDetail(item);
      row.onkeydown = (event) => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); openDetail(item); }
      };
      box.appendChild(row);
    });

    // Paging footer. `total` comes from the server, so "no more" is a fact
    // rather than a guess from the page size.
    const foot = document.createElement("div");
    foot.className = "hmore";
    if (historyHasMore()) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn sm";
      btn.id = "histMore";
      btn.disabled = S.historyLoading;
      btn.textContent = S.historyLoading ? tr("loadingMore") : tr("loadMore");
      btn.onclick = () => loadHistory({ append: true }).catch(reportError);
      foot.appendChild(btn);
    } else {
      const done = document.createElement("span");
      done.className = "hmore-done";
      done.textContent = tr("noMore");
      foot.appendChild(done);
    }
    box.appendChild(foot);
  }

  /* One page of history. Paging is manual: the list renders a "load more"
     button while more records exist, so the first paint stays small however
     many records accumulate. */
  const HISTORY_PAGE = 80;

  async function loadHistory(options) {
    const append = !!(options && options.append);
    const offset = append ? S.history.length : 0;
    const query = S.historyQuery || "";
    const params = new URLSearchParams({ limit: String(HISTORY_PAGE), offset: String(offset) });
    if (query) params.set("q", query);
    S.historyLoading = true;
    renderHistory();
    try {
      const data = await api(`/api/jobs?${params.toString()}`);
      const items = data.items || [];
      S.history = append ? S.history.concat(items) : items;
      S.historyTotal = typeof data.total === "number" ? data.total : S.history.length;
    } finally {
      S.historyLoading = false;
    }
    renderHistory();
  }

  const historyHasMore = () => S.history.length < S.historyTotal;

  function downloadItem(item) {
    if (!item.image_url) return;
    const a = document.createElement("a");
    a.href = item.image_url;
    a.download = `imgen-${item.seed ?? "nosd"}.png`;
    a.click();
    toast("ok", tr("toastSaved"), `${item.width || "?"}×${item.height || "?"}`);
  }

  /* Reuse writes the left panel only — browsing history must not touch the canvas. */
  function applyItemParams(item) {
    const p = item.params || {};
    // Park the outgoing panel first: reuse can switch modes, and the set being
    // left behind must not be lost.
    if (item.mode) parkAndSwitch(item.mode);
    if (p.scale && scaleSpec(p.scale)) S.scale = p.scale;
    if (p.aspect) S.aspect = p.aspect;
    const size = sizesForScale()[S.aspect];
    S.width = p.width || item.width || (size && size[0]) || S.width;
    S.height = p.height || item.height || (size && size[1]) || S.height;
    $("width").value = S.width;
    $("height").value = S.height;
    if (p.output_resolution) { $("outRes").value = p.output_resolution; S.out = p.output_resolution; }

    S.steps = p.steps || S.steps;
    S.cfg = p.true_cfg_scale != null ? Number(p.true_cfg_scale) : S.cfg;
    $("steps").value = S.steps;
    $("stepsNum").value = S.steps;
    $("cfg").value = S.cfg;
    $("cfgNum").value = S.cfg.toFixed(1);
    $("seed").value = item.seed != null ? item.seed : "";
    // Reusing a run carries its seed, and that write is not typing: the hint /
    // clear pair has to be re-read or the field would claim to be empty.
    renderSeed();
    $("nImages").value = p.num_images || 1;
    $("kv").setAttribute("aria-checked", p.use_kv_cache === false ? "false" : "true");
    $("rgba").setAttribute("aria-checked", p.transparent ? "true" : "false");
    $("follow").setAttribute("aria-checked", p.follow_ref_aspect === false ? "false" : "true");
    if (p.vae_tiling != null) $("vaeTiling").value = vaeTilingValue(p.vae_tiling);
    const source = p.model_key ? modelByKey(p.model_key) : null;
    // Only adopt a model that is actually usable; the engine resolves the
    // source itself if the weights live in the other hub.
    if (source && source.downloaded_any) S.modelKey = source.key;
    renderPins();

    $("prompt").value = item.prompt || "";
    $("negative").value = item.negative_prompt || "";

    // These values are now the panel's state, so they become the snapshot for
    // this mode — otherwise switching away and back would resurrect the old set.
    S.params[S.mode] = readParams();
    // Same ordering rule as `setMode`: the chips are rebuilt by the scale and
    // aspect renders, so the follow state must be applied after them.
    renderScale(); renderRatios(); renderMode(); renderValues();
    renderFollowState(); renderTools();
    toast("ok", tr("toastReuseKept"), tr("toastReuseKeptDetail"));
  }

  /* ======================================================================
     History detail overlay
     ====================================================================== */
  const STATUS = {
    running: ["statusRunning", "i-alert", "is-warn"],
    succeeded: ["statusSucceeded", "i-check", ""],
    failed: ["statusFailed", "i-alert", "is-err"],
    cancelled: ["statusCancelled", "i-x", "is-warn"],
  };

  function openDetail(item) {
    S.detailId = item.id;
    // A record opens on its first frame; the strip switches from there.
    S.detailCurrent = { url: item.image_url || null, index: 0 };

    renderDetailMeta(item);
    renderDetailRefs(item);
    showDetailSource(item, 0, null);
    renderDetailFooter(item);
    $("detail").classList.remove("hidden");
    // Only measurable once the overlay is displayed.
    wirePromptFolds();
    syncOverlayState();
    renderHistory();
  }

  /* Key/value cells for the detail panel: flat, no gridlines (see `.kv.flat`).
     The settings sheet keeps the bordered `.kv` variant. */
  function kvCells(rows) {
    return `<div class="kv flat">${rows
      .map(([k, v, full]) => `<div${full ? ' class="full"' : ""}><span class="k">${esc(k)}</span><span class="v">${esc(v == null || v === "" ? tr("unknown") : v)}</span></div>`)
      .join("")}</div>`;
  }

  /* A group heading with an optional trailing action, e.g. copy the prompt. */
  function grpHead(label, action) {
    return `<div class="grp"><span>${esc(label)}</span>` +
      (action
        ? `<button type="button" class="grp-act" id="${esc(action.id)}" title="${esc(action.title)}" aria-label="${esc(action.title)}">${icon("i-copy", 12)}<span class="lbl">${esc(action.label)}</span></button>`
        : "") +
      `</div>`;
  }

  async function copyText(text, label) {
    const value = String(text || "");
    if (!value) return;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(value);
      } else {
        // http:// on a LAN address is not a secure context, so fall back to the
        // legacy path rather than failing silently.
        const ta = document.createElement("textarea");
        ta.value = value;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
      }
      toast("ok", tr("toastCopied"), label || "");
    } catch (err) {
      reportError(err);
    }
  }

  /* The spec sheet. Groups run from what the record *is* (when it was made, in
     which mode, which id) through what it produced to how it was produced. The
     panel header carries only the word "详情" now, so anything the old top band
     used to say has to be said here or it is lost — that is why status, mode
     and the timestamp all live in the first group. */
  function renderDetailMeta(item) {
    const p = item.params || {};
    const vaeRaw = p.vae_tiling;
    const vae = vaeRaw === true || vaeRaw === "true" ? tr("vOn") : vaeRaw === false || vaeRaw === "false" ? tr("vOff") : tr("vAuto");
    const status = STATUS[item.status] || STATUS.succeeded;
    const statusText = item.status === "failed" && item.error
      ? `${tr(status[0])} · ${item.error}`
      : tr(status[0]);
    $("dMeta").innerHTML =
      grpHead(tr("grpTask")) +
      kvCells([
        [tr("colMode"), item.mode === "edit" ? tr("modeEdit") : tr("modeGen")],
        [tr("colStatus"), statusText],
        // The timestamp gets the full width: it is long, and half a 360px
        // column is not enough for it to stay on one line.
        [tr("colCreated"), fullTime(item.created_at), true],
      ]) +
      grpHead(tr("grpImage")) +
      kvCells([
        [tr("colSize"), item.width && item.height ? `${item.width} × ${item.height}` : ""],
        [tr("colAspect"), p.aspect],
        [tr("colScale"), p.scale === "1k" ? tr("scale1k") : tr("scale2k")],
        [tr("colDuration"), item.duration_ms ? `${(item.duration_ms / 1000).toFixed(1)} s` : ""],
      ]) +
      grpHead(tr("grpSampling")) +
      kvCells([
        [tr("colSteps"), p.steps],
        [tr("colCfg"), p.true_cfg_scale != null ? Number(p.true_cfg_scale).toFixed(1) : ""],
        [tr("colSeed"), item.seed],
        [tr("colN"), p.num_images],
      ]) +
      grpHead(tr("grpEnv")) +
      kvCells([
        [tr("colModel"), modelLabel(item.model_key)],
        [tr("colHub"), hubLabel(item.hub)],
        [tr("colKv"), p.use_kv_cache === false ? tr("vOff") : tr("vOn")],
        [tr("colVae"), vae],
      ]) +
      grpHead(tr("grpPrompt"), { id: "dCopyPrompt", label: tr("copy"), title: tr("copyPrompt") }) +
      promptBlock(item.prompt, "dPrompt") +
      (item.negative_prompt
        ? grpHead(tr("negative"), { id: "dCopyNegative", label: tr("copy"), title: tr("copyNegative") }) +
          promptBlock(item.negative_prompt, "dNegative")
        : "");
    $("dCopyPrompt").onclick = () => copyText(item.prompt, tr("grpPrompt"));
    if ($("dCopyNegative")) $("dCopyNegative").onclick = () => copyText(item.negative_prompt, tr("negative"));
  }

  /* The prompt folds are wired after the overlay is on screen. Measuring while
     `#detail` is still `display: none` reads every element as zero-sized, so the
     overflow test would say "fits" for any prompt and the link would never
     appear. */
  function wirePromptFolds() {
    wirePromptFold("dPrompt");
    wirePromptFold("dNegative");
  }

  /* A prompt block that folds when it is long. It renders already clamped, so a
     verbose prompt never flashes its full height first; the toggle is only
     revealed when the clamped box really is hiding text — a "show all" link
     under two lines is a control that does nothing, and people learn to ignore
     those. */
  function promptBlock(text, id) {
    return `<div class="dprompt clamped" id="${id}">${esc(text || "")}</div>` +
      `<button type="button" class="prompt-more hidden" id="${id}More" aria-expanded="false"><span></span>${icon("i-chev", 12)}</button>`;
  }

  function wirePromptFold(id) {
    const box = $(id);
    const more = $(`${id}More`);
    if (!box || !more) return;
    // Measured while folded: with the fold expressed as a `max-height`, the box
    // reports the full content height in `scrollHeight` and the visible height in
    // `clientHeight`, so comparing them tells us whether anything is hidden. The
    // tolerance is small on purpose — the heights are exact here, so anything
    // larger would leave a sliver of a line hidden with no way to reveal it.
    const overflows = box.scrollHeight > box.clientHeight + 2;
    more.classList.toggle("hidden", !overflows);
    box.classList.toggle("clamped", overflows);
    if (!overflows) return;
    let open = false;
    const paint = () => {
      box.classList.toggle("clamped", !open);
      more.setAttribute("aria-expanded", open ? "true" : "false");
      more.querySelector("span").textContent = open ? tr("showLess") : tr("showAll");
    };
    more.onclick = () => { open = !open; paint(); };
    paint();
  }

  /* The detail filmstrip. It carries everything the record has: every frame of
     the run, then the references it was given — a multi-image generation has
     just as much to choose from as an edit does.

     Frames come first and references after, in that order, because that is the
     order they were made. `kind` + `index` say which group a tile belongs to;
     the two groups number themselves from 1, and a wider gap marks the seam
     between them. Which tile is showing is carried by the `on` class (a
     highlight), so no text label is needed. `aria-current` says the same thing
     to assistive tech. */
  /* The picture picker in the panel. It carries everything the record has:
     every frame of the run, then the references it was given — a multi-image
     generation has just as much to choose from as an edit does.

     Frames come first and references after, in that order, because that is the
     order they were made. `kind` + `index` say which group a tile belongs to;
     the two groups number themselves from 1, and a wider gap marks the seam
     between them. Which tile is showing is carried by the `on` class (a
     highlight), so no text label is needed. `aria-current` says the same thing
     to assistive tech.

     References are shown for an edit even when there is only one: it is the
     other half of what happened, so the frame it produced never appears alone. */
  function renderDetailRefs(item) {
    const outs = item.image_urls || (item.image_url ? [item.image_url] : []);
    const refs = item.ref_urls || [];
    const showOuts = outs.length > 1;
    const showRefs = refs.length > 0;
    const show = showOuts || showRefs;
    $("dRefs").classList.toggle("hidden", !show);
    if (!show) return;
    const list = $("dRefList");
    list.innerHTML = "";
    // Each group is named on its own first tile rather than by a heading above
    // the strip. The strip holds two kinds of picture — what the run made and
    // what it was given — and a number alone cannot say which is which, so the
    // first tile of each group carries the word.
    const tiles = outs
      .map((url, index) => ({
        url, kind: "out", index, pos: index, label: String(index + 1),
        tag: index === 0 ? tr("framesLabel") : null,
      }))
      .concat(refs.map((url, index) => ({
        url, kind: "ref", index, pos: outs.length + index, label: String(index + 1),
        tag: index === 0 ? tr("refsLabel") : null,
      })));
    tiles.forEach((tile) => {
      const btn = document.createElement("button");
      btn.type = "button";
      // The first reference opens a new group; the gap comes from CSS rather
      // than an extra element, so every child of the strip stays a tile.
      btn.className = `reftile${tile.kind === "ref" && tile.index === 0 && outs.length > 0 ? " grp-start" : ""}`;
      btn.dataset.kind = tile.kind;
      btn.dataset.pos = String(tile.pos);
      btn.dataset.index = String(tile.index);
      btn.innerHTML = `<img src="${esc(tile.url)}" alt="" loading="lazy" />` +
        (tile.tag ? `<span class="gtag">${esc(tile.tag)}</span>` : "") +
        `<i>${esc(tile.label)}</i>`;
      // Describes the tile itself; `aria-current` marks the one on screen.
      btn.setAttribute("aria-label", tile.kind === "out"
        ? tfx("tileFrame", { n: tile.index + 1 })
        : tfx("tileRef", { n: tile.index + 1 }));
      btn.onclick = () => (tile.kind === "out"
        ? showDetailSource(item, tile.index, null)
        : showDetailSource(item, null, tile.index));
      list.appendChild(btn);
    });
    syncDetailTiles();
  }

  /* The tile on screen: brighter frame, full opacity, and a slight lift. The
     rest stay dimmed so the current one reads at a glance. Frames and
     references are separate index spaces, so the kind is part of the match. */
  function syncDetailTiles() {
    const ref = S.detailRefIndex;
    $$("#dRefList .reftile").forEach((tile) => {
      const isFrame = tile.dataset.kind === "out";
      const pos = Number(tile.dataset.index);
      const on = isFrame ? (ref == null && pos === S.detailCurrent.index) : (ref != null && pos === ref);
      tile.classList.toggle("on", on);
      if (on) tile.setAttribute("aria-current", "true");
      else tile.removeAttribute("aria-current");
    });
  }

  /* A record with no image (failed or cancelled) must not inherit whatever the
     viewer was showing before. Exactly one of `outIndex` / `refIndex` is set:
     the first picks a frame of the run, the second a reference image. */
  function showDetailSource(item, outIndex, refIndex) {
    const outs = item.image_urls || (item.image_url ? [item.image_url] : []);
    const refs = item.ref_urls || [];
    const viewingOutput = refIndex == null;
    const url = viewingOutput ? outs[outIndex || 0] : refs[refIndex];
    // The download button and the strip both follow what is on screen.
    // `detailRefIndex` is null while a frame is showing.
    if (viewingOutput) {
      S.detailCurrent = { url: url || null, index: outIndex || 0 };
      S.detailRefIndex = null;
    } else {
      S.detailRefIndex = refIndex;
    }
    if (!url) {
      detailZoom.clear();
      $("dview").classList.add("no-image");
      $("dview").querySelector(".vmain").classList.add("no-image");
      $("dEmpty").classList.remove("hidden");
      $("dEmptyText").textContent = viewingOutput ? tr("noOutput") : tr("noRefImage");
      syncDetailTiles();
      renderStageId(item);
      return;
    }
    $("dview").classList.remove("no-image");
    $("dview").querySelector(".vmain").classList.remove("no-image");
    $("dEmpty").classList.add("hidden");
    // The record's width/height describe the *output*. A reference is stored at
    // its own size, so treat these as a first guess only — the viewer replaces
    // them with the decoded bitmap's real dimensions once it loads.
    detailZoom.load(item.width || 1024, item.height || 1024, url);
    syncDetailTiles();
    // The id follows the picture, so it is refreshed wherever the picture is.
    renderStageId(item);
  }

  /* The id of the picture on the stage.

     Each file of a multi-image run has its own id — `{task}` for the first frame,
     `{task}_1`, `{task}_2` … for the rest, which is exactly how they are named on
     disk and in their URLs. Showing the record id instead made four different
     pictures look like the same one. The id is read from the URL rather than
     rebuilt from the task id, so it follows whatever the engine actually named
     the file.

     A reference image has no id of its own: it is stored as
     `refs/{task}/ref_{nn}.png` and addressed by index, and the upload does not
     keep the original filename. So the box is hidden rather than filled with a
     made-up value. */
  function renderStageId(item) {
    const box = $("dStageId");
    const imageId = detailImageId(item);
    box.classList.toggle("hidden", !imageId);
    if (imageId) $("dImageId").textContent = imageId;
  }

  /* The id of the picture currently on screen, or null while a reference is
     showing. Both kinds are addressed the same way — the id is the last path
     segment of the URL — so nothing has to second-guess how the engine named
     things. */
  function detailImageId(item) {
    if (S.detailRefIndex != null) return null;
    const url = (item.image_urls || [])[S.detailCurrent.index || 0] || item.image_url;
    return url ? decodeURIComponent(String(url).split("/").pop()) : null;
  }

  /* Which picture the actions act on. The buttons follow the tile the viewer is
     showing, so "download" means the one on screen rather than always the first
     — the whole point of a multi-image run is choosing between them. The frame
     counter that used to say so is gone; the highlighted tile says it. */
  const detailFrame = () => S.detailCurrent;

  function renderDetailFooter(item) {
    // Delete is destructive and acts on the record; the other two act on the
    // picture. It keeps its trash glyph and drops the word from the button: the
    // glyph is read on its own, and the word was the only thing pushing the
    // English footer past the panel — three labels ask for 358px where the
    // panel has 328. What the eye loses `title` says on hover and `aria-label`
    // says to a screen reader, so the button is never unlabelled.
    // The pair is one element so the footer can still move it whole if the
    // words ever outgrow the panel: half a row beats half a button.
    $("dFoot").innerHTML =
      `<button type="button" class="btn danger" id="dDelete" title="${esc(tr("delete"))}" aria-label="${esc(tr("delete"))}">${icon("i-trash")}</button>` +
      `<span class="foot-pair">` +
        `<button type="button" class="btn" id="dReuse" aria-label="${esc(tr("reuse"))}">${icon("i-reuse")}<span class="lbl">${esc(tr("reuse"))}</span></button>` +
        `<button type="button" class="btn" id="dSend" aria-label="${esc(tr("sendEdit"))}">${icon("i-send")}<span class="lbl">${esc(tr("sendEdit"))}</span></button>` +
      `</span>`;
    const hasImage = !!item.image_url;
    $("dReuse").onclick = () => { closeDetail(); applyItemParams(item); };
    $("dSend").onclick = () => { if (hasImage) { closeDetail(); sendToEdit(detailFrame().url || item.image_url); } };
    $("dDelete").onclick = () => askDelete(item);
    // A record with no picture (failed or cancelled) has nothing to send, so the
    // control is genuinely disabled — not merely dimmed, or it stays focusable
    // and announces itself as usable.
    $("dSend").disabled = !hasImage;
    $("dSend").classList.toggle("dim", !hasImage);
    // Download sits with the zoom controls under the picture, but it is rendered
    // here so it keeps following the tile on screen.
    $("dDownload").onclick = () => downloadItem({ ...item, image_url: detailFrame().url || item.image_url });
    $("dDownload").disabled = !hasImage;
    $("dDownload").classList.toggle("dim", !hasImage);
    $("dDownload").title = tr("downloadImg");
    $("dDownload").setAttribute("aria-label", tr("downloadImg"));
  }

  /* Deleting asks first, and the message gets its own row: the record's prompt
     can be long, and sharing a row with two buttons is how a confirmation ends
     up unreadable at exactly the moment it matters. */
  function askDelete(item) {
    $("dFoot").innerHTML =
      `<div class="confirm">` +
        `<div class="confirm-head">${icon("i-alert", 16)}` +
          `<span>${esc(tr("confirmDeleteAsk"))}<b>${esc((item.prompt || "").slice(0, 60))}</b></span>` +
        `</div>` +
        `<div class="confirm-btns">` +
          `<button type="button" class="btn" id="cNo">${esc(tr("confirmDeleteNo"))}</button>` +
          `<button type="button" class="btn btn-danger-solid" id="cYes">${esc(tr("confirmDeleteYes"))}</button>` +
        `</div>` +
      `</div>`;
    $("cNo").onclick = () => renderDetailFooter(item);
    $("cYes").onclick = async () => {
      try {
        await api(`/api/jobs/${item.id}`, { method: "DELETE" });
      } catch (err) {
        return reportError(err);
      }
      if (S.detailId === item.id) closeDetail();
      toast("ok", tr("toastDeleted"), tr("toastDeletedDetail"));
      await loadHistory().catch(reportError);
    };
  }

  function closeDetail() {
    $("detail").classList.add("hidden");
    S.detailId = null;
    // Filling the window is a look at the picture, not a state of the app: the
    // next record opens two-column, with its panel where it belongs.
    setDetailWide(false);
    syncOverlayState();
    renderHistory();
  }

  /* The viewer's "give me the whole sheet" toggle. The panel leaves the grid
     and the viewer takes its column — the record itself is untouched, which is
     the point: the same picture, more room, and the panel back the moment you
     want it.
     Not the Fullscreen API: the picture expands inside the app's own overlay,
     so Esc, the click-outside-to-close and the window chrome all keep behaving
     exactly as they did, and the panel is only hidden rather than removed —
     the arrow keys still walk the history while it is away. */
  function renderDetailExpand() {
    const btn = $("dExpand");
    const wide = S.detailWide;
    // The class goes on `#detail`, not on the sheet: the overlay's own padding
    // has to go too, or the picture keeps a 20px frame of the window it was
    // just handed.
    $("detail").classList.toggle("wide", wide);
    btn.setAttribute("aria-pressed", wide ? "true" : "false");
    // The label names the next click, so it flips with the state.
    btn.title = tr(wide ? "paneRestore" : "paneFill");
    btn.setAttribute("aria-label", btn.title);
  }

  function setDetailWide(on) {
    const wide = !!on;
    if (S.detailWide === wide) return;
    S.detailWide = wide;
    renderDetailExpand();
    // A fitted picture was sized against the old viewer, so it has to be
    // measured again or it keeps the box the other layout gave it. `resize`
    // only re-fits: a 1:1 or a free zoom is the user's own number and stays.
    detailZoom.resize();
    // Folds are measured, and the panel cannot be measured while it is
    // `display: none` — so a record rendered during a fill would come back
    // with every prompt open. Re-measure on the way out, never on the way in.
    if (!wide && !$("detail").classList.contains("hidden")) wirePromptFolds();
  }

  function navDetail(dir) {
    if (!S.detailId) return;
    const list = filteredHistory();
    const index = list.findIndex((item) => item.id === S.detailId);
    if (index < 0) return;
    const next = list[clamp(index + dir, 0, list.length - 1)];
    if (next && next.id !== S.detailId) openDetail(next);
  }

  /* Sending a picture is sending a picture: it adds one reference image and
     nothing else. It deliberately does *not* touch `#prompt` — the box belongs
     to whoever is typing in it, and a run's own prompt is a click away through
     复用参数 (`applyItemParams`), which is the control that owns restoring a
     record's settings. Seeding the prompt here used to overwrite a draft
     silently, with no confirmation and no undo, while the toast said only
     "sent as a reference" and told the user to name it themselves. */
  async function sendToEdit(url) {
    if (!url) return;
    setMode("edit");
    try {
      const res = await fetch(url);
      const blob = await res.blob();
      addFiles([new File([blob], "reference.png", { type: "image/png" })]);
    } catch (_) {
      toast("err", tr("errFailed"), "");
    }
    renderValues();
    toast("ok", tr("toastToEdit"), tr("toastToEditDetail"));
  }

  /* ======================================================================
     Popovers
     ====================================================================== */
  function closePops() {
    $("popPresets").classList.add("hidden");
    $("popNeg").classList.add("hidden");
    $("popModel").classList.add("hidden");
    $("btnPresets").setAttribute("aria-expanded", "false");
    $("btnPresets").classList.remove("on");
    $("btnNegPresets").setAttribute("aria-expanded", "false");
    $("btnNegPresets").classList.remove("on");
    $("statusBtn").setAttribute("aria-expanded", "false");
    $("statusBtn").classList.remove("on");
  }

  function placeBelow(pop, anchor) {
    const appRect = $("app").getBoundingClientRect();
    const anchorRect = anchor.getBoundingClientRect();
    const w = pop.offsetWidth;
    const h = pop.offsetHeight;
    const left = clamp(anchorRect.right - appRect.left - w, 8, appRect.width - w - 8);
    let top = anchorRect.bottom - appRect.top + 8;
    if (top + h > appRect.height - 12) top = Math.max(58, appRect.height - 12 - h);
    pop.style.left = `${left}px`;
    pop.style.top = `${top}px`;
    pop.style.maxHeight = `${appRect.height - top - 12}px`;
    pop.style.setProperty("--caret", `${clamp(anchorRect.left - appRect.left + anchorRect.width / 2 - left - 4, 14, w - 22)}px`);
  }

  /* Anchored to the right of the button so the prompt box stays readable. */
  function placePresets() {
    const pop = $("popPresets");
    if (pop.classList.contains("hidden")) return;
    const appRect = $("app").getBoundingClientRect();
    const btnRect = $("btnPresets").getBoundingClientRect();
    const paneRect = $("composer").getBoundingClientRect();

    if (isNarrow()) {
      pop.classList.remove("side-right");
      pop.style.width = `${Math.round(paneRect.width - 20)}px`;
      pop.style.left = `${Math.round(paneRect.left - appRect.left + 10)}px`;
      pop.style.top = `${Math.round(paneRect.top - appRect.top + 14)}px`;
      pop.style.maxHeight = `${Math.round(paneRect.height - 24)}px`;
      return;
    }
    pop.classList.add("side-right");
    const width = 400;
    let left = Math.round(paneRect.right - appRect.left) + 10;
    if (left + width > appRect.width - 12) left = Math.max(8, appRect.width - 12 - width);
    let top = Math.round(btnRect.top - appRect.top) - 9;
    let maxHeight = appRect.height - top - 16;
    if (maxHeight < 300) {
      top = Math.max(58, appRect.height - 16 - 460);
      maxHeight = appRect.height - top - 16;
    }
    pop.style.width = `${width}px`;
    pop.style.left = `${left}px`;
    pop.style.top = `${top}px`;
    pop.style.maxHeight = `${Math.max(260, maxHeight)}px`;
    const caret = Math.round(btnRect.top - appRect.top) + Math.round(btnRect.height / 2) - top - 4;
    pop.style.setProperty("--carety", `${clamp(caret, 16, Math.max(16, (pop.offsetHeight || 320) - 24))}px`);
  }

  const presetPool = () => {
    const packs = (S.bootstrap && S.bootstrap.prompts) || {};
    return packs[S.mode === "edit" ? "edit" : "generate"] || [];
  };
  const presetTitle = (preset) => (S.lang === "zh" ? preset.title_zh : preset.title_en) || preset.prompt;

  function renderPresets() {
    const cats = presetPool();
    if (!cats.length || !$("presetCats")) return;
    if (!S.presetCat || !cats.some((c) => c.id === S.presetCat)) S.presetCat = cats[0].id;

    const query = ($("presetSearch").value || "").trim().toLowerCase();
    const total = cats.reduce((sum, cat) => sum + (cat.prompts || []).length, 0);

    $("presetCats").innerHTML = "";
    cats.forEach((cat) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `cat${cat.id === S.presetCat && !query ? " on" : ""}`;
      btn.textContent = `${cat.icon || ""} ${S.lang === "zh" ? cat.zh : cat.en}`.trim();
      // This handler rebuilds #presetCats, so the event target is detached by the
      // time it bubbles to document and the outside-click check would misread it.
      btn.onclick = (event) => { event.stopPropagation(); S.presetCat = cat.id; renderPresets(); };
      $("presetCats").appendChild(btn);
    });

    let pool;
    if (query) {
      pool = cats.flatMap((cat) => cat.prompts || []);
      pool = pool.filter((p) => `${presetTitle(p)} ${p.prompt}`.toLowerCase().includes(query));
      $("presetCount").textContent = tfx("countMatch", { m: pool.length, t: total });
    } else {
      const active = cats.find((cat) => cat.id === S.presetCat) || cats[0];
      pool = active.prompts || [];
      $("presetCount").textContent = tfx("countTotal", { n: total });
    }

    revealActiveCat();
    updateCatNav();

    const list = $("presetList");
    list.innerHTML = "";
    if (!pool.length) {
      list.innerHTML = `<p class="empty-row">${esc(tfx("presetsEmpty", { q: $("presetSearch").value }))}</p>`;
      return;
    }
    pool.forEach((preset) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "pcard";
      btn.innerHTML = `<b>${esc(presetTitle(preset))}</b><span>${esc(preset.prompt)}</span>`;
      btn.onclick = (event) => { event.stopPropagation(); pickPreset(preset); };
      list.appendChild(btn);
    });
  }

  /* The category strip scrolls, but its scrollbar is hidden — show a chevron
     only on the side that actually has more chips. */
  function updateCatNav() {
    const row = $("presetCats");
    const wrap = $("catsWrap");
    if (!row || !wrap) return;
    const max = row.scrollWidth - row.clientWidth;
    wrap.classList.toggle("can-prev", max > 2 && row.scrollLeft > 2);
    wrap.classList.toggle("can-next", max > 2 && max - row.scrollLeft > 2);
  }

  /* Bring the active chip into view, but never nudge a chip the user can
     already see — re-centring on every click makes the strip feel jumpy. */
  function revealActiveCat() {
    const row = $("presetCats");
    const on = row && row.querySelector(".cat.on");
    if (!on) return;
    const max = row.scrollWidth - row.clientWidth;
    if (max <= 2) return;
    const left = row.scrollLeft;
    const right = left + row.clientWidth;
    if (on.offsetLeft >= left && on.offsetLeft + on.offsetWidth <= right) return;
    row.scrollTo({
      left: clamp(on.offsetLeft - (row.clientWidth - on.offsetWidth) / 2, 0, max),
      behavior: "smooth",
    });
  }

  function nudgeCats(direction) {
    const row = $("presetCats");
    if (!row) return;
    row.scrollBy({
      left: direction * Math.max(120, row.clientWidth * 0.8),
      behavior: "smooth",
    });
  }

  function pickPreset(preset) {
    $("prompt").value = preset.prompt;
    renderValues();
    closePops();
    toast("ok", tr("toastPresetPicked"), tfx("toastPresetPickedDetail", { title: presetTitle(preset), key: runKey() }));
  }

  function randomPreset() {
    const pool = presetPool().flatMap((cat) => cat.prompts || []);
    if (!pool.length) return;
    pickPreset(pool[Math.floor(Math.random() * pool.length)]);
  }

  /* ---- Negative presets -------------------------------------------------
     A positive preset is a paragraph and replaces the field; a negative prompt
     is a keyword list, so these bundles are additive instead. Clicking one
     merges its terms into the field and clicking it again takes them back out.

     That makes the on/off state *derived* rather than stored: it is recomputed
     from the textarea on every repaint. Hand-typed terms therefore join in
     without a second source of truth to drift out of sync. */
  const negPool = () =>
    (S.bootstrap && S.bootstrap.prompts && S.bootstrap.prompts.negative) || [];
  /* Both languages ship their own term list — a short keyword list costs
     nothing to translate, unlike the prose of the positive library. */
  const negTitle = (bundle) => (S.lang === "zh" ? bundle.zh : bundle.en) || bundle.zh;
  const negTerms = (bundle) => (S.lang === "zh" ? bundle.terms_zh : bundle.terms_en) || bundle.terms_zh || [];

  /* The field is a list the user also types into, so split it the way a reader
     would and compare term by term: "blurry," and "blurry" are one request, and
     the original spelling has to survive a round trip. */
  const negSegments = (text) =>
    String(text || "")
      .split(/[,，;；\n]/)
      .map((part) => part.trim())
      .filter(Boolean);
  const negKey = (term) => term.trim().toLowerCase();

  /* Lit when every term of *either* language is already in the field, so a
     bundle does not go dark just because the interface language changed. */
  function negActive(bundle) {
    const have = new Set(negSegments($("negative").value).map(negKey));
    const covers = (terms) =>
      Array.isArray(terms) && terms.length > 0 && terms.every((t) => have.has(negKey(t)));
    return covers(bundle.terms_zh) || covers(bundle.terms_en);
  }

  function renderNegPresets() {
    const list = $("negList");
    if (!list) return;
    const pool = negPool();
    const on = pool.filter(negActive).length;
    $("negCount").textContent = tfx("negCount", { on, n: pool.length });
    list.innerHTML = "";
    pool.forEach((bundle) => {
      const lit = negActive(bundle);
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `pcard${lit ? " on" : ""}`;
      btn.setAttribute("aria-pressed", lit ? "true" : "false");
      btn.innerHTML = `<b>${esc(negTitle(bundle))}<span class="tick">${icon("i-check", 13)}</span></b><span>${esc(negTerms(bundle).join(", "))}</span>`;
      // `renderNegPresets` rebuilds this list, so without stopping the event the
      // target is detached by the time it bubbles to the outside-click rule.
      btn.onclick = (event) => { event.stopPropagation(); toggleNegPreset(bundle); };
      list.appendChild(btn);
    });
    // Clearing an empty field is a no-op, so the control says so instead.
    $("negClear").disabled = !$("negative").value.trim();
  }

  /* One toggle has to work from either language's term list: a field filled in
     Chinese then switched to English must still clear all of it. */
  function toggleNegPreset(bundle) {
    const remove = negActive(bundle);
    const drop = new Set(
      (bundle.terms_zh || []).concat(bundle.terms_en || []).map(negKey)
    );
    let parts = negSegments($("negative").value);
    if (remove) {
      parts = parts.filter((part) => !drop.has(negKey(part)));
    } else {
      const have = new Set(parts.map(negKey));
      negTerms(bundle).forEach((term) => {
        if (!have.has(negKey(term))) {
          parts.push(term);
          have.add(negKey(term));
        }
      });
    }
    $("negative").value = parts.join(", ");
    renderNegPresets();
  }

  function clearNegative() {
    $("negative").value = "";
    renderNegPresets();
  }

  /* Anchored to the right of the panel, like the positive library, and for a
     sharper reason: the trigger sits directly above the field it fills, so a
     popover dropped underneath would cover that field. Watching the field fill
     up is the entire feedback for a toggle.

     Returns false when there is nowhere to put it — the caller then leaves the
     popover shut rather than open and unanchored. */
  function placeNeg() {
    const pop = $("popNeg");
    if (pop.classList.contains("hidden")) return false;
    const appRect = $("app").getBoundingClientRect();
    const btnRect = $("btnNegPresets").getBoundingClientRect();
    const paneRect = $("composer").getBoundingClientRect();

    if (isNarrow()) {
      /* The panel is a bottom drawer here and it folds shut. A folded drawer has
         no rect worth anchoring to — the popover would land off the bottom of
         the window — so it goes away with the drawer. */
      if (paneRect.height < 120) {
        closePops();
        return false;
      }
      pop.classList.remove("side-right");
      pop.style.width = `${Math.round(paneRect.width - 20)}px`;
      pop.style.left = `${Math.round(paneRect.left - appRect.left + 10)}px`;
      pop.style.top = `${Math.round(paneRect.top - appRect.top + 14)}px`;
      pop.style.maxHeight = `${Math.round(paneRect.height - 24)}px`;
      return true;
    }

    pop.classList.add("side-right");
    const width = 348;
    let left = Math.round(paneRect.right - appRect.left) + 10;
    if (left + width > appRect.width - 12) left = Math.max(8, appRect.width - 12 - width);

    /* Measure first, then fit: the list is scrollable, so a popover that only
       reads as complete at full height should get it when the window has room
       rather than being pinned to the trigger and cut off at the bottom. */
    pop.style.maxHeight = "";
    const natural = pop.offsetHeight;
    const bottom = appRect.height - 16;
    let top = Math.round(btnRect.top - appRect.top) - 9;
    if (top + natural > bottom) top = Math.max(58, bottom - natural);
    pop.style.width = `${width}px`;
    pop.style.left = `${left}px`;
    pop.style.top = `${top}px`;
    pop.style.maxHeight = `${Math.max(220, bottom - top)}px`;
    const caret = Math.round(btnRect.top - appRect.top) + Math.round(btnRect.height / 2) - top - 4;
    pop.style.setProperty("--carety", `${clamp(caret, 16, Math.max(16, (pop.offsetHeight || 320) - 24))}px`);
    return true;
  }

  /* The dot reports health, not residency. An idle engine that has never
     loaded a pipeline is healthy — a resident one is not an error — so only a
     device warning or a failed load colours it. The reason goes on the pill
     itself; a tooltip on a 6px dot would never be found. */
  function renderEnv() {
    const device = (S.bootstrap && S.bootstrap.device) || {};
    const engine = (S.bootstrap && S.bootstrap.engine) || {};
    $("envDevice").textContent = device.device_name || "—";
    $("envMeta").textContent = [
      device.vram_gb ? `${device.vram_gb} GB ${tr("vram")}` : device.device || "",
      device.torch ? `torch ${device.torch}` : "",
    ].filter(Boolean).join(" · ");

    const warning = (device.warnings || [])[0] || "";
    const failure = engine.last_error;
    const dot = $("pillDot");
    dot.className = "dot";
    if (warning) dot.classList.add("warn");
    if (failure) dot.classList.add("err");

    const brief = (text) => (text && text.length > 140 ? `${text.slice(0, 140)}…` : text || "");
    const health = failure
      ? tfx("dotFailed", { msg: brief(failure.message) })
      : warning
        ? tfx("dotWarn", { msg: brief(warning) })
        : engine.loaded
          ? tfx("dotReady", { model: modelLabel(engine.loaded.model_key) })
          : tr("dotIdle");
    $("statusBtn").title = `${tr("envTitle")} · ${health}`;
    // The release row shows how much of the card is in use, and that number
    // lives in the bootstrap payload this runs right after.
    renderReleaseSlots();
    renderPins();
  }

  /* The switch popover lists every model that has weights on disk, whichever
     source they came from. */
  function renderModels() {
    const box = $("modelList");
    if (!box) return;
    const local = modelList().filter((m) => m.downloaded_any);
    $("modelCount").textContent = tfx("modelsAvailable", { n: local.length });
    box.innerHTML = "";
    if (!local.length) {
      box.innerHTML = `<p class="empty-row">${esc(tr("noModelLocal"))}</p>`;
      return;
    }
    local.forEach((model) => {
      const active = model.key === S.modelKey;
      const origin = model.downloaded ? tr("onDisk") : tfx("onDiskOtherHub", { hub: hubLabel(model.local_hub) });
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `model-row${active ? " on" : ""}`;
      btn.innerHTML =
        `<span class="radio"></span>` +
        `<span class="mid"><b>${esc(model.label)} · ${esc(model.precision)}</b>` +
        `<span>${esc(tfx("gb", { n: model.approx_gb }))} · ${esc(origin)}</span></span>` +
        `<span class="model-tag${active ? "" : " muted"}">${esc(active ? tr("currentModel") : tr("clickToSwitch"))}</span>`;
      btn.onclick = (event) => { event.stopPropagation(); switchModel(model); };
      box.appendChild(btn);
    });
  }

  function renderHubSegments() {
    $$("[data-hub]").forEach((btn) => {
      btn.classList.toggle("on", btn.getAttribute("data-hub") === S.hub);
    });
  }

  async function switchModel(model, hubOverride) {
    if (!model) return;
    const hub = hubOverride || S.hub;
    if (model.key === S.modelKey && hub === S.hub) return;
    const hubChanged = hub !== S.hub;
    const loaded = S.bootstrap && S.bootstrap.engine && S.bootstrap.engine.loaded;
    try {
      await putConfig(hubChanged ? { model_key: model.key, hub } : { model_key: model.key });
      if (loaded && loaded.model_key && (loaded.model_key !== model.key || hubChanged)) {
        await api("/api/models/unload", { method: "POST" }).catch(() => {});
      }
      S.modelKey = model.key;
      S.hub = hub;
      await refreshBootstrap();
      renderHubSegments();
      renderModels(); renderWeightsList(); renderPins(); renderStorage();
      toast("ok", tfx("toastModelSwitched", { label: model.label }),
        hubChanged ? tfx("toastModelSwitchedFromHub", { hub: hubLabel(hub) })
                   : tfx("toastModelSwitchedDetail", { precision: model.precision }));
    } catch (err) {
      reportError(err);
    }
  }

  /* ======================================================================
     Model & download sheet, shared by settings and first run
     ====================================================================== */
  function renderWeightsList() {
    const box = $("weightsList");
    if (!box) return;
    const all = modelList();
    const counter = $("weightsCount");
    if (counter) {
      counter.textContent = tfx("downloadedCount", {
        a: all.filter((m) => m.downloaded_any).length,
        b: all.length,
      });
    }

    box.innerHTML = "";
    all.forEach((model) => {
      const busy = S.download.key === model.key;
      const present = !!model.downloaded_any;
      // Weights may sit in the other source's cache: usable, but not in the
      // source the studio would download into.
      const otherHub = !model.downloaded && present ? model.available_hubs[0] : null;
      const note = S.lang === "zh" ? model.notes_zh : model.notes_en;
      const size = tfx("gb", { n: model.approx_gb });

      let meta;
      // A usable copy in an added folder outranks the per-source wording —
      // "on disk · from another folder" is where it actually is.
      if (present && model.external) meta = `${tr("onDiskExtraDir")} · ${size}`;
      else if (model.downloaded) meta = `${tr("onDisk")} · ${size}`;
      else if (otherHub) meta = `${tfx("onDiskOtherHub", { hub: hubLabel(otherHub) })} · ${size}`;
      else if (model.incomplete_any) meta = `${tfx("weightsIncompleteShort", { n: model.missing_count })} · ${size}`;
      else meta = `${note || tr("notDownloaded")} · ${size}`;

      // A percentage needs a known total. When the source cannot report one
      // (e.g. a remote listing failed), show the bytes received instead of a
      // frozen 0% — a stalled number reads as "broken", a moving one does not.
      const received = S.download.key === model.key ? S.download.bytes : 0;
      const readout = S.download.total
        ? `${S.download.pct}%`
        : received
          ? tfx("downloadedSoFar", { size: fmtBytes(received) })
          : "…";
      const barPct = S.download.total ? S.download.pct : 0;

      // The caveat line only replaces the state text when the weights are
      // absent; a downloaded model keeps the note on hover.
      const row = document.createElement("div");
      row.className = `model-row${model.key === S.modelKey ? " on" : ""}${present ? "" : " off"}`;
      row.innerHTML =
        `<span class="radio"></span>` +
        `<span class="mid"${note ? ` title="${esc(note)}"` : ""}><b>${esc(model.label)} · ${esc(model.precision)}</b>` +
        `<span>${esc(meta)}</span></span>` +
        (busy
          ? `<span class="mini plain">${esc(readout)}</span>`
          : `<button type="button" class="mini" data-dl>${icon("i-download", 12)}${esc(model.downloaded ? tr("redownload") : model.incomplete_any ? tr("resumeDownload") : tr("downloadNow"))}</button>`) +
        (busy ? `<span class="prog${S.download.total ? "" : " pulse"}"><i style="width:${barPct}%"></i></span>` : "");
      row.onclick = (event) => {
        if (event.target.closest("[data-dl]")) {
          event.stopPropagation();
          startDownload(model, !!model.downloaded);
          return;
        }
        if (!present) {
          if (model.incomplete_any) {
            toast("err", tr("weightsIncomplete"),
              tfx("weightsIncompleteDetail", { n: model.missing_count }));
          } else {
            toast("err", tr("needDownloadFirst"), tfx("needDownloadFirstDetail", { hub: hubLabel(S.hub) }));
          }
          return;
        }
        switchModel(model, otherHub);
      };
      box.appendChild(row);
    });
  }

  /* Where the selected source keeps its weights, and what decided that path.
     Reads from the backend so HF_HOME / HF_HUB_CACHE / MODELSCOPE_CACHE are
     reflected instead of a hard-coded guess. Both the settings sheet and the
     first-run sheet carry a box; `data-storage="compact"` (first run) shows
     only the weights line. */
  function renderStorage() {
    const storage = (S.bootstrap && S.bootstrap.storage) || {};
    const info = (storage.hub_dirs || {})[S.hub] || {};
    const origin = info.env_var ? tfx("dirFromEnv", { name: info.env_var }) : tr("dirDefault");
    const model = modelByKey(S.modelKey);
    // The snapshot only belongs to this row when it lives in the selected source.
    const snapshot = model && model.local_hub === S.hub && model.path ? model.path : "";
    const missing = info.path && !info.exists
      ? ` <em>${esc(tr("dirMissing"))}</em>`
      : "";
    const weightsRow =
      `<div class="full"><span class="k">${esc(tr("weightsDir"))}` +
      `<em>${esc(hubLabel(S.hub))} · ${esc(origin)}</em></span>` +
      `<span class="v"${snapshot ? ` title="${esc(snapshot)}"` : ""}>${esc(info.path || tr("unknown"))}${missing}</span></div>`;
    const outputsRow =
      `<div class="full"><span class="k">${esc(tr("outputsDir"))}</span>` +
      `<span class="v">${esc(storage.outputs || tr("unknown"))}</span></div>`;
    $$("[data-storage]").forEach((box) => {
      box.innerHTML =
        `<div class="kv">${weightsRow}${box.getAttribute("data-storage") === "compact" ? "" : outputsRow}</div>`;
    });
  }

  /* Extra weight folders: places that already hold weights, downloaded by
     hand or by another tool. They are only ever searched — a download always
     lands in the hub's own cache — so a pre-existing copy shows up as ready
     without a second download. */
  function extraDirs() {
    return (S.bootstrap && S.bootstrap.config && S.bootstrap.config.extra_weight_dirs) || [];
  }

  const DIR_REASON_KEYS = {
    empty: "dirReasonEmpty",
    invalid: "dirReasonInvalid",
    missing: "dirReasonMissing",
    not_a_dir: "dirReasonNotADir",
  };

  function dirReasonText(reason) {
    return tr(DIR_REASON_KEYS[reason] || "dirReasonInvalid");
  }

  const extraDirsPanelHtml = () => (
    `<div class="grp">${esc(tr("extraDirsTitle"))}</div>` +
    `<div class="dir-list" data-extra-dirs></div>`
  );

  function renderExtraDirs() {
    const dirs = (S.bootstrap && S.bootstrap.storage && S.bootstrap.storage.extra_dirs) || [];
    const joiner = S.lang === "zh" ? "、" : ", ";
    const rows = dirs.map((dir) => {
      const note = !dir.ok
        ? dirReasonText(dir.reason)
        : dir.models && dir.models.length
          ? tfx("dirHoldsModels", { models: dir.models.join(joiner) })
          : tr("dirNoModels");
      return (
        `<div class="dir-row">` +
        `<span class="d-ico">${icon("i-folder", 14)}</span>` +
        `<span class="d-path" title="${esc(dir.path)}">${esc(dir.path)}</span>` +
        `<span class="d-note${dir.ok ? "" : " bad"}">${esc(note)}</span>` +
        `<button type="button" class="d-x" data-rmdir="${esc(dir.path)}" title="${esc(tr("removeDir"))}" aria-label="${esc(tr("removeDir"))}">${icon("i-x", 12)}</button>` +
        `</div>`
      );
    }).join("");
    $$("[data-extra-dirs]").forEach((box) => {
      box.innerHTML =
        (rows || `<p class="sub" style="margin:2px 0 0">${esc(tr("extraDirsEmpty"))}</p>`) +
        `<div class="dir-add">` +
        `<input type="text" data-dir-input placeholder="${esc(tr("addDirPlaceholder"))}" spellcheck="false" autocomplete="off" />` +
        `<button type="button" class="btn sm" data-pickdir title="${esc(tr("browse"))}" aria-label="${esc(tr("browse"))}">${icon("i-folder", 13)}</button>` +
        `<button type="button" class="btn sm on" data-adddir>${esc(tr("addDir"))}</button>` +
        `</div>`;
      bindExtraDirs(box);
    });
  }

  async function addWeightDir(raw) {
    const text = String(raw || "").trim();
    if (!text) return;
    // Validate before saving so a bad path is rejected now, not discovered
    // later as a silently dead search path.
    let check = null;
    try {
      check = await api("/api/weights/check-dir", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: text }),
      });
    } catch (err) {
      reportError(err);
      return;
    }
    const path = (check && check.path) || text;
    if (!check || !check.ok) {
      toast("err", tr("toastDirRejected"), dirReasonText(check && check.reason));
      return;
    }
    const list = extraDirs().filter((item) => item !== path);
    list.push(path);
    try {
      await putConfig({ extra_weight_dirs: list });
      await refreshBootstrap();
      renderWeightsList(); renderStorage(); renderExtraDirs();
      const found = check.models && check.models.length
        ? tfx("dirHoldsModels", { models: check.models.join(S.lang === "zh" ? "、" : ", ") })
        : tr("dirNoModels");
      toast("ok", tr("toastDirAdded"), found);
    } catch (err) {
      reportError(err);
    }
  }

  async function removeWeightDir(path) {
    const list = extraDirs().filter((item) => item !== path);
    try {
      await putConfig({ extra_weight_dirs: list });
      await refreshBootstrap();
      renderWeightsList(); renderStorage(); renderExtraDirs();
      toast("ok", tr("toastDirRemoved"), path);
    } catch (err) {
      reportError(err);
    }
  }

  function bindExtraDirs(box) {
    const input = box.querySelector("[data-dir-input]");
    const addBtn = box.querySelector("[data-adddir]");
    const pickBtn = box.querySelector("[data-pickdir]");
    if (addBtn) addBtn.onclick = () => addWeightDir(input && input.value);
    if (input) {
      input.onkeydown = (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          addWeightDir(input.value);
        }
      };
    }
    if (pickBtn) {
      pickBtn.onclick = async () => {
        pickBtn.disabled = true;
        try {
          const res = await api("/api/weights/pick-dir", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
          });
          if (res && res.ok && res.path) {
            await addWeightDir(res.path);
          } else if (res && res.reason === "unavailable") {
            toast("err", tr("pickDirTitle"), tr("pickDirUnavailable"));
          }
          // A cancelled dialog is not news — stay quiet.
        } catch (err) {
          reportError(err);
        } finally {
          pickBtn.disabled = false;
        }
      };
    }
    $$("[data-rmdir]", box).forEach((btn) => {
      btn.onclick = (event) => {
        event.stopPropagation();
        removeWeightDir(btn.getAttribute("data-rmdir"));
      };
    });
  }

  function hubSegmentHtml() {
    const hubs = (S.bootstrap && S.bootstrap.hubs) || {};
    return Object.values(hubs).map((hub) =>
      `<button type="button" data-hub="${esc(hub.key)}" class="${hub.key === S.hub ? "on" : ""}">${esc(hubLabel(hub.key))}</button>`
    ).join("");
  }

  function bindHubSegment(root) {
    $$("[data-hub]", root).forEach((btn) => {
      btn.onclick = async () => {
        const hub = btn.getAttribute("data-hub");
        if (hub === S.hub) return;
        S.hub = hub;
        renderHubSegments();
        try {
          await putConfig({ hub });
          await refreshBootstrap();
          renderPins(); renderWeightsList(); renderStorage();
          toast("ok", tr("toastHubSwitched"), hubLabel(hub));
        } catch (err) {
          reportError(err);
        }
      };
    });
  }

  /* Tokens are persisted together with the download action — no separate save. */
  async function persistCredentials() {
    const patch = {};
    if ($("sHf") && $("sHf").value) patch.hf_token = $("sHf").value;
    if ($("sMs") && $("sMs").value) patch.ms_token = $("sMs").value;
    if (!Object.keys(patch).length) return;
    try {
      await putConfig(patch);
      $("sHf").value = "";
      $("sMs").value = "";
    } catch (err) {
      reportError(err);
    }
  }

  async function startDownload(model, force) {
    if (S.download.key) return;
    await persistCredentials();
    S.download = { key: model.key, pct: 0, bytes: 0, total: null };
    renderWeightsList();
    toast("ok", tfx("toastDownloadStart", { label: model.label }),
      tfx("toastDownloadStartDetail", { hub: hubLabel(S.hub), gb: model.approx_gb }));
    try {
      await api("/api/models/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_key: model.key, hub: S.hub, force: !!force }),
      });
    } catch (err) {
      S.download = { key: null, pct: 0, bytes: 0, total: null };
      renderWeightsList();
      reportError(err);
    }
  }

  const weightsPanelHtml = () => `<div class="models" id="weightsList"></div>`;

  function downloadFieldsHtml() {
    return (
      `<div class="grp">${esc(tr("tokenSection"))}<span class="sec-val" style="margin-left:auto">${esc(tr("tokenOptional"))}</span></div>` +
      `<div class="field"><label class="flabel" for="sHf">${esc(tr("hfToken"))}<span class="flabel-note">${esc(tr("tokenLeaveEmpty"))}</span></label>` +
      `<div class="token-field"><input id="sHf" type="password" placeholder="hf_…" autocomplete="off" />` +
      `<button type="button" data-reveal="sHf" aria-label="${esc(tr("hfToken"))}">${icon("i-eye", 15)}</button></div></div>` +
      `<div class="field"><label class="flabel" for="sMs">${esc(tr("msToken"))}<span class="flabel-note">${esc(tr("tokenLeaveEmpty"))}</span></label>` +
      `<div class="token-field"><input id="sMs" type="password" placeholder="—" autocomplete="off" />` +
      `<button type="button" data-reveal="sMs" aria-label="${esc(tr("msToken"))}">${icon("i-eye", 15)}</button></div></div>` +
      `<p class="sub">${esc(tr("tokenAutoSave"))}</p>`
    );
  }

  function bindReveal(root) {
    $$("[data-reveal]", root).forEach((btn) => {
      btn.onclick = () => {
        const input = $(btn.getAttribute("data-reveal"));
        const shown = input.type === "text";
        input.type = shown ? "password" : "text";
        btn.innerHTML = icon(shown ? "i-eye" : "i-eye-off", 15);
      };
    });
  }

  function openSettings() {
    closePops();
    $("settingsBody").innerHTML =
      `<div class="grp">${esc(tr("downloadSource"))}</div>` +
      `<div class="seg" id="settingsHubs">${hubSegmentHtml()}</div>` +
      `<p class="sub">${esc(tr("sourceHint"))}</p>` +
      `<div class="grp">${esc(tr("weights"))}<span class="sec-val" id="weightsCount" style="margin-left:auto">—</span></div>` +
      weightsPanelHtml() +
      extraDirsPanelHtml() +
      downloadFieldsHtml() +
      `<div class="grp">${esc(tr("storage"))}</div>` +
      `<div id="storageBlock" data-storage="full"></div>` +
      `<p class="sub">${esc(tr("licenseNote"))}</p>`;
    $("settingsFoot").innerHTML =
      `<button type="button" class="btn" id="openFolder">${icon("i-folder")}<span class="lbl">${esc(tr("openFolder"))}</span></button>` +
      `<span class="grow"></span>` +
      `<button type="button" class="btn on" id="settingsDone">${esc(tr("doneBtn"))}</button>`;
    bindHubSegment($("settingsBody"));
    bindReveal($("settingsBody"));
    renderWeightsList();
    renderStorage();
    renderExtraDirs();
    $("openFolder").onclick = revealOutput;
    $("settingsDone").onclick = closeSettings;
    $("settings").classList.remove("hidden");
    syncOverlayState();
  }

  function closeSettings() {
    $("settings").classList.add("hidden");
    $$("[data-reveal]", $("settings")).forEach((btn) => {
      const input = $(btn.getAttribute("data-reveal"));
      if (input) input.type = "password";
    });
    syncOverlayState();
  }

  function openFirstRun() {
    $("firstRunBody").innerHTML =
      `<p class="sub" style="margin-top:0">${esc(tr("firstRunWelcome"))}</p>` +
      `<div class="grp">${esc(tr("language"))}</div>` +
      `<div class="seg" id="firstRunLang">` +
      `<button type="button" data-lang="zh" class="${S.lang === "zh" ? "on" : ""}">中文</button>` +
      `<button type="button" data-lang="en" class="${S.lang === "en" ? "on" : ""}">English</button>` +
      `</div>` +
      `<div class="grp">${esc(tr("downloadSource"))}</div>` +
      `<div class="seg" id="firstRunHubs">${hubSegmentHtml()}</div>` +
      `<div data-storage="compact"></div>` +
      `<div class="grp">${esc(tr("weights"))}<span class="sec-val" id="weightsCount" style="margin-left:auto">—</span></div>` +
      weightsPanelHtml() +
      extraDirsPanelHtml() +
      downloadFieldsHtml();
    $("firstRunFoot").innerHTML =
      `<span class="grow"></span><button type="button" class="btn on" id="firstRunDone">${esc(tr("enterStudio"))}</button>`;
    bindHubSegment($("firstRunBody"));
    bindReveal($("firstRunBody"));
    $$("[data-lang]", $("firstRunBody")).forEach((btn) => {
      btn.onclick = async () => {
        S.lang = btn.getAttribute("data-lang");
        $$("[data-lang]", $("firstRunBody")).forEach((b) => b.classList.toggle("on", b === btn));
        applyI18n();
        openFirstRun();
        await putConfig({ language: S.lang }).catch(() => {});
      };
    });
    renderWeightsList();
    renderStorage();
    renderExtraDirs();
    $("firstRunDone").onclick = async () => {
      await persistCredentials();
      try {
        await putConfig({ setup_completed: true, language: S.lang, hub: S.hub, model_key: S.modelKey });
      } catch (_) { /* entering the studio should not be blocked by this */ }
      $("firstRun").classList.add("hidden");
      syncOverlayState();
    };
    $("firstRun").classList.remove("hidden");
    syncOverlayState();
  }

  function syncOverlayState() {
    const open = !$("detail").classList.contains("hidden")
      || !$("settings").classList.contains("hidden")
      || !$("firstRun").classList.contains("hidden");
    $("app").classList.toggle("overlay-open", open);
  }

  async function revealOutput() {
    try {
      await api("/api/reveal", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) });
      toast("ok", tr("toastOpened"), tr("toastOpenedDetail"));
    } catch (err) {
      reportError(err);
    }
  }

  /* ======================================================================
     Run
     ====================================================================== */
  function setBusy(busy) {
    S.busy = busy;
    renderRun();
    renderTools();
    // The progress block offers "release VRAM" only while there is something to
    // stop, so it follows the busy flag like the rest of that block.
    renderReleaseSlots();
    // Every exit path (complete, cancel, error) funnels through here; the
    // stage state must never outlive the job that started it.
    if (busy) {
      S.lastEventAt = Date.now();
      S.offline = false;
      startWatchdog();
    } else {
      stopPhaseUi();
      stopWatchdog();
      stopJobPoll();
      setTimeout(() => $("progress").classList.add("hidden"), 800);
    }
  }

  /* The engine names the stage it actually entered. Conditioning, the VAE
     encode and the VAE decode carry no per-step signal and can each take
     minutes, so those get a sweeping bar and a live timer instead of a bar
     that sits at 2% with a label that means nothing. */
  const PHASE_LABEL = {
    condition: "phaseConditionRun",
    encode: "phaseEncodeRun",
    decode: "phaseDecodingRun",
  };
  const PHASE_SLOW_NOTE = {
    condition: "conditioningSlowNote",
    encode: "encodingSlowNote",
    decode: "decodingSlowNote",
  };

  function startPhaseUi(phase, slow, elapsedMs) {
    const track = $("progressTrack");
    if (!track) return;
    if (S.phaseName === phase && S.phaseTimer) {
      // A poll repeats the same stage every 5s: re-anchor the clock from the
      // server's own elapsed time instead of restarting the timer each round.
      S.phaseSlow = !!slow;
      S.phaseAt = Date.now() - (elapsedMs || 0);
      renderPhase();
      renderReleaseSlots();
      return;
    }
    stopPhaseUi();
    S.phaseName = phase;
    S.phaseSlow = !!slow;
    S.phaseAt = Date.now() - (elapsedMs || 0);
    $("progress").classList.remove("hidden");
    $("progressPhase").textContent = tr(PHASE_LABEL[phase] || "phaseSampling");
    track.classList.add("pulse");
    S.phaseTimer = setInterval(renderPhase, 1000);
    renderPhase();
    // A stage the engine already calls slow is the other half of "you may want
    // to stop waiting", so the release row follows this flag.
    renderReleaseSlots();
  }

  function renderPhase() {
    if (!S.phaseName) return;
    $("progressPhase").textContent = tr(PHASE_LABEL[S.phaseName] || "phaseSampling");
    const seconds = ((Date.now() - S.phaseAt) / 1000).toFixed(0);
    $("progressText").textContent = [
      tfx("metaElapsed", { s: seconds }),
      S.phaseSlow ? tr(PHASE_SLOW_NOTE[S.phaseName] || "decodingSlowNote") : "",
    ].filter(Boolean).join(" · ");
  }

  function stopPhaseUi() {
    if (S.phaseTimer) {
      clearInterval(S.phaseTimer);
      S.phaseTimer = 0;
    }
    S.phaseName = null;
    S.phaseSlow = false;
    const track = $("progressTrack");
    if (track) track.classList.remove("pulse");
    // The release row is offered while a slow stage is running, so it goes away
    // with the stage rather than with the run.
    renderReleaseSlots();
  }

  /* Nothing inside a long encode checks a cancel flag, so a run that stops
     reporting looks exactly like one that is working. The stall notice is the
     difference: after a minute of silence it says so, and points at the two
     things it can be. */
  const STALL_MS = 60000;

  function noteServerEvent() {
    S.lastEventAt = Date.now();
    const hint = $("progressHint");
    if (hint) hint.classList.add("hidden");
  }

  function startWatchdog() {
    stopWatchdog();
    S.watchdog = setInterval(() => {
      const hint = $("progressHint");
      if (!hint) return;
      if (!S.busy) return;
      const since = S.lastEventAt || S.startedAt || Date.now();
      const idle = Date.now() - since;
      const stalled = idle >= STALL_MS;
      if (!stalled) {
        hint.classList.add("hidden");
      } else {
        hint.textContent = tfx(S.offline ? "stallHintOffline" : "stallHint", {
          s: Math.round(idle / 1000),
        });
        hint.classList.remove("hidden");
      }
      // The release action shows up with the stall notice, so the two arrive
      // together: the moment the studio stops claiming progress is the moment
      // offering a way out is worth the row it takes.
      if (stalled !== S.stalled) {
        S.stalled = stalled;
        renderReleaseSlots();
      }
    }, 1000);
  }

  function stopWatchdog() {
    if (S.watchdog) {
      clearInterval(S.watchdog);
      S.watchdog = 0;
    }
    S.stalled = false;
    const hint = $("progressHint");
    if (hint) hint.classList.add("hidden");
  }

  /* ======================================================================
     Releasing VRAM
     Stopping a run and giving the card's memory back are two different things,
     and the studio used to announce the second whenever it did the first. A
     cancel keeps the weights resident on purpose — that is what makes the next
     run start in a second — and it frees nothing at all. This is the action
     that really frees, and because it costs a reload afterwards it asks first.

     One implementation, two hosts: the progress block, where someone looks when
     a run has gone quiet, and the environment popover, for when nothing is
     running and the card itself is the thing to hand back.
     ====================================================================== */
  const RELEASE_HOSTS = ["releaseRow", "releaseEnvRow"];
  // A stage does not check the stop flag while it runs — one CUDA op cannot be
  // interrupted from Python — so the request is repeated until the run returns
  // from whatever it is inside. A round costs two waits, not one: the server
  // holds each request for RELEASE_WAIT_S (15s) before answering `stopping`, and
  // the gap below runs after that. So 120 rounds is about forty minutes, not the
  // ten the old comment claimed — enough for the worst measured stage (a 2K
  // conditioning pass, 737s) without leaving a spinner forever.
  const RELEASE_POLL_MS = 5000;
  const RELEASE_TRIES = 120;

  function vramReadout() {
    const engine = (S.bootstrap && S.bootstrap.engine) || {};
    const vram = engine.vram;
    if (vram && vram.used_gb != null) {
      return tfx("releaseUsed", { used: vram.used_gb, total: vram.total_gb });
    }
    // The reading failed. Dropping the note entirely is what made a device that
    // had run out of room look like one with nothing to report — and the
    // reading is exactly what fails when the card fills up.
    if (engine.vram_error) return tr("vramUnavailable");
    return "";
  }

  function releaseStoppingText() {
    const key = PHASE_LABEL[S.releasePhase];
    return key ? tfx("releaseStoppingPhase", { phase: tr(key) }) : tr("releaseStopping");
  }

  /* The progress block only offers it once waiting has stopped being the plan:
     the run has gone quiet, or the engine already said this stage takes
     minutes. The environment popover offers it always, because freeing the card
     is not only something you do about a run. */
  function wantsRelease() {
    return S.busy && (S.stalled || S.phaseSlow);
  }

  function renderReleaseSlots() {
    RELEASE_HOSTS.forEach((id) => {
      const host = $(id);
      if (!host) return;
      // The confirmation owns this host until it is answered. Repainting would
      // throw its buttons away, so its text follows the language through
      // `data-i` instead.
      if (S.releaseStage === "ask" && S.releaseHost === id) return;
      // A wait in flight is shown wherever it was started, even if the row
      // would otherwise have gone away underneath it.
      if (id === "releaseRow" && S.releaseStage !== "stopping") {
        host.classList.toggle("hidden", !wantsRelease());
      }
      if (S.releaseStage === "stopping") {
        // The note and the button travel together. A stage can hold the request
        // for as long as it runs, and a row showing nothing but "stop requested"
        // reads as a studio that has stopped listening — there is no way back to
        // the action and no way to tell it apart from a wedged tab. Disabled
        // while the poll is in flight, live again once it gives up.
        host.innerHTML =
          `<button type="button" class="btn sm"${S.releasing ? " disabled" : ""}>` +
            `${esc(tr("releaseVram"))}</button>` +
          `<span class="relnote">${esc(releaseStoppingText())}</span>`;
        host.querySelector("button").onclick = () => askRelease(id);
        return;
      }
      const used = vramReadout();
      host.innerHTML =
        `<button type="button" class="btn sm"${S.releasing ? " disabled" : ""}>` +
          `${esc(tr("releaseVram"))}</button>` +
        (used ? `<span class="relnote">${esc(used)}</span>` : "");
      host.querySelector("button").onclick = () => askRelease(id);
    });
  }

  async function releaseVram() {
    if (S.releasing) return;
    S.releasing = true;
    S.releaseStage = "stopping";
    S.releasePhase = (((S.bootstrap || {}).engine || {}).phase || {}).name || null;
    renderReleaseSlots();
    try {
      for (let attempt = 0; attempt < RELEASE_TRIES; attempt += 1) {
        let out;
        try {
          out = await api("/api/engine/release", { method: "POST" });
        } catch (err) {
          S.releaseStage = null;
          reportError(err);
          return;
        }
        if (out.released) {
          S.releaseStage = null;
          // `released` means the engine answered from an idle state, so whatever
          // this tab was showing as running is over.
          S.jobId = null;
          setBusy(false);
          await refreshBootstrap().catch(() => {});
          renderEnv();
          renderModels();
          // Nothing resident, or the card was already empty: a measured zero is
          // a number worth dropping rather than printing.
          toast("ok",
            out.freed_gb ? tfx("toastReleased", { n: out.freed_gb }) : tr("toastReleasedNone"),
            tr("toastReleasedDetail"));
          return;
        }
        // Still inside a stage. Keep the note up, say which one, and ask again.
        S.releasePhase = out.phase || S.releasePhase;
        renderReleaseSlots();
        await new Promise((resolve) => setTimeout(resolve, RELEASE_POLL_MS));
      }
      // Ten minutes inside one stage is not a working run. The note stays up so
      // the state is visible, and the reader can ask again.
      toast("err", tr("toastReleasePending"), tr("toastReleasePendingDetail"));
    } finally {
      S.releasing = false;
      // The wait is over, whichever way it ended. Keeping `stopping` here is
      // what turned "you can also ask again" into a dead end: the row kept the
      // note and never rebuilt a button, in either host, so the only way out was
      // a reload. Clearing it puts the action back.
      S.releaseStage = null;
      renderReleaseSlots();
    }
  }

  /* One confirmation, two hosts, so the wording and the consequence cannot drift
     apart between them. */
  function askRelease(hostId) {
    S.releaseStage = "ask";
    S.releaseHost = hostId;
    const host = $(hostId);
    host.innerHTML =
      `<div class="confirm">` +
        `<div class="confirm-head">${icon("i-alert", 16)}` +
          `<span><span data-i="confirmReleaseAsk"></span><b data-i="confirmReleaseWhy"></b></span>` +
        `</div>` +
        `<div class="confirm-btns">` +
          `<button type="button" class="btn r-no" data-i="confirmReleaseNo"></button>` +
          `<button type="button" class="btn btn-danger-solid r-yes" data-i="confirmReleaseYes"></button>` +
        `</div>` +
      `</div>`;
    paintI18n(host);
    host.querySelector(".r-no").onclick = () => closeReleaseAsk();
    host.querySelector(".r-yes").onclick = () => {
      closeReleaseAsk();
      releaseVram().catch(reportError);
    };
  }

  function closeReleaseAsk() {
    S.releaseStage = null;
    S.releaseHost = null;
    renderReleaseSlots();
  }

  /* Adopt the stage the server reports. A reload or a second tab has no socket
     history, and the stage is the only thing that separates a slow encode from
     a run that is over. */
  function adoptPhase(live) {
    if (!live || !live.name) return;
    if (live.name === "sample") {
      // The sampler has its own per-step readout; do not cover it with a timer.
      stopPhaseUi();
      $("progressPhase").textContent = tr("phaseSampling");
      return;
    }
    startPhaseUi(live.name, !!live.slow, live.elapsed_ms || 0);
  }

  /* ======================================================================
     Job lifecycle
     A job outlives the socket that announces it: tabs reconnect, the server
     restarts, the machine sleeps, a long decode outlasts a dropped frame.
     So the terminal states are reached from two directions — the socket event
     and the poll/reconnect fallback — and both land on the same functions.
     ====================================================================== */
  async function applyJobComplete(jobId, imageUrl) {
    // Only the job this tab is tracking may finish it; a late duplicate (poll
    // and event racing) or a stale event from an older run is ignored.
    if (S.jobId !== jobId) return;
    const started = S.startedAt;
    setBusy(false);
    S.jobId = null;
    $("progressBar").style.width = "100%";
    let item = null;
    try { item = await api(`/api/jobs/${jobId}`); } catch (_) { /* fall back to local state */ }
    const w = (item && item.width) || S.width;
    const h = (item && item.height) || S.height;
    // A multi-image run made several pictures; the record lists them all. Fall
    // back to the single URL from the event when the row cannot be read.
    const stamp = Date.now();
    const urls = (item && item.image_urls) || (imageUrl ? [imageUrl] : [`/api/outputs/${jobId}`]);
    const thumbs = (item && item.thumb_urls) || [];
    const frames = urls.map((url, index) => ({
      url: `${url}?t=${stamp}`,
      thumb: thumbs[index] ? `${thumbs[index]}?t=${stamp}` : "",
      w, h,
    }));
    showResult(frames, w, h);
    const seconds = item && item.duration_ms ? (item.duration_ms / 1000).toFixed(1) : ((Date.now() - started) / 1000).toFixed(1);
    const count = frames.length;
    toast("ok", count > 1 ? tfx("toastDoneMulti", { n: count }) : tr("toastDone"),
      tfx("toastDoneDetail", { w, h, s: seconds }));
    await loadHistory().catch(() => {});
  }

  function failJobUi(message) {
    setBusy(false);
    S.jobId = null;
    S.download = { key: null, pct: 0, bytes: 0, total: null };
    renderWeightsList();
    toast("err", tr("errFailed"), message || "");
    refreshBootstrap().then(renderEnv).catch(() => {});
  }

  function cancelJobUi() {
    setBusy(false);
    S.jobId = null;
    $("progress").classList.add("hidden");
    // A release in flight is about to say something more useful than "stopped",
    // and it is the same event: one toast, from the release, not two.
    if (!S.releasing) toast("ok", tr("toastCancelled"), tr("toastCancelledDetail"));
  }

  /* Ask the server what happened to the job this tab is tracking. Never infer
     "still running" from a missing event — that is how a finished picture used
     to stay invisible behind a bar that swept forever. */
  async function resyncJob() {
    const jobId = S.jobId;
    if (!jobId) return;
    let item = null;
    try {
      item = await api(`/api/jobs/${jobId}`);
      S.offline = false;
    } catch (_) {
      // The server itself may be gone. Remember that, so the stall notice says
      // "nothing is answering" instead of implying the render is still busy.
      S.offline = true;
      return;
    }
    if (!item || S.jobId !== jobId) return;
    if (item.status === "succeeded") await applyJobComplete(jobId, item.image_url);
    else if (item.status === "failed") failJobUi(item.error || "");
    else if (item.status === "cancelled") cancelJobUi();
    else if (item.status === "running") adoptPhase(item.live_phase);
  }

  function startJobPoll() {
    stopJobPoll();
    S.pollTimer = setInterval(() => { resyncJob().catch(() => {}); }, 5000);
  }

  function stopJobPoll() {
    if (S.pollTimer) {
      clearInterval(S.pollTimer);
      S.pollTimer = 0;
    }
  }

  /* A reload — or a second tab — has no socket history. If the engine is still
     working on the newest row, re-attach instead of showing an idle studio. */
  async function adoptRunningJob() {
    if (S.busy) return;
    let data = null;
    try { data = await api("/api/jobs?limit=5"); } catch (_) { return; }
    const running = (data.items || []).find((item) => item.status === "running");
    if (!running) return;
    S.jobId = running.id;
    S.startedAt = Date.parse(running.created_at) || Date.now();
    setBusy(true);
    $("progress").classList.remove("hidden");
    $("progressBar").style.width = "2%";
    adoptPhase(running.live_phase);
    if (!S.phaseName) $("progressPhase").textContent = tr("phaseSampling");
    startJobPoll();
  }

  function buildForm() {
    const fd = new FormData();
    fd.set("mode", S.mode);
    fd.set("prompt", $("prompt").value.trim());
    fd.set("negative_prompt", $("negative").value);
    fd.set("model_key", S.modelKey);
    fd.set("hub", S.hub);
    fd.set("scale", S.scale);
    fd.set("aspect", S.aspect);
    fd.set("width", $("width").value || "0");
    fd.set("height", $("height").value || "0");
    fd.set("output_resolution", $("outRes").value || "0");
    fd.set("steps", $("steps").value);
    fd.set("true_cfg_scale", $("cfg").value);
    fd.set("seed", $("seed").value || "-1");
    fd.set("use_kv_cache", $("kv").getAttribute("aria-checked") === "true" ? "true" : "false");
    fd.set("num_images", $("nImages").value || "1");
    fd.set("transparent", $("rgba").getAttribute("aria-checked") === "true" ? "true" : "false");
    fd.set("follow_ref_aspect", $("follow").getAttribute("aria-checked") === "true" ? "true" : "false");
    fd.set("vae_tiling", $("vaeTiling").value);
    // References belong to 改图. The section is hidden in 生图 but the files stay
    // in `S.refs` (switching modes must not throw away a reference the user just
    // picked, and `sendToEdit` sets the mode *before* it adds one), so without
    // this guard a generate run would post them anyway: the engine ignores them
    // (`images=refs if mode == "edit" else None`) but the server saved them and
    // the detail overlay listed a 参考图 the run never used.
    if (S.mode === "edit") {
      S.refs.forEach((ref) => fd.append("files", ref.file, ref.file.name));
    }
    return fd;
  }

  async function run() {
    if (S.busy) {
      if (S.jobId) api(`/api/jobs/${S.jobId}/cancel`, { method: "POST" }).catch(() => {});
      return;
    }
    const prompt = $("prompt").value.trim();
    if (!prompt) {
      $("prompt").focus();
      toast("err", tr("toastNoPrompt"), tr("toastNoPromptDetail"));
      return;
    }
    if (S.mode === "edit" && !S.refs.length) {
      toast("err", tr("toastNeedRefs"), tr("toastNeedRefsDetail"));
      return;
    }
    setBusy(true);
    $("progress").classList.remove("hidden");
    $("progressPhase").textContent = tr("phaseQueued");
    $("progressText").textContent = "";
    $("progressBar").style.width = "2%";
    S.startedAt = Date.now();
    // A fresh run must start from a clean slate even if a previous stage UI
    // state somehow survived.
    stopPhaseUi();
    toast("ok", tr("toastQueued"), tfx("toastQueuedDetail", {
      model: modelLabel(S.modelKey), w: S.width, h: S.height, steps: S.steps,
    }));
    try {
      const job = await api("/api/jobs", { method: "POST", body: buildForm() });
      S.jobId = job.id;
      // Belt and braces beside the socket: the poll notices a finished job
      // even if every event was lost on the way.
      startJobPoll();
    } catch (err) {
      setBusy(false);
      $("progress").classList.add("hidden");
      reportError(err);
    }
  }

  /* ======================================================================
     WebSocket
     ====================================================================== */
  function connectWs() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onmessage = (event) => {
      let msg;
      try { msg = JSON.parse(event.data); } catch (_) { return; }
      handleEvent(msg).catch(() => {});
    };
    ws.onopen = () => {
      // A reconnect has missed everything that happened while it was down.
      resyncJob().catch(() => {});
    };
    ws.onclose = () => setTimeout(connectWs, 1500);

    async function handleEvent(msg) {
      if (msg.type === "job_queued" && S.busy) {
        noteServerEvent();
        S.startedAt = S.startedAt || Date.now();
        $("progressPhase").textContent = tr("phaseQueued");
      }
      if (msg.type === "load_start" || msg.type === "load_stage") {
        noteServerEvent();
        $("progressPhase").textContent = tr("phaseLoading");
      }
      if (msg.type === "load_complete") {
        // Health changed: the engine now holds a pipeline, or recovered from a
        // previous failure. Re-read the status instead of guessing.
        await refreshBootstrap().then(renderEnv).catch(() => {});
      }
      if (msg.type === "notice" && msg.code === "int8_streamed") {
        // The weights were moved off the card because this run would not have
        // fitted with them resident. Without saying so the only visible
        // difference is a slower run and no reason for it.
        noteServerEvent();
        toast(
          "ok",
          tr("noticeLowVramTitle"),
          tfx("noticeLowVram", { free: msg.free_gb, resident: msg.resident_gb })
        );
      }
      if (msg.type === "generate_start") {
        // The pipeline call is about to begin; the engine announces the real
        // stages (conditioning, encode, decode) from here on.
        noteServerEvent();
        $("progressPhase").textContent = tr("phasePreparing");
      }
      if (msg.type === "generate_phase") {
        // A real stage, announced by the engine the moment it is entered.
        // These carry no per-step signal, so switch to a sweeping bar and a
        // live timer — a stale 2% bar with a stale label is what a hang looks
        // like from the outside.
        noteServerEvent();
        startPhaseUi(msg.phase, !!msg.slow);
      }
      if (msg.type === "generate_progress") {
        noteServerEvent();
        // Sampler steps only, and they always read as sampling: the old
        // "step <= 2 → loading model" guess named a stage the run had already
        // left, which is exactly what made an edit run look stuck.
        stopPhaseUi();
        const total = msg.total || S.steps;
        const step = msg.step || 0;
        const pct = total ? step / total : 0;
        const elapsed = S.startedAt ? (Date.now() - S.startedAt) / 1000 : 0;
        $("progressPhase").textContent = tr("phaseSampling");
        $("progressBar").style.width = `${Math.round(pct * 100)}%`;
        $("progressText").textContent = [
          tfx("metaStep", { s: step, t: total }),
          elapsed ? tfx("metaElapsed", { s: elapsed.toFixed(0) }) : "",
          step > 1 ? tfx("metaRemaining", { s: Math.max(0, (elapsed / step) * (total - step)).toFixed(0) }) : "",
        ].filter(Boolean).join(" · ");
      }
      if (msg.type === "download_progress" && S.download.key) {
        noteServerEvent();
        const total = msg.total;
        const n = msg.n || 0;
        S.download.bytes = n;
        S.download.total = total || null;
        // The source may not know the total up front (a remote listing can
        // fail). Fall back to a byte readout rather than a frozen 0%.
        if (total) S.download.pct = clamp(Math.round((n / total) * 100), 1, 99);
        renderWeightsList();
      }
      if (msg.type === "download_ok" || msg.type === "download_complete") {
        const label = modelLabel(msg.model_key || S.download.key);
        S.download = { key: null, pct: 0, bytes: 0, total: null };
        toast("ok", tfx("toastReady", { label }), tr("toastReadyDetail"));
        try {
          await refreshBootstrap();
          renderWeightsList(); renderModels(); renderEnv(); renderStorage();
        } catch (_) { /* keep the toast */ }
      }
      if (msg.type === "job_complete") {
        await applyJobComplete(msg.id, msg.image);
      }
      if (msg.type === "job_cancelled") {
        cancelJobUi();
      }
      if (msg.type === "error") {
        if (msg.stage === "generate" && S.jobId) {
          failJobUi(msg.message || "");
        } else {
          setBusy(false);
          S.download = { key: null, pct: 0, bytes: 0, total: null };
          renderWeightsList();
          toast("err", tr("errFailed"), msg.message || "");
          // A failed load is what turns the dot red — pick it up right away.
          if (msg.stage === "generate" || msg.stage === "load") {
            await refreshBootstrap().then(renderEnv).catch(() => {});
          }
        }
      }
    }
  }

  /* ======================================================================
     Bootstrap
     ====================================================================== */
  async function refreshBootstrap() {
    S.bootstrap = await api("/api/bootstrap");
    const config = cfg();
    S.lang = config.language || S.lang;
    S.hub = config.hub || S.hub;
    S.modelKey = config.model_key || S.modelKey;
    renderEnv();
    return S.bootstrap;
  }

  function syncControlsFromConfig() {
    const sizes = (S.bootstrap && S.bootstrap.sizes) || {};
    const scales = sizes.scales || {};
    // The server owns the default scale; only fall back to the first entry.
    if (!scales[S.scale]) {
      const preferred = sizes.default_scale;
      S.scale = scales[preferred] ? preferred : Object.keys(scales)[0] || S.scale;
    }
    if (!aspectList().includes(S.aspect)) S.aspect = aspectList()[0] || "1:1";
    const spec = scaleSpec();
    if (spec) {
      $("outRes").value = spec.output_resolution;
      S.out = spec.output_resolution;
    }
    const pair = sizesForScale()[S.aspect];
    if (pair) { S.width = pair[0]; S.height = pair[1]; }
    $("width").value = S.width;
    $("height").value = S.height;
    $("steps").value = S.steps;
    $("stepsNum").value = S.steps;
    $("cfg").value = S.cfg;
    $("cfgNum").value = S.cfg.toFixed(1);
    if (cfg().vae_tiling != null) $("vaeTiling").value = vaeTilingValue(cfg().vae_tiling);
  }

  /* ======================================================================
     Bindings
     ====================================================================== */
  /* The composer holds one parameter set per mode. `setMode` parks the current
     panel under the mode it belongs to and restores the other one, so tuning a
     generation never disturbs an edit in progress — and vice versa.

     Two things stay out of the snapshot on purpose:
     - `refs` belong to the edit workflow, not to a mode. Parking them under
       "generate" (which never shows them) would wipe them on the way back, and
       a removed reference has its object URL revoked, so a stale copy would
       render a broken thumbnail.
     - `vae_tiling` is an engine setting persisted to config, not a per-run
       parameter; reverting it on a mode switch would contradict that. */
  function readParams() {
    return {
      prompt: $("prompt").value,
      negative: $("negative").value,
      scale: S.scale,
      aspect: S.aspect,
      width: S.width,
      height: S.height,
      out: S.out,
      steps: S.steps,
      cfg: S.cfg,
      seed: $("seed").value,
      nImages: $("nImages").value || "1",
      kv: $("kv").getAttribute("aria-checked") === "true",
      rgba: $("rgba").getAttribute("aria-checked") === "true",
      follow: $("follow").getAttribute("aria-checked") === "true",
    };
  }

  function writeParams(p) {
    if (!p) return;
    S.scale = p.scale;
    S.aspect = p.aspect;
    S.width = p.width;
    S.height = p.height;
    S.out = p.out;
    S.steps = p.steps;
    S.cfg = p.cfg;
    $("prompt").value = p.prompt || "";
    $("negative").value = p.negative || "";
    $("width").value = S.width;
    $("height").value = S.height;
    $("outRes").value = S.out;
    $("steps").value = S.steps;
    $("stepsNum").value = S.steps;
    $("cfg").value = S.cfg;
    $("cfgNum").value = S.cfg.toFixed(1);
    $("seed").value = p.seed || "";
    // Same reason as `applyItemParams`: restoring a snapshot is a write, not
    // typing, so the seed well's state is stale until it is re-read.
    renderSeed();
    $("nImages").value = p.nImages;
    $("kv").setAttribute("aria-checked", p.kv ? "true" : "false");
    $("rgba").setAttribute("aria-checked", p.rgba ? "true" : "false");
    $("follow").setAttribute("aria-checked", p.follow ? "true" : "false");
  }

  /* Park the panel under the mode that owns it, then move to `mode`. */
  function parkAndSwitch(mode) {
    if (mode === S.mode) return;
    S.params[S.mode] = readParams();
    S.mode = mode;
    // A mode visited for the first time has no snapshot yet: it inherits what
    // is on screen, which is what you want when going straight from a
    // generation to editing its result.
    writeParams(S.params[mode]);
  }

  function setMode(mode) {
    parkAndSwitch(mode);
    // Scale and aspect come back with the restored set, so both strips need a
    // repaint — not just the mode-dependent bits. `renderFollowState` runs last
    // because it disables the chips, and those are rebuilt by the two calls
    // above: running it earlier would disable the outgoing elements instead.
    renderMode(); renderScale(); renderRatios(); renderValues();
    renderFollowState(); renderTools();
    if (!$("popPresets").classList.contains("hidden")) {
      $("presetSearch").value = "";
      renderPresets(); placePresets();
    }
  }

  function bind() {
    $("modeGen").onclick = () => setMode("generate");
    $("modeEdit").onclick = () => setMode("edit");
    $("modeGen2").onclick = () => setMode("generate");
    $("modeEdit2").onclick = () => setMode("edit");

    $("prompt").addEventListener("input", renderValues);
    // The negative toggles are derived from the field rather than stored, so a
    // hand edit has to repaint them — otherwise typing the terms in by hand
    // would leave the bundle that describes them sitting dark.
    $("negative").addEventListener("input", () => {
      if (!$("popNeg").classList.contains("hidden")) renderNegPresets();
    });

    /* Presets popover. Both the toolbar button and the empty-stage card open it,
       and they share one handler rather than the card forwarding a `.click()` to
       the button: a forwarded click makes the popover open during the target
       phase, while the *original* click keeps bubbling to `document`, whose
       outside-click rule does not recognise the card as a trigger and shuts the
       popover again — the click looked dead. Sharing the handler also means the
       card can close an open popover, exactly like the button. */
    const togglePresets = (event) => {
      if (event) event.stopPropagation();
      const pop = $("popPresets");
      if (!pop.classList.contains("hidden")) return closePops();
      closePops();
      renderPresets();
      pop.classList.remove("hidden");
      placePresets();
      // Only measurable once the popover is laid out, and the strip is sized
      // by then — chevrons and the active chip both depend on that width.
      updateCatNav();
      revealActiveCat();
      $("btnPresets").setAttribute("aria-expanded", "true");
      $("btnPresets").classList.add("on");
    };
    $("btnPresets").onclick = togglePresets;
    $("startPresets").onclick = togglePresets;
    $("presetSearch").addEventListener("input", renderPresets);
    $("presetSearch").onclick = (event) => event.stopPropagation();
    $("presetLucky").onclick = (event) => { event.stopPropagation(); randomPreset(); };

    /* Negative-preset popover. Unlike the positive library this one stays open
       across picks — the whole point is combining several bundles, and a
       popover that closed on every click would make that a chore. */
    $("btnNegPresets").onclick = (event) => {
      if (event) event.stopPropagation();
      const pop = $("popNeg");
      if (!pop.classList.contains("hidden")) return closePops();
      closePops();
      renderNegPresets();
      pop.classList.remove("hidden");
      // Not every panel state can host the popover (a folded drawer cannot), and
      // `placeNeg` shuts it again in that case rather than leaving it stranded.
      if (!placeNeg()) return;
      $("btnNegPresets").setAttribute("aria-expanded", "true");
      $("btnNegPresets").classList.add("on");
    };
    $("negClear").onclick = (event) => { event.stopPropagation(); clearNegative(); };

    /* Category strip: chevrons, wheel-to-scroll, and the wheel must not steal
       the scroll once the strip has hit either end. */
    const catRow = $("presetCats");
    catRow.addEventListener("scroll", updateCatNav, { passive: true });
    $("catPrev").onclick = () => nudgeCats(-1);
    $("catNext").onclick = () => nudgeCats(1);
    catRow.addEventListener("wheel", (event) => {
      const max = catRow.scrollWidth - catRow.clientWidth;
      if (max <= 2) return;
      const delta = Math.abs(event.deltaY) > Math.abs(event.deltaX) ? event.deltaY : event.deltaX;
      const next = clamp(catRow.scrollLeft + delta, 0, max);
      if (next === catRow.scrollLeft) return; // at an end: let the popover scroll instead
      event.preventDefault();
      catRow.scrollLeft = next;
      updateCatNav();
    }, { passive: false });
    window.addEventListener("resize", updateCatNav);

    /* Empty-state shortcuts */
    $("startRandom").onclick = () => {
      const pool = presetPool().flatMap((cat) => cat.prompts || []);
      if (!pool.length) return;
      const preset = pool[Math.floor(Math.random() * pool.length)];
      $("prompt").value = preset.prompt;
      renderValues();
      toast("ok", tr("toastRandom"), presetTitle(preset));
    };
    $("startLast").onclick = () => {
      const item = S.history[0];
      if (!item) return toast("err", tr("toastNoHistory"), tr("toastNoHistoryDetail"));
      applyItemParams(item);
      toast("ok", tr("toastLastParams"), tr("toastLastParamsDetail"));
    };

    /* Runtime / model popover */
    $("statusBtn").onclick = (event) => {
      event.stopPropagation();
      const pop = $("popModel");
      if (!pop.classList.contains("hidden")) return closePops();
      closePops();
      renderModels();
      pop.classList.remove("hidden");
      placeBelow(pop, $("statusBtn"));
      $("statusBtn").setAttribute("aria-expanded", "true");
      $("statusBtn").classList.add("on");
    };
    $("goDownload").onclick = (event) => { event.stopPropagation(); openSettings(); };

    /* Zoom bars */
    const bindZoomBar = (root, zoom) => {
      $$("[data-z],[data-dz]", root).forEach((btn) => {
        btn.onclick = () => {
          const action = btn.getAttribute("data-z") || btn.getAttribute("data-dz");
          if (action === "in") zoom.step(1);
          else if (action === "out") zoom.step(-1);
          else if (action === "fit") zoom.fit();
          else zoom.one();
        };
      });
    };
    bindZoomBar($("zoombar"), canvasZoom);
    bindZoomBar($("detail"), detailZoom);

    /* Size */
    ["width", "height"].forEach((id) => {
      $(id).addEventListener("input", () => {
        S.width = Number($("width").value) || 0;
        S.height = Number($("height").value) || 0;
        const sizes = sizesForScale();
        const hit = Object.keys(sizes).find((key) => sizes[key][0] === S.width && sizes[key][1] === S.height);
        if (hit) S.aspect = hit;
        renderRatios(); renderValues();
      });
      $(id).addEventListener("blur", () => {
        const snapped = floor32($(id).value);
        if (snapped && String(snapped) !== String($(id).value)) {
          $(id).value = snapped;
          S.width = Number($("width").value);
          S.height = Number($("height").value);
          renderValues();
        }
      });
    });

    /* Sampling */
    $("steps").addEventListener("input", () => {
      S.steps = Number($("steps").value);
      $("stepsNum").value = S.steps;
      renderValues();
    });
    $("stepsNum").addEventListener("input", () => {
      S.steps = clamp(Number($("stepsNum").value) || 40, 4, 80);
      $("steps").value = S.steps;
      renderValues();
    });
    $$("#stepQuick button").forEach((btn) => {
      btn.onclick = () => {
        S.steps = Number(btn.getAttribute("data-steps"));
        $("steps").value = S.steps;
        $("stepsNum").value = S.steps;
        renderValues();
      };
    });
    $("cfg").addEventListener("input", () => {
      S.cfg = Number($("cfg").value);
      $("cfgNum").value = S.cfg.toFixed(1);
      renderValues();
    });
    $("cfgNum").addEventListener("input", () => {
      S.cfg = clamp(Number($("cfgNum").value) || 1, 1, 8);
      $("cfg").value = S.cfg;
      renderValues();
    });
    $("outRes").addEventListener("input", () => {
      S.out = Number($("outRes").value) || 1024;
      renderValues();
    });

    /* Seed. Typing swaps the hint for the clear button, so the state has to be
       re-read on every keystroke — a number pasted in, or one the browser
       restores, arrives without an `input` event of its own. Clearing focuses
       the field again: the button took the focus to be clicked, and leaving it
       on a dead control strands the caret. */
    $("seed").addEventListener("input", renderSeed);
    $("seedClear").onclick = () => {
      $("seed").value = "";
      renderSeed();
      $("seed").focus();
    };

    /* Count stepper. `readonly` keeps the field a value display, so the arrow
       keys and the two buttons are the ways in — and both clamp identically. */
    ["nImagesMinus", "nImagesPlus"].forEach((id) => {
      $(id).onclick = () => {
        const delta = id === "nImagesPlus" ? 1 : -1;
        $("nImages").value = clamp((Number($("nImages").value) || 1) + delta, 1, maxImages());
        renderCount();
      };
    });
    $("nImages").addEventListener("keydown", (event) => {
      const delta = { ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1 }[event.key];
      if (delta === undefined) return;
      event.preventDefault();
      $("nImages").value = clamp((Number($("nImages").value) || 1) + delta, 1, maxImages());
      renderCount();
    });

    /* Switches */
    ["follow", "kv", "rgba"].forEach((id) => {
      const btn = $(id);
      btn.onclick = () => {
        const next = btn.getAttribute("aria-checked") !== "true";
        btn.setAttribute("aria-checked", next ? "true" : "false");
        if (id === "follow") { renderFollowState(); renderValues(); }
        if (id === "kv") toast("ok", `${tr("kvCache")} · ${next ? tr("vOn") : tr("vOff")}`, tr("kvHintShort"));
        if (id === "rgba") toast("ok", `${tr("transparent")} · ${next ? tr("vOn") : tr("vOff")}`, tr("rgbaHintShort"));
      };
    });

    /* Advanced */
    $("btnAdv").onclick = () => {
      const opening = $("advBody").classList.contains("hidden");
      // The negative-preset popover is anchored to a button inside this block,
      // so collapsing the block has to take the popover with it or it would
      // float over the panel pointing at nothing.
      if (!opening) closePops();
      $("advBody").classList.toggle("hidden", !opening);
      $("btnAdv").setAttribute("aria-expanded", opening ? "true" : "false");
      $("btnAdv").querySelector("span").textContent = opening ? tr("collapse") : tr("expand");
      $("btnAdv").querySelector(".chev").style.transform = opening ? "rotate(180deg)" : "";
    };

    /* References */
    $("dropzone").onclick = () => $("fileInput").click();
    $("fileInput").addEventListener("change", (event) => addFiles(event.target.files));
    ["dragenter", "dragover"].forEach((name) => {
      $("dropzone").addEventListener(name, (event) => { event.preventDefault(); $("dropzone").classList.add("drag"); });
    });
    $("dropzone").addEventListener("dragleave", () => $("dropzone").classList.remove("drag"));
    $("dropzone").addEventListener("drop", (event) => {
      event.preventDefault();
      $("dropzone").classList.remove("drag");
      addFiles(event.dataTransfer.files);
    });
    document.addEventListener("paste", (event) => {
      if (S.mode !== "edit") return;
      const files = Array.from((event.clipboardData && event.clipboardData.items) || [])
        .filter((entry) => entry.type.startsWith("image/"))
        .map((entry) => entry.getAsFile())
        .filter(Boolean);
      if (files.length) addFiles(files);
    });

    /* Run */
    $("run").onclick = run;
    $("run2").onclick = run;
    $("cancel").onclick = run;

    $("downloadBtn").onclick = () => {
      if (!S.currentImage) return;
      const a = document.createElement("a");
      a.href = S.currentImage;
      a.download = "imgen.png";
      a.click();
    };
    $("sendEditBtn").onclick = () => sendToEdit(S.currentImage);

    /* History */
    $("historySearch").addEventListener("input", () => {
      S.historyQuery = ($("historySearch").value || "").trim();
      // Searching runs on the server so it reaches records past the first page.
      // Debounced, because every keystroke would otherwise be a round trip.
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => {
        loadHistory().catch(reportError);
      }, 220);
    });
    $("btnHistory").onclick = () => {
      if (compact.matches) {
        const open = $("app").classList.toggle("hist-open");
        $("btnHistory").classList.toggle("on", open);
        $("btnHistory").setAttribute("aria-pressed", open ? "true" : "false");
        return;
      }
      const hidden = $("app").classList.toggle("hide-hist");
      $("btnHistory").classList.toggle("on", !hidden);
      $("btnHistory").setAttribute("aria-pressed", hidden ? "false" : "true");
    };
    $("btnComposer").onclick = () => {
      const open = $("app").classList.toggle("composer-open");
      $("btnComposer").classList.toggle("on", open);
      $("btnComposer").setAttribute("aria-expanded", open ? "true" : "false");
    };

    /* Detail */
    $("dClose").onclick = closeDetail;
    $("dPrev").onclick = () => navDetail(-1);
    $("dNext").onclick = () => navDetail(1);
    $("dExpand").onclick = () => setDetailWide(!S.detailWide);
    $("detail").onclick = (event) => { if (event.target === $("detail")) closeDetail(); };

    /* Settings */
    $("openSettings").onclick = openSettings;
    $("settingsClose").onclick = closeSettings;
    $("settings").onclick = (event) => { if (event.target === $("settings")) closeSettings(); };

    $("langToggle").onclick = async () => {
      S.lang = S.lang === "zh" ? "en" : "zh";
      applyI18n();
      await putConfig({ language: S.lang }).catch(() => {});
    };

    /* Popovers close on an outside click. The DOM guard matters: a handler that
       rebuilds a list detaches the event target before it reaches this point. */
    document.addEventListener("click", (event) => {
      if (
        $("popPresets").classList.contains("hidden") &&
        $("popNeg").classList.contains("hidden") &&
        $("popModel").classList.contains("hidden")
      ) return;
      if (!document.contains(event.target)) return;
      if (event.target.closest(".pop") || event.target.closest("#btnPresets") || event.target.closest("#btnNegPresets") || event.target.closest("#statusBtn")) return;
      closePops();
    });

    document.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        run();
        return;
      }
      if (event.key === "Escape") {
        if (!$("detail").classList.contains("hidden")) return closeDetail();
        if (!$("settings").classList.contains("hidden")) return closeSettings();
        if (!$("firstRun").classList.contains("hidden")) return;
        if (!$("popPresets").classList.contains("hidden") || !$("popNeg").classList.contains("hidden") || !$("popModel").classList.contains("hidden")) return closePops();
        return;
      }
      if (!$("detail").classList.contains("hidden")) {
        if (event.key === "ArrowLeft") { event.preventDefault(); navDetail(-1); }
        if (event.key === "ArrowRight") { event.preventDefault(); navDetail(1); }
      }
    });

    window.addEventListener("resize", () => {
      canvasZoom.resize();
      detailZoom.resize();
      placePresets();
      placeNeg();
      updateCatNav();
      if (!$("popModel").classList.contains("hidden")) placeBelow($("popModel"), $("statusBtn"));
    });
    narrow.addEventListener("change", () => {
      $("app").classList.remove("composer-open");
      $("btnComposer").classList.remove("on");
      // Crossing the breakpoint folds the drawer, and every popover is anchored
      // to something inside it — they have to go with it.
      closePops();
      placePresets();
      canvasZoom.resize();
    });
  }

  /* ======================================================================
     Init
     ====================================================================== */
  async function init() {
    bind();
    try {
      await refreshBootstrap();
    } catch (err) {
      reportError(err);
      return;
    }
    syncControlsFromConfig();
    applyI18n();
    renderScale();
    renderRatios();
    renderStage();
    renderTools();
    renderRun();
    try {
      await loadHistory();
    } catch (err) {
      reportError(err);
    }
    connectWs();
    if (!cfg().setup_completed) openFirstRun();
    // A reload or a second tab should pick up a job the engine is still
    // running rather than sit idle while the picture is being made.
    await adoptRunningJob().catch(() => {});
  }

  init().catch(reportError);
})();
