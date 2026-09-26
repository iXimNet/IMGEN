# Changelog

## Unreleased — interface overhaul

**Breaking (HTTP):** `GET /api/jobs` and `GET /api/jobs/{id}` no longer return
`image_path`, `thumb_path` or `ref_paths`. They return `ref_count` and
`ref_urls` instead, so the browser never sees absolute paths on disk.

### Canvas
- Zoom viewport: fit / 1:1 / stepped zoom / cursor-anchored `⌘`+wheel / drag to
  pan / double-click to toggle. Zoom bar with a live percentage.
- The stage is a matte now — no more checkerboard field competing with the
  picture, and a 1 px ring so white-background images stay readable.

### Composition panel
- **Generate and edit keep separate parameters.** The composer used to hold one
  set, so tuning a generation overwrote an edit in progress and vice versa.
  Switching modes now parks the panel under the mode that owns it and restores
  the other set — prompt, negative prompt, scale, aspect, width/height, the
  reference area, steps, CFG, seed, image count, and the KV / RGBA / follow
  switches. A mode visited for the first time inherits what is on screen, which
  is what you want when going straight from a generation to editing its result.
  References and `vae_tiling` stay out of the snapshot: references belong to the
  edit workflow (parking them under "generate", which never shows them, would
  wipe them, and a removed reference has its object URL revoked), and
  `vae_tiling` is an engine setting persisted to config rather than a per-run
  parameter.
- "Follow last reference aspect" moved out of the reference-scale block and into
  the 成图 section, above the scale chips — it decides the output geometry, so it
  belongs with the geometry. While it is on, the scale chips, the seven aspect
  presets and the width/height inputs are disabled and dimmed, because the engine
  ignores all three in that state. The reference area size stays live: it is what
  the reference is resized to, and the readout already reflected it.
- Fixed the follow-the-reference size: the engine derived it from the scale's
  area (1024 / 2048) while the readout used the reference-area input, so a custom
  area produced a size the studio never showed. `follow_reference_size()` now
  takes the resolved area, and the engine reads that input first — the two agree.
- The run button spans the full row and stays optically centred. Cancel no longer
  reserves a 44 px slot beside it — it floats inside the primary's right end — so
  the primary is no longer pushed 26 px left of centre, and it still keeps the
  same width before, during and after a run.
- Flat sections separated by hairlines instead of nested boxes; each section
  shows its current value next to the title.
- Aspect presets are drawn at their real proportions instead of seven text chips.
- Steps and CFG values are editable numbers with 20 / 40 / 60 quick presets.
- Prompt placeholder and hint now say different things instead of repeating.
- The run button keeps its width while busy — cancel sits in a reserved slot.

### History
- Clicking a row opens a detail overlay with a zoomable large image, grouped
  parameters and the full prompt. **Browsing no longer writes to the left panel
  or the canvas**; "Reuse settings" is an explicit button.
- The parameter groups in the detail panel keep their rounded frame but drop the
  **internal** gridlines: the cells no longer paint a background and the row gap
  is zero, so the frame reads as one continuous surface. Row height follows the
  content instead of the old 52px min-height, which takes the twelve cells from
  317px+ down to 279px. The settings sheet keeps the fully bordered card
  (`.kv` vs `.kv.flat`).
- The prompt group heading carries a right-aligned **copy** button (the negative
  prompt gets one too when present). It writes to the clipboard via
  `navigator.clipboard` with an `execCommand` fallback, because the studio is
  often served over plain http on a LAN address, which is not a secure context.
- Edit records carry a reference filmstrip under the viewer; click a tile to
  swap the main view. `←` `→` walk the history, `Esc` closes.
- The filmstrip no longer carries a "viewing · …" text label. Which image is on
  screen is shown by the tile itself: the current one keeps full brightness with
  a light frame, a marker dot and a slight lift, while the rest sit at 55%
  opacity. `aria-current` marks it for assistive tech, and each tile's label now
  describes the tile ("reference 1") instead of claiming it is the one being
  viewed.
- The detail header drops the duration — the spec sheet already shows it, so the
  line is now just timestamp · size. Rows carry `data-id`, which lets a test find
  a record by identity rather than by its position in the list.
- **Reference images keep their own aspect ratio in the viewer.** They used to be
  sized from the record's `width`/`height`, which describe the *output* — a
  635×956 portrait reference was drawn as 1103×717, a 2.3× horizontal stretch.
  The viewer now adopts the decoded bitmap's real dimensions, with `object-fit:
  contain` as a backstop so a stale size can letterbox but never distort.
- Rows are grouped by day and show the time.
- **The list pages instead of silently capping at 80.** `GET /api/jobs` now
  returns `total`, `offset` and `limit`, so the list knows whether another page
  exists rather than guessing from the page size. A "load more" button sits at
  the end and becomes "no more records" once the set is exhausted — the records
  past 80 used to be unreachable, not merely hidden.
- **Search runs on the server.** It used to filter only the page already in
  memory, so a record past the first 80 could never be found. Typing is
  debounced, and clearing the box restores the first page.
- `history.count()` and `history.list()` share one WHERE clause, so the total
  can never disagree with what the list returns.
- Rows no longer carry hover action buttons. Reuse, download and delete live in
  the detail overlay only, so the list has one behaviour: open the record. The
  row is a button and says so with a pointer cursor.
- The second line drops the reference count (visible in the detail overlay) and
  pushes the clock to the right edge. Its tooltip carries the full date, since
  the row itself only has room for the time.
- **A record with no picture no longer shows the previous record's image.** The
  viewer now clears itself and states "this run produced no image"; download and
  send-to-edit disable, reuse-settings stays available (the parameters are still
  valid). A missing reference image says so in the same place.
- The two history arrows point opposite ways. The markup was missing the
  `prev` / `next` classes, so neither rotation rule matched and both chevrons
  pointed down.
- The sheet is no longer capped at 1140px: width now follows the same rule as
  height — fill the overlay, bounded by the viewport — so the picture gets the
  room. At 1600px wide it measures 1560px instead of 1140px.
- Zoom controls fade out unless the pointer is over the picture, so the image is
  never competing with a floating toolbar — in both the detail overlay and the
  stage viewer. They stay clickable and focusable (`opacity`, not `visibility` —
  a control that leaves the accessibility tree cannot be reached without a mouse
  first), and focusing one reveals the bar. On touch devices they stay visible.
- Fit mode shows no scrollbars. The inner wrapper keeps a zoomed image
  scrollable back to its top-left, but at fit size its rounded dimensions left a
  few pixels of overflow; scrollbar chrome now appears only once the view is
  actually pannable, and the wrapper is capped while it is not.

### Top bar
- **New brand mark: an aperture ring around a glowing safelight core.** The old
  one was two CSS pseudo-elements — a flat tile with a plain dot — which said
  nothing about what the app does. The new mark keeps the safelight idea (the
  app's only saturated colour) and adds the lens/aperture read that fits an
  image studio. It lives in the sprite as `#i-brand` on a 48-unit grid and is
  drawn at 22px in the header; the favicon uses the same artwork with the ticks
  reduced from eight to four, because eight turns to mush at 16px.
- The status pill opens a runtime popover that switches between weights already
  on disk, with a "download more models" entry.
- A **GitHub** icon at the far right links to the repository. It is a real
  anchor (`target="_blank"` + `rel="noopener noreferrer"`), so middle-click and
  open-in-new-tab work without any scripting, and its tooltip follows the
  language. The glyph is a stroked Feather path in the shared sprite, matching
  the other top-bar icons.
- The panel button now toggles the history panel only.
- **The dot reports health, not residency.** It used to turn red when a pipeline
  was resident — so a perfectly healthy studio showed an error colour, and it
  only appeared after a refresh because nothing handled `load_complete`. Now:
  amber for a device warning, red only for a failed load, green otherwise. The
  reason (or "no weights loaded yet, they load on the first run") goes on the
  pill's title, where it is actually discoverable.
- `Engine` gained `last_error`, cleared when a load starts and set when one
  fails, so health has its own field instead of being inferred from `loaded`.
  `load_complete` and error events refresh the pill live — no reload needed.

### Model & download
- Renamed from "Settings" (it used to inherit the title "Generation settings").
- Download moves onto each model row, labelled Download / Re-download / Resume,
  with inline progress. Tokens are stored when a download starts, so the footer
  no longer needs a save button.
- Weight presence is now reported per source *and* overall: a model downloaded
  from the other hub reads "on disk · from Hugging Face" instead of "not
  downloaded", and picking it switches the download source along with the model.
  Generating resolves the source the same way, so on-disk weights are used even
  when the studio is pointed at the other one.
- "Not downloaded" is only said when that is true; partial snapshots now report
  the number of missing shards and offer Resume instead.
- ModelScope snapshots are looked up under `hub/models/<namespace>/<name>` — the
  layout current ModelScope builds use — plus the older `hub/` and cache-root
  paths. `HF_HUB_CACHE` is honoured for Hugging Face.
- The storage block now names the real weights directory for the selected source
  (`HF_HUB_CACHE` > `HF_HOME` > `MODELSCOPE_CACHE` > platform default), with a
  badge saying which setting decided the path, and it follows the download source
  as you switch. The output directory is read from the app paths instead of the
  hard-coded `~/.imgen/models`, which never existed.

### Corrected defaults — Image21-INT8 errata (2026-09-24)
- **VAE tiling now defaults to off and nothing turns it on implicitly.** The card
  traced the short vertical stains in 2048px edits to tiled VAE decoding — the
  same latent decoded untiled is clean — and withdrew its earlier general 2048px
  recommendation. `auto` is gone from the panel, `resolve_vae_tiling()` no longer
  consults mode or resolution, and a stored `auto` is retired to `off` when the
  config loads — so existing installs need no manual edit.
- **The default scale is now 1K.** New `DEFAULT_SCALE = "1k"` in the constants, used
  by the form defaults, the engine fallback and `sizes.catalog()`, so the studio
  opens at 1024 × 1024 with a 1024px reference area. The 2K table is untouched and
  one click away; per-history "reuse settings" still restores whatever a record
  used. The frontend reads `sizes.default_scale` instead of hard-coding a table.
- The parameter copy no longer sells anything as an official recommendation:
  `原生 2K（官方推荐）`, the `官方推荐` note next to 步数, and "这是官方默认" in the
  CFG tip are gone. Scale chips are descriptive — `原生 2K` and `1K · 更省显存`.
- The Image21-INT8 note no longer says "edit at 2048 with VAE tiling". It now
  reads "start edits at 1024, keep VAE tiling off", and stays reachable as a
  hover title on the model row instead of only appearing while undownloaded.
- `README.md` carries the same corrections in the model table, the defaults table
  (new `vae_tiling = false` row) and the consumer-GPU note.

### Presets
- The category strip is scrollable with a mouse: a chevron appears on whichever
  side still has chips, a plain wheel scrolls the strip horizontally (and stops
  hijacking the wheel once it hits either end), and the active category is
  scrolled into view without yanking the strip when it is already visible. The
  scrollbar itself stays hidden.

### Fixes
- **Switching language now repaints everything that reads the language.**
  `applyI18n()` re-ran a hand-picked list of renderers, and the scale chips
  (`renderScale`) were not on it, so 原生 2K / 1K kept the old language until the
  next unrelated repaint. The same gap hit the open detail overlay (title, status
  pill, footer buttons, group headers), the reference-thumbnail delete labels and
  the run button. All of them re-run now, and a test derives the required list
  from the source so a new renderer cannot be forgotten again.
- Removed the stage's "reuse settings" button — it did nothing. Its handler
  only raised a toast, and the toast claimed "settings written to the left panel,
  prompt/size/sampling restored", so it lied as well. The composer already holds
  the parameters for the current run; reusing a *past* record's settings belongs
  to the history detail overlay, which has its own working button. The stage
  toolbar is now download + send-to-edit.
- Preset categories no longer close the popover when clicked (the list rebuild
  detached the event target before the outside-click check ran).
- Chinese labels are no longer letter-spaced and uppercased.
- Fonts load from the system stack — Google Fonts is unreachable on many
  China networks, which silently degraded the whole type scale.
- `:focus-visible` ring everywhere; switches expose `role="switch"` and
  `aria-checked`; labels are bound to their inputs.
- Toast messages carry a human line plus the backend detail, with a close button.
- `--demo` reports the new per-source weight fields, so the popover no longer
  came up empty in demo mode.
- **ModelScope downloads now report progress and are found afterwards.** Three
  defects compounded: the studio only probed the pre-1.38 cache layout
  (`hub/models/<owner>/<name>`) while current SDKs write
  `models/<owner>--<name>/snapshots/<revision>/`, so a finished download looked
  absent and the Download button came straight back; the progress watcher
  reported `total: null` (and sampled the whole cache root rather than the
  repo), so the row sat at 0% even while bytes were landing; and the ignore
  patterns were regex-style (`.*\.png$`) while ModelScope matches with
  `fnmatch`, where `$` is a literal — nothing was ever skipped, so README and
  every gallery image came down with the weights. The lookup now walks both
  layouts and descends into `snapshots/`, the watcher derives a real byte total
  from the remote file list (falling back to a byte readout when that listing
  fails) and samples only the repo's own directory, and one glob pattern list
  serves both hubs.

## 0.1.0 — 2026-09-21

- First public studio: generate, multi-reference edit, history, bilingual UI.
- First-run download from Hugging Face or ModelScope.
- Qwen-Image-2.1 (BF16) and Image21-INT8 (CUDA) loaders.
- Official 2K aspect table, 40-step / CFG-off defaults, RGBA template.
- Categorized preset prompts and `--demo` mode for UI-only use.
