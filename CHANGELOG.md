# Changelog

## Unreleased — interface overhaul

**Breaking (HTTP):** `GET /api/jobs` and `GET /api/jobs/{id}` no longer return
`image_path`, `thumb_path` or `ref_paths`. They return `ref_count` and
`ref_urls` instead, so the browser never sees absolute paths on disk.

They also carry `image_urls` / `thumb_urls` / `output_count` now, and
`params.extra_images` (absolute paths) is stripped before the row leaves the
server — see **Multi-image runs** below.

### Multi-image runs
- **A run with `num_images > 1` made several pictures and showed one.** The
  engine has always returned every frame and `outputs/` has always held all of
  them, but only `images[0]` reached the browser: `_public_job()` published a
  single `image_url` / `thumb_url`, and the extras lived in
  `params.extra_images` as absolute paths, which nothing in `static/` ever read.
  Three pictures sat on disk, unreachable from the canvas, the history list and
  the detail overlay alike.
- The record now publishes `image_urls` / `thumb_urls` (one per frame, in the
  order the engine made them) plus `output_count`, and drops the paths.
  `image_url` / `thumb_url` stay as the first frame, so older callers keep
  working. The extras were always in the database, so **existing records light
  up as they are — no migration**.
- Canvas: a frame strip under the picture, on its own row so the chooser never
  covers what it chooses between. Click a tile to switch the canvas; the
  toolbar's download and send-to-edit follow the selection. It stays hidden for
  a single-frame run, where there is nothing to choose.
- History: a `×N` badge on the thumbnail, because the row can only show one of
  them and "there are three more" is the useful fact.
- Detail overlay: the filmstrip is no longer edit-only. It lists **every frame
  of the run, then the references**, each group numbered from 1 with a wider gap
  at the seam. A generation with four frames therefore gets a strip too. The
  footer names the frame its buttons will act on ("Frame 3 of 4") and its
  download / send-to-edit act on the tile on screen rather than always the
  first.
- **Edit can make several frames too.** The sampling section is shared by both
  modes, so the control was always reachable while editing and the engine always
  honoured it; verified against a real run that an edit with `num_images=3`
  produces three distinct pictures, laid out as `frame frame frame │ reference`.
- `History.delete()` now removes the extra frames. It only unlinked
  `image_path` / `thumb_path` / `ref_paths`, so deleting a four-frame run left
  three pictures and three thumbnails behind, invisible to every screen.
- **Security:** the extra-frame fallback in the output/thumb routes joined the
  request path onto the outputs folder unchecked, so `GET
  /api/outputs/..%5Csecret` returned a file from outside it. The name must now
  match the id shape (`16 hex`, optionally `_<n>`).

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
- **The overlay is two columns now: picture left, panel right.** The old layout
  stacked a header band, the picture and a footer band, and the picture is
  height-bound — measured 677×677 in a 1145×721 viewport, with 424px of unused
  width. Every row spent on chrome was a row the picture did not get. The bands
  are gone; the panel carries the title, the metadata, the thumbnails and the
  actions, so the picture keeps the full height of its column: **774×774, +14.3%**,
  with chrome down from 197px to 166px (21% → 17% of the overlay).
- The panel is a three-row grid — head (`详情` + prev/next/close), scrolled body,
  fixed action row — so the actions never scroll away from the record.
- A **任务** group joins the spec sheet: when the run happened (full timestamp),
  whether it was a generate or an edit, its status, and the record id. Status
  moved down from the old header pill rather than being dropped, and failed runs
  now show `error` here too — otherwise the reason disappeared with the pill.
- The id is **selectable** — it is the one value a user may need to copy into a
  bug report or a filename. It sits in the stage's top-left corner under the same
  rule as the zoom bar (fades in with the pointer, permanently visible on a
  touch device). It takes pointer events only while visible; leaving it
  permanently clickable would silently swallow drags started in that corner.
  **Each frame shows its own id** (`{run}` for the first, `{run}_1` … after) —
  they used to all print the run id, which made a four-image run look like four
  copies of one file. A reference has no id of its own (it is stored as
  `refs/{run}/ref_{nn}.png`), so the box is hidden rather than filled with a
  made-up value.
- **Card spacing now has an order**: label→value 2px < row→row 6px < row→card edge
  10px. The padding moved from the cells to the card so the edge gap is
  independent of the row gap; previously the row gap (10px) exceeded the edge
  gap (5px), which read as cramped despite the generous spacing.
- Thumbnails moved into the panel and wrap into a grid — a 360px column cannot
  scroll a single row without hiding half of it. The group is named **on its
  first tile** (`成图` / `参考图`, above the tile so the picture stays
  unobstructed), left-aligned with that tile, instead of by a heading that
  restated itself. A lone reference now shows at all; the old rule required more
  than one picture in total.
- **The prompt folds past five lines** with a right-aligned "显示全部" link, and
  the negative prompt folds the same way. The link only appears when the text
  actually overflows — a control that reveals nothing is worse than no control.
  `-webkit-line-clamp` turned out to be a no-op in this Chromium build, so the
  fold is a `max-height` locked to a whole number of lines plus a fade; leaving
  `padding-bottom` in place let the next line show through the transparent
  padding, which is why the first fix still leaked half a line.
- Download moved into the zoom bar under the picture (both act on the frame on
  screen), and the "第 N / M 张" counter is gone — the highlighted tile already
  says which frame the buttons will act on.
- Delete sits alone at the left of the action row, spaced away from reuse and
  send-to-edit. Its confirmation now stacks: the message takes its own row with
  the buttons beneath, instead of squeezing both onto one line.
- **The detail viewer can take the whole window.** A toggle in the viewer's
  top-right corner drops the panel out of the grid and hands its 360px to the
  picture; the overlay's own 20px frame goes with it, so the same picture gets
  the whole window with nothing cropped, stretched or re-zoomed. It is
  deliberately **not** the Fullscreen API: the overlay stays an overlay, so
  `Esc`, click-outside-to-close and the window chrome keep behaving as they did,
  and the panel is only `display: none` — `←` `→` still walk the history while
  it is away. A fitted picture is re-fitted to the new box, while a 1:1 or a
  free zoom is the user's own number and is left alone. The button carries its
  state (`aria-pressed`) and its label names the *next* click, so it flips with
  the state and follows the language like every other readout. It earns its
  keep on landscape frames, where width was the binding constraint: the demo
  strip goes from 1165×457 to 1568×615 at a 1600×1000 window. A record with no
  picture offers no toggle — there is nothing to look at — but if one is reached
  by walking the history *while filled*, the toggle stays: it is the only way
  back to the panel.
- **The detail panel is no longer stretched by its own widest row.** In English
  the footer's three labels ask for 398px inside a 360px panel, and the panel's
  single grid column was implicit — an `auto` track, which is sized by the
  widest row. The track grew to 398, and because every row stretches to the
  track, the metadata cards came with it: they ended 20px past the panel's right
  edge, having spent their right padding, so the right-hand column of values sat
  flush against the cut and the last button was sliced by the sheet's
  `overflow: hidden`. The column is now declared (`minmax(0, 1fr)`), so a row
  that does not fit overflows on its own instead of dragging its siblings out
  with it — cards measure 324px again, with both paddings intact.
- The footer's two picture actions (`Reuse settings`, `Send to edit`) are now
  one flex item, so when the labels cannot share the row with `Delete` they move
  to a second row **together**, right-aligned, instead of the last one being cut
  off. Half a row beats half a button. The second row is only spent when the
  words need it: Chinese fits on one line and still renders as one row, and the
  panel's `1fr` body absorbs the extra 41px, so the picture loses nothing.
  Verified in a real browser at 960/1024/1280 wide in both languages, plus the
  narrow layout, the delete confirmation, a four-frame record and fill-the-window.

### Top bar
- **New brand mark: an aperture ring around a glowing safelight core.** The old
  one was two CSS pseudo-elements — a flat tile with a plain dot — which said
  nothing about what the app does. The new mark keeps the safelight idea (the
  app's only saturated colour) and adds the lens/aperture read that fits an
  image studio. It lives in the sprite as `#i-brand` on a 48-unit grid and is
  drawn at 22px in the header; the favicon uses the same artwork with the ticks
  reduced from eight to four, because eight turns to mush at 16px.
- **The wordmark is artwork now.** The supplied IMGEN logotype ships as a
  transparent PNG (`imgen/static/brand/`) instead of live text next to the mark.
  The header takes `imgen-wordmark-dark.png` — the original is near-black, which
  on `#16181b` measures **1.00:1** and is invisible; reversing it to `--ink`
  gives 14.7:1. The README takes the original black at 17.8:1, and a gold variant
  is kept as a spare. `alt` carries the name and follows `data-i-alt`, so the
  accessible label still switches language (`IMGEN` is the same in both).
  Below 460px the wordmark is dropped and the mark stands alone.
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
- **The pill sizes to its content instead of a fixed cap.** It was capped at
  340px, which cut "ModelScope 魔搭" down to "Model…" on this machine — the three
  things it names (device, model, source) all vary per machine, so a fixed number
  guarantees truncation on some configuration. It now grows with its text and
  only shrinks when the bar genuinely runs out of room, with a 56px floor so a
  squeezed pill still shows its dot and chevron. Both halves are clipped to the
  rounded edge while shrinking; a flex container does not clip by default, so the
  chevron was ending up outside the border.
- **It no longer names a weight source at all.** `S.hub` is the *preference* —
  what the config says or what was last picked in the sheet — not where the
  weights came from: `resolve_local_hub` silently falls back to whichever source
  actually holds a snapshot, logging "using weights from X (Y has no local copy)"
  while the pill still showed Y. It was wrong on exactly the configuration it
  appeared to describe. The sheet behind the pill reports `local_hub`, the real
  location, and every history record stores the resolved value, so nothing is
  lost. The pill is 82px narrower for it.
- **Below 900px the top bar no longer repeats the mode switch.** The bottom bar
  carries its own 生图/改图 next to the run button — the one in reach there — so
  the top bar's copy spent 166px restating it, and was the reason the status pill
  had to shrink at all. With the duplicate gone, every width from 380px to 1920px
  measures zero overflow in the bar.

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
- **Extra weight folders.** The first-run sheet shows where downloads land (the
  compact weights line, following the selected source), and both the first-run
  and settings sheets let you add folders that already hold weights — a
  hand-download, another tool's cache, another drive. Folders are only ever
  searched, never written to: a download still lands in the hub's own cache.
  Adding goes through `POST /api/weights/check-dir`, which classifies the path
  (missing / not a folder / which models it holds) before anything is stored,
  and `POST /api/weights/pick-dir` opens the OS folder picker where one is
  available. Every candidate is anchored to a repo by name — the plain
  `owner/name`, the flat `owner--name` id, or a hub cache layout — so a folder
  holding Image21-INT4 never satisfies a lookup for another model, and the hub
  cache always outranks an extra folder when both hold a copy. A model whose
  weights come from an added folder reads "ready · from another folder" on its
  row and generates from there directly.
- **Folders named after their precision are read by content.** A working copy is
  often called `bf16` or `int8` rather than after its repo, and name-anchored
  probing could never match one — the folder was accepted as a search path and
  then found nothing, forever. Such a folder is now sniffed instead: the
  snapshot's own `model_index.json` has to name the Qwen-Image-2.1 pipeline, and
  its `conversion.json` has to say which precision it is (`bitsandbytes` → INT8,
  `sdnq`/`uint4` → INT4, no conversion metadata → BF16). Anything else is
  refused rather than guessed at, so a quantised snapshot that does not record
  its method is left alone — INT8 and INT4 are not distinguishable from file
  names, and loading the wrong one is worse than asking for a rename. Named
  candidates are still probed first, so a folder that does say who it is is
  never overruled by an unnamed sibling. The scan descends up to three levels
  (skipping dot-folders), so a dump that keeps a tool folder in between is
  reached too, and its result is cached for a few seconds — one walk answers
  every model and both hubs, instead of re-reading the disk per lookup.
- `dirNoModels` no longer promises "the folder is still searched". That was true
  only while every folder was name-matched; now that unnamed folders are read,
  the honest answer is that nothing usable was found in it.

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
  CFG tip are gone. Scale chips are descriptive — `原生 2K` and `1K`. The `1K`
  chip dropped its `更省显存` subtitle, which brought the segmented control to the
  same height as the other single-line controls (36px) — the reserved subtitle
  row inside `.seg` was pure height once the subtitle was gone.
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
- **Negative prompts got a preset library too.** The advanced block offered one
  empty field and a placeholder, while the positive prompt had a whole library a
  click away. A `预置` button now rides on the field's own label row and opens
  seven bundles — 基础兜底, 画质, 人体结构, 文字与标志, AI 感与伪影, 构图问题,
  风格跑偏 — shipped in `prompts.json` next to the positive ones. The button sits
  *beside* the label rather than inside it: a button nested in a `<label>` would
  also forward the click to the field it points at.
- **A bundle is a toggle, not a fill.** Clicking one merges its terms into the
  field, clicking it again takes them back out, so bundles combine and the
  user's own wording is never overwritten. On/off is therefore *derived* from the
  textarea on every repaint instead of stored: typing a bundle's terms by hand
  lights it up, dropping one puts it out, and there is no second source of truth
  to drift. Removal works term by term and ignores case and spacing. The bundles
  are kept pairwise disjoint — asserted in the tests — because a term shared by
  two bundles would be torn out of a bundle still meant to be on.
- Both languages carry their own term list, so a click inserts Chinese in Chinese
  and English in English, and a bundle applied in one still reads as applied —
  and still clears completely — in the other. The popover opens *beside* the
  panel rather than under its trigger: dropped underneath it would cover the very
  field whose filling is the only feedback a toggle gives. It also goes away with
  the drawer when the window crosses into the narrow layout, where a folded
  drawer leaves it nothing to anchor to.

### Fixes
- **An INT8 run stuck for minutes in conditioning was reported as fast.** The
  "expected to be slow" flag on the `condition` stage was `_group_offload()`,
  which only recognises the INT4 group-offload recipe, so the run that prompted
  this — an INT8 2K edit that sat there for 19 minutes — showed a silent bar.
  Two independent things make that stage long, and the flag now knows both:
  weights being **streamed** (INT4 group offload, or an INT8 pipeline loaded
  with CPU offload) and a **large reference workload**. The second one turned
  out to be the dominant cost and is not about precision at all: the pipeline
  resizes every reference to the output-resolution *area*, so a 2K edit gives
  the vision tower roughly four times the pixels of a 1K one, and the cost grows
  faster than the token count. Measured on the INT8 path with the weights
  resident and two references: **3.1s at 1K, over 8 minutes at 2K**. The
  threshold sits at the geometric midpoint of those two points (4 megapixels of
  reference area), so the flag errs towards announcing a stage the reader will
  actually wait on. Generation is never flagged here — a text-only prompt never
  runs the vision tower. The note text gained the large-reference case, since it
  previously named low-VRAM streaming only.
- **Image21-INT8 forced CPU offload on every machine.** `_load` hard-coded
  `offload = True` for the INT8 loader, so `enable_model_cpu_offload()` handed
  every module to the accelerator for its turn and took it back afterwards —
  even on a card that can hold the ~18.6 GB of weights outright, where the
  copies buy nothing but the streaming that made conditioning slow in the first
  place. INT4 has chosen its placement from the card's VRAM since it was
  written; INT8 now does the same. The floor is the BF16 path's own headroom
  (~7 GB above the weights, i.e. 25.6 GB here), so a 24 GB card still streams
  and a 32 GB card keeps the weights resident. Verified on a 32 GB RTX 5090:
  `cpu_offload: false`, ~19 GB on the card, 20s to load, and a 1K two-reference
  edit whose conditioning took 3.1s. The sequential load is unchanged — that
  guards the *transient* peak, which is a separate question. Note that this does
  **not** rescue 2K: conditioning there is minutes long either way, so a 2K edit
  is still best run on BF16 (or at `true_cfg_scale=1.0`, which halves the encode
  by turning off the negative-prompt pass).
- **"送到改图" also overwrote the prompt.** The button was meant to put the
  picture on screen into the edit panel as a reference image, and it did — but
  `sendToEdit` then ran `if (prompt) $("prompt").value = prompt`, replacing
  whatever the user had typed with the source record's own prompt. Silently: no
  confirmation, no undo, and the toast said only "已作为参考图送到改图", which
  told the user to name the image themselves while the box had already been
  filled in. The replacement persisted too — parking a mode on the way out
  captures the box as it stands — so a draft was gone for good. The two call
  sites disagreed about it, which is what gave the bug away: the detail footer
  passed `item.prompt` while the stage toolbar passed `null`, so the same button
  meant two different things depending on where it was clicked. Sending a
  picture is now sending a picture and nothing else; the prompt box is left
  alone. Restoring a record's prompt is what 复用参数 is for, and it restores the
  negative prompt with it — carrying over half of a settings restore was worse
  than carrying over none of it. Asserted from the source in the tests, since
  the behaviour lives in the DOM.
- **A reference image picked for an edit leaked into the next generation.** The
  panel keeps `S.refs` across a mode switch — rightly, since a reference the user
  just chose should survive a trip through 生图, and `sendToEdit` sets the mode
  *before* it adds the file — but the reference section is hidden in 生图, and the
  run posted the files anyway. What made it insidious is that the picture came out
  correct: the engine has always ignored them (`images=refs if mode == "edit" else
  None`), while the server saved them and the detail overlay listed a 参考图 the
  run never used. Closed at three layers — the client does not send them, the
  server does not decode or store them, and `_public_job` does not publish them —
  so a record written before this fix also stops showing a reference it ignored.
  The stored paths stay in the row for `delete()` to clean up.
- **"送到改图" stayed enabled on a record with no picture.** The rewrite that
  moved the actions into the panel kept the dimmed styling but dropped the
  `disabled` attribute, so the button still took focus and clicks and would send
  an empty source. It disables together with download now, and carries the same
  `dim` treatment.
- **The prompt fold measured while the overlay was still hidden.** Sizes are all
  zero under `display: none`, so `scrollHeight > clientHeight` was never true and
  the "显示全部" link never appeared for any prompt, however long. The fold is
  wired after the overlay is shown; the language switch re-runs it, since the
  link text changes with the language.
- **`-webkit-line-clamp` was doing nothing, here and in the history cards.** This
  Chromium build normalises `display: -webkit-box` to `flow-root` and does not
  recognise the standard `line-clamp` at all (the computed value is `null`), so
  every clamp silently did nothing. The visible symptom was a prompt box at the
  right height with the next line showing through underneath. Both places now use
  a `max-height` locked to a whole number of lines plus a bottom fade, with
  `padding-bottom: 0` while folded — the transparent padding is what let the extra
  line show.
- **The history card summary was never truncated either** — same root cause. Its
  two-line clamp has been a no-op since it was written, so long prompts ran the
  card on for as many lines as they needed.
- **"打开预置库" on the empty stage did nothing.** The card forwarded a synthetic
  `.click()` to the toolbar button. That opened the popover during the target
  phase, but the *original* click kept bubbling to `document`, whose
  outside-click rule does not treat the card as a trigger — so it closed the
  popover again inside the same event. The card and the button now share one
  handler instead of one forwarding to the other, which stops the event where it
  should and also lets the card close an open popover like the button does.
- **An edit run no longer sits in a silent stage.** Everything around the
  sampler — reading the prompt and the reference images, encoding those
  references through the VAE, and the decode at the end — carries no per-step
  signal, and all of it used to be invisible: the studio showed "载入模型（首次
  较慢）" (set when the pipeline call starts) and the console said nothing, which
  is exactly what a hang looks like. The engine now announces each stage the
  moment it is really entered (`generate_phase` with `condition`, `encode` or
  `decode`), times each one on the console, and reports the live stage through
  `/api/jobs` so every tab — including one that just reloaded — can say what the
  run is doing and for how long. Sampler steps always read as sampling: the old
  "step ≤ 2 → loading model" label named a stage the run had already left.
- **The VAE's encode direction borrows the accelerator as well.** Edit runs
  encode their references *before* sampling, and that half of the VAE was still
  pinned to the CPU, so a 1024px edit crawled for tens of minutes at the exact
  point the studio could not see. Encode and decode now share one borrowing
  helper: the VAE moves over for its turn, returns to the CPU afterwards, and a
  turn that runs out of VRAM latches back to the CPU so the run still finishes.
  Both directions report where they landed (`vae_encode` / `vae_decode`) — and
  the fallback is now recorded where the studio actually reads it, which the
  decode-only version of this was not.
- **A run that stops reporting says so, and a stop really stops.** After a
  minute without progress the studio states plainly that it has not heard
  anything — it may still be encoding/decoding (check the console), or the
  machine may be out of memory (turn VAE tiling on) — and when the server
  itself stops answering it says that instead of implying progress. The engine
  also checks the cancel flag at every stage boundary and after the pipeline
  returns, so "stop" lands at the next boundary instead of after a whole run
  nobody wants, and a stopped run is recorded as cancelled rather than failed.
- **Decoding no longer runs on the CPU while the GPU sits idle.** INT4's
  small-card recipe pinned the VAE to the CPU for both directions, so a 1024px
  run finished its sampler in 30 s and then sat in "decoding" for tens of
  minutes — the picture never arrived. Encode still stays on the CPU (it runs
  *before* sampling, while the transformer owns the GPU) but decode now borrows
  the accelerator, because sampling is over by then and the VRAM is free. A
  decode that still runs out of VRAM latches back to the CPU for the rest of
  the process, so a run always finishes; the settled device is reported in the
  job's runtime info, and the console prints the decode's start, finish and
  elapsed time — including the VRAM fallback.
- **The studio re-reads a job's real state instead of trusting the socket.**
  Losing a `job_complete` frame (reconnect, restarted server, a sleeping
  laptop) used to leave the progress bar sweeping forever with a finished
  picture nowhere on screen. The studio now polls the job while it runs,
  re-syncs on every reconnect, and re-attaches to a running job after a reload
  or in a second tab — the row, not the event stream, decides when a job is
  done.
- **The VAE decode no longer looks like a hang.** On small-GPU INT4 runs the
  VAE decodes on the CPU (group offload pins it there to protect VRAM), which
  takes minutes at 1024px — and nothing said so: the sampler's last step froze
  the studio on "100%, ~0s left" while the decode ran silently, and the
  "VAE 解码" label shown during sampling was only a `pct > 0.92` guess. The
  engine now wraps the VAE's decode and announces a real `generate_phase`
  event the moment it is entered (flagging the slow CPU case), restoring the
  exact callable it borrowed — including INT4's own instance-level CPU-decode
  wrapper. The studio answers with a sweeping bar, a live decode timer and, on
  CPU-decode runs, a "can take minutes" note; sampling steps always read
  采样中, the fake label is gone.
- **Restarting mid-job no longer leaves phantom "running" rows.** A job lives
  inside one server process, so closing the console window during a slow
  decode left the row claiming a running job forever — which misread every
  later diagnosis. Startup flips leftover running rows to failed with a
  readable reason before the studio can read them.
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
