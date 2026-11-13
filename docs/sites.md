# The showcase and the control panel

How the two dev-box web apps work: the showcase (`streettracker showcase`, port 8090) and the
operator control panel (`streettracker control`, port 8095). [`CLAUDE.md`](../CLAUDE.md) keeps
the rules that matter in every session; this is the full reference.

Moved from `CLAUDE.md` on 2026-11-11 with the text unchanged, so the control-panel section
still reads as a build log ("Phase 1 shipped", "Phase 2 in progress"). Both phases are
complete. Both apps now start at logon as the scheduled tasks `StreetTrackerShowcase` and
`StreetTrackerControlPanel`.

## Showcase website

`streettracker showcase` serves a local website (`src/streettracker/web/`:
`aggregate.py` + `metadata.py` + `stats.py` + `server.py` + jinja2
`templates/`) that browses the _enriched_ cars across every session in an
output root — plate-read (ANPR) cars joined to DVSA make/model/year/colour —
with the regularly-appearing ones featured, plus a per-car metadata editor
("this is my car", "this is Shaun's car"), and a traffic-statistics page.

```bash
uv run streettracker showcase --output-root output   # http://127.0.0.1:8090/
```

- **Identified cars only.** `web/aggregate.build_showcase` pools the plated
  subset from `analysis.vehicles.build_vehicles` (rebuilt **live** — the
  on-disk `*_vehicles.json` can be stale, e.g. written before `dvsa-label`)
  into one card per physical car, keyed by canonical plate and merged across
  sessions with the same fuzzy clustering the `vehicles --across` cohort uses.
  It also surfaces the richer DVSA fields (`primary_colour`/`fuel_type`/
  `engine_size_cc`) read straight from `<session>_dvsa_labels.json`.
  "Regulars" = the `different-day` kind (seen on ≥2 calendar dates).
- **User metadata** (name / owner / notes / favourite) persists atomically to
  `<output-root>/showcase_metadata.json`, keyed by plate so a tag follows a
  car across every session. This is the _only_ state the site writes;
  everything else is read-only upstream data.
- **Local-only by default** (binds `127.0.0.1`; the page shows plate data +
  personal tags). `--host 0.0.0.0` opts into LAN exposure. `POST /api/refresh`
  re-aggregates in place after new sessions are pulled. Images are served from
  the output root through a `.jpg`-only, single-segment-validated route.
- **Statistics page** (`/stats`, `web/stats.build_stats`) — each session's
  contribution is computed once and kept until one of that session's files
  changes (`_tally_session`, `common/filememo.py`), so a refresh re-reads only
  the sessions that changed; the first build after a start reads them all.
  Traffic analytics
  over **all** vehicle tracks (`class_name=="car"`, not just the plated subset;
  bucketed by camera-local `time_start`, so a multi-date session splits
  correctly): daily L→R/R→L journeys (click a day → its 15-min profile),
  day-of-week histogram, weekday×hour heatmap, speed distribution + fastest
  cars (pictured by the tracker's HQ crop of the car, then the tile, then a
  4K snap: a fast car has often left the frame before its 4K snap lands),
  and make/colour mix. A **People** section aggregates person tracks
  (footfall, dwell, heatmap) and — where sessions carry `_people.json` —
  rolls up walker/jogger/cyclist/dog-walk counts (cyclists classified since
  2026-06-13, dog walks since 2026-07-07; earlier jogger counts include
  unfiltered riders, ~+4pp). Charts are dependency-free SVG/CSS. One car track
  ≈ one pass (BotSORT can split). **Speed**: `speed_px_s` is inference-frame
  pixels, shown as px/s unless a calibration is set, then mph. Calibrate via
  `configs/showcase.json` (`{"m_per_px": X}` or `{"road_length_m": D}` →
  `m_per_px = D/801`, the traced road's travel-axis pixel length) or the
  `--m-per-px` / `--road-length-m` flags; see `configs/showcase.example.json`.
  mph is approximate (single global factor, ignores perspective).

## Operator control panel

`streettracker control` serves a standalone web app (`src/streettracker/control/`:
`server.py` + `orin.py` + `introspect.py` + jinja2 `templates/`) on the dev box
that lets the **operator drive the day-to-day procedures without Claude attached
— zero tokens once running**. It is a sibling to the showcase site (separate
port, separate concern), built on the same aiohttp + jinja2 + dependency-free
SVG stack.

```bash
uv run streettracker control --output-root output   # http://localhost:8095/
pwsh -NoProfile -File scripts/control_panel.ps1      # detached launcher (opens browser)
```

- **Hosting = on-demand.** `scripts/control_panel.ps1` starts the panel detached
  (outlives the launching terminal, runs outside any Claude helper tree). Ends at
  logout/reboot or when you close it.
- **Access split.** Binds `0.0.0.0` so dashboards are LAN-viewable (phone / wall
  screen), but every control action is **localhost-only**, enforced server-side
  (`server._require_local`) — a LAN viewer can watch but can't restart the camera
  service or promote a model. `--host 127.0.0.1` to keep it fully local.
- Background pollers refresh the Orin (~45 s, non-fatal/timeout-bounded) and
  rescan local `output/`+`runs/` (~30 s); model metadata is read sidecar-first
  (`makemodel_b0.meta.json`, written at promotion) then a one-off torch fallback.
  Introspection reads are mtime-cached so a closed session is parsed once.

**Phase 1 — decision radiator (shipped, read-only).** Orin live-session status
(service state, live session, runtime, images banked, on-demand pull size/ETA via
`remote_inventory`), local session inventory + enrichment badges (ALPR / DVSA /
vehicles / make-model), corpus stats (`runs/uk_crops_*/manifest.json`),
production-model metadata, training-run make@1 charts, and **retrain / promote
recommendations with the evidence numbers** (`introspect.retrain_recommendation`
/ `promote_recommendation`; defaults: retrain if +15 % cars / +3 makes / +5k
crops vs the model's recorded training corpus, promote if a run beats production
make@1 by ≥0.5 pp). Reuses `cli.pull` SSH/inventory helpers (made non-fatal +
timeout-bounded in `control/orin.py`), `common.output`, and
`analysis.snap_assets`. Tests: `tests/test_control/`.

**Phase 2 — in progress (process control).** Shipped + tested:

- `control/progress.py` — pure per-command **stdout progress parsers**
  (`alpr`/`makemodel` `N/total`; `makemodel-train-uk` per-epoch loss/make@1 rows
  for the live curve; pull/dvsa/build counts + summaries) + `estimate_eta`. No
  clock/IO, fully unit-tested.
- `control/jobs.py` — a **subprocess job runner**: streams stdout into the
  parser + a 500-line ring buffer, classifies the exit, holds an in-process
  `SetThreadExecutionState` **wake-lock** while busy, and cancels via Windows
  `taskkill /T`. **Per-lane execution**: a serial **GPU lane** (alpr / makemodel
  / build / train / export-engine) + a serial **net lane** (pull / dvsa /
  vehicles) run concurrently, so a pull overlaps a train but two GPU jobs never
  do (`JobSpec.lane` overrides). In-memory (no on-disk history yet).
- `control/prompts.py` — **`build_prompt`** turns a finished job into a
  complete, self-contained Claude Code prompt (project, command, cwd, exit code,
  captured output tail, command-specific pointer); `summarize_issue` decides
  _whether_ to surface one (non-zero exit, or a traceback / OOM / config-error /
  … marker even on a clean exit).
- server `/api/jobs` (GET list, POST submit [localhost + kind-allowlisted],
  GET `{id}`, POST `{id}/cancel` [localhost]) + a **Processes** dashboard panel:
  a localhost run-bar, live job cards with progress bars/ETA, a **live training
  chart** (loss + make@1 per epoch + pre-training facts, rendered from the parsed
  `epochs` metrics for `makemodel-train-uk` jobs), and — when a job fails or
  looks off — a **"Copy Claude prompt"** button so the operator gets a
  paste-ready help request with zero hand-writing. The localhost guard +
  `POST /api/refresh` from Phase 1 pre-staged this access model.
- **Pull byte-ETA.** The runner forces unbuffered child stdout
  (`PYTHONUNBUFFERED`) so lines stream live, and a per-job **dir-watcher**
  samples the local session dir as it fills toward the remote byte total — a
  real transfer progress bar + ETA, not a spinner. `pull` prints a
  machine-readable `size_bytes` (the main snaps + vehicle HQ crops under
  `--only-main`, via `RemoteInventory.main_bytes` + `vehicle_hq_bytes`, so the
  denominator matches what's copied);
  `PullParser` turns the session/target/size_bytes lines into a generic
  `watch` directive the runner consumes.
- `control/playbooks.py` — a **multi-step playbook engine** (`PlaybookRunner`):
  ordered steps run sequentially, stopping on the first failure (rest →
  _skipped_); each step is a **job** (run on the shared `JobRunner`, so it
  inherits lanes + live progress + the copy-prompt) or an in-process **action**
  (for SSH/file steps). Playbooks (`PLAYBOOKS` registry; `build_playbook(name,
ctx, …)` dispatches, `PlaybookContext` carries paths + device config):
  - **enrich** (`alpr-run` → `dvsa-label` → `dvsa-apply` → `vehicles` →
    `makemodel` → `bodytype` → `people`) and **build-train**
    (`makemodel-build-uk` → `makemodel-train-uk` → `makemodel-compare`,
    dated dirs; the last step writes the head-to-head `compare.json` the
    promote recommendation needs) — pure job-chains. build-train takes an
    optional per-car crop cap (`max_per_car`, the dashboard's "max
    crops/car" field → `makemodel-build-uk --max-per-car`, recorded in the
    corpus manifest); a capped run writes `uk_crops_MMDD_576_capN` /
    `uk_make_MMDD_b6_capN`, so it never overwrites the same day's uncapped
    build.
  - **roll** (action: `orin.restart_service` finalises the live session + starts
    a new one, verifies the handover + counts finalised tracks → then a `pull`
    job of the closed session) and **promote** (action: back up `makemodel_b0.pt`
    → copy the best run's `best.pt` in → write the metadata sidecar + corpus
    fingerprint → then a `model-release` job publishes the model as the
    **public** GitHub Release `makemodel-<run>`, its only off-box copy since it
    is too big to commit; since 2026-10-26) — **destructive**, so the submit route refuses them without
    `confirm=true` and the UI shows a confirm dialog. `roll` resolves the live
    session over SSH at submit and refuses if the device is down/idle.
  - **reinfer** (one `makemodel` job per local session → an action that POSTs
    the showcase's `/api/refresh`) — re-runs the classifier everywhere + updates
    the showcase.
  - **rescore** (`alpr-rescore` → `dvsa-label` → `dvsa-apply` → `vehicles` on
    every session whose ALPR stamp lacks `"ocr_conf": "min_char"` → showcase
    refresh) — applies the 2026-09-28 OCR-confidence fix to already-enriched
    sessions. The sessions table badges them **Plates v2** (amber) until done,
    **Plates v3** after.
  - **platecheck** (`alpr-colour` on every session with ALPR output first --
    the combined gate counts plate support across all rollups -- then
    `dvsa-label` → `dvsa-apply` → `vehicles` per session → showcase refresh)
    -- applies the 2026-10-04 plate-colour check and the current plate gate.
    Sessions badge **Plates v3** (amber) until colour-checked, **Plates v4**
    (green) after. `alpr-colour` shares the GPU lane so it never overlaps an
    `alpr-run` writing the same `_alpr.json`. Plates newly passing the gate
    cost DVSA lookups.
  - `/api/playbooks` routes (localhost submit/cancel; name + session validated,
    destructive-confirm gated) + a **Playbooks** dashboard panel: step list with
    live status, the running/failed step's job **inlined** (progress bar /
    training chart / copy-prompt), and Cancel. (Fixed a cancel race: a cancel
    landing during a job's spawn window now reaps the process once it exists.)
- `control/history.py` — **on-disk job/playbook history**: each finished item's
  terminal snapshot (incl. the failed-job copy-prompt) is appended to a capped,
  atomically-written JSON list under the output root (`control_jobs.json`,
  `control_playbooks.json`), and the runners' `snapshots()` merge prior-run
  history with live items — so the panel's recent activity (and prompts) survive
  a restart.
- **`/training` full-page view** (`templates/training.html`) — picks the current
  `makemodel-train-uk` job and shows pre-training facts (backbone / input /
  makes / train+val crops), a large live **loss + make@1 curve** with the
  production make@1 as a dashed reference line, a per-epoch table, the training
  corpus's per-make distribution (brand logos), and — once finished — a
  candidate-vs-production verdict + a one-click **Promote** (localhost,
  confirm-gated, run derived from the checkpoint path). Top-bar nav switches
  Dashboard ↔ Training. The shared brand-mark + `trainChart` helpers live in
  `base.html` so both pages use them.

**Phase 2/3 — complete** for the planned scope. The panel covers the full
operator loop tokenlessly: radiator → run individual processes or one-click
playbooks (enrich / build-train / roll / promote / reinfer) with live
progress + ETAs + a copy-paste help prompt on failure, plus a dedicated live
training page. Possible future polish (not required): per-session enrich/pull
shortcut buttons on the sessions table, and surfacing `control_*.json` in
`.gitignore`.
