# Architecture artifact

`index.html` is the source of the **StreetTracker Architecture** page, published as a private
claude.ai artifact: <https://claude.ai/artifact/KKtkUTiBnJhmNfNCxi86Ry>. It covers the
hardware, the live Orin pipeline, offline enrichment, the training loop, storage and the four
web apps. Every component has a plate-style discussion ID (`HW-1`, `RT-6`, `EN-5` …), and
`#<id>` links (e.g. `#rt-snap`) open its pop-up.

Figures are as of **2026-10-05** (data 2026-05-24 → 2026-10-04). Update them when a component
changes.

## Files

| Path | What |
| --- | --- |
| `index.html` | Artifact source. No `<!doctype>`/`<head>`: the publisher adds the skeleton. |
| `assets/site_*.webp` | Redacted screenshots of the showcase, control panel and Orin dashboard (`*_th` = card thumbnails). `site_training.webp` is the July capture from `docs/assets/`. |
| `assets/scene_overlay.jpg`, `track_snaps.jpg`, `sample_hq.jpg`, `plate_masked.png` | Pipeline samples from track 2676 of `session_20261001_190826`. |
| `assets/hw_*.jpg` | Hardware photos (credits below). |

## Republish

From Claude Code, publish `docs/architecture/index.html` to the existing URL with every
`assets/*` file in `files` (root `docs/architecture`). Read the artifact first if this session
hasn't published it.

## Refresh the screenshots and samples

Scripts live in `.claude/` and write raw captures to `.claude/arch_work/` (gitignored, because
raw captures still show plates):

1. `uv run --no-sync python .claude/arch_capture.py [page ...]`: headless Edge captures with
   plate strings and operator tags covered by yellow bars. The plate and tag lists are built in
   memory from the showcase API and `output/showcase_metadata.json`, and never written to disk.
   Capture the control dashboard once only: each load fires a pull-estimate SSH call.
2. `uv run --no-sync python .claude/arch_redact.py [page ...]`: pixelates plates inside photos
   (detector union at several scales) and exports WebP files into `assets/`.
3. **Look at every redacted image at full resolution** and put boxes for missed plates in
   `.claude/arch_work/manual_boxes.json` (CSS px per page), then re-run step 2. The detector
   misses motion-blurred and oblique front plates; the 2026-10-05 capture needed 12 manual boxes.
4. `uv run --no-sync python .claude/arch_samples.py`: regenerates the pipeline samples.
5. `uv run --no-sync python .claude/arch_check.py`: checks widths, themes, overflow, console
   errors and pop-ups before republishing.

## Credits

- RLC-1224A product photos © Reolink.
- Jetson Orin Nano Developer Kit photos © NVIDIA.
- Alienware Aurora R13 photo © Dell.
- M.2 NVMe drive by User5515, CC0, via Wikimedia Commons (representative; the Orin's drive is a
  Lexar 1 TB).
- Jetson Nano Developer Kit by SparkFun Electronics, CC BY 2.0, via Wikimedia Commons.
