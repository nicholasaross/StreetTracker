"""The E1.2 plate-read audit set: draw it, label it, score it.

Every plate-read rate so far counts a car as read when a snap gave a UK-shaped
string; nobody checked the characters, or that the plate was on the tracked
car (review findings R1-R3, docs/data_integrity_review.md). This draws a
stratified sample of reads (streettracker.analysis.alpr.audit), serves a
page to check each against the 4K snap, and turns the answers into rates for
every read (streettracker.analysis.alpr.audit_score).

Run from the repo root:

  uv run python .claude/audit_e12.py select --lat 51.5 --lon -0.1
      Draw the set (once) and render each item's pictures. --lat/--lon place
      the camera to the nearest 0.1 degree, for the sun's height (day / dusk /
      dark); they are kept only in the gitignored manifest.
  uv run python .claude/audit_e12.py serve
      Label on http://127.0.0.1:8093/. Every answer is saved as it is given;
      reloading resumes at the first unlabelled item.
  uv run python .claude/audit_e12.py report
      The rates the labels so far support, with 95 % intervals -> report.json.
  uv run python .claude/audit_e12.py rescore
      After re-running enrich (or changing the pipeline): how the audited
      tracks' best reads moved against the labelled truth.

The labelling order is random, so any labelled part of the set is a random
part of every stratum: `report` is meaningful long before the end.

Everything lives in .claude/audit_e12/ (gitignored: it holds plates). The
labels are about three hours of operator work and the set is meant to stay
frozen as a regression test, so back the folder up with output/.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from PIL import Image, ImageDraw

from streettracker.analysis.alpr.audit import (
    ATTRIBUTIONS,
    SampleSizes,
    clean_label,
    draw_audit_set,
    kept_best_reads,
    session_pool,
    skip_reason,
)
from streettracker.analysis.alpr.audit_score import build_report, format_report, rescore
from streettracker.analysis.alpr.fullframe import load_road_polygon
from streettracker.analysis.alpr.gate import load_plate_support, resolve_plate_gate
from streettracker.analysis.snap_assets import image_size
from streettracker.common.fileio import atomic_write_json, read_json
from streettracker.common.session_files import SessionFiles

AUDIT_DIR = Path(__file__).resolve().parent / "audit_e12"
IMG_DIR = AUDIT_DIR / "img"
MANIFEST_PATH = AUDIT_DIR / "manifest.json"
LABELS_PATH = AUDIT_DIR / "labels.json"
REPORT_PATH = AUDIT_DIR / "report.json"
MANIFEST_VERSION = 1

# Box colours, the same on every picture and in the page's legend.
COLOURS = {
    "hint": (0, 200, 255),  # where alpr-run looked for the car
    "done_box": (255, 0, 220),  # where the tracker had it when the snap landed
    "vehicle_box": (60, 230, 60),  # the vehicle the plate crop came from
    "det_bbox": (255, 150, 0),  # the plate that was read
}
CONTEXT_MAX = (1400, 820)
PLATE_TARGET_W = 560


# ----------------------------------------------------------------------
# select


def _sessions(root: Path, only: list[str] | None) -> list[Path]:
    dirs = sorted(d for d in root.glob("session_*") if d.is_dir())
    if only:
        wanted = set(only)
        dirs = [d for d in dirs if d.name in wanted]
        missing = wanted - {d.name for d in dirs}
        if missing:
            sys.exit(f"no such session(s) under {root}: {sorted(missing)}")
    return dirs


def _usable(root: Path):
    def usable(c: dict) -> bool:
        return image_size(root / c["session"] / c["image"]) is not None

    return usable


def _draw_boxes(img: Image.Image, item: dict, offset: tuple[int, int], width: int) -> None:
    draw = ImageDraw.Draw(img)
    ox, oy = offset
    for name in ("hint", "done_box", "vehicle_box", "det_bbox"):
        box = item.get(name)
        if box:
            x1, y1, x2, y2 = box
            draw.rectangle([x1 - ox, y1 - oy, x2 - ox, y2 - oy], outline=COLOURS[name], width=width)


def _fit(img: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    scale = min(max_size[0] / img.width, max_size[1] / img.height, 1.0)
    if scale >= 1.0:
        return img
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)


def _context_region(item: dict, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Every drawn box plus generous margin, so oncoming and parked cars
    around the plate are in view."""
    boxes = [item[k] for k in ("hint", "done_box", "vehicle_box", "det_bbox") if item.get(k)]
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    pad_x = max(0.6 * (x2 - x1), 450)
    pad_y = max(0.6 * (y2 - y1), 260)
    w, h = size
    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(w, int(x2 + pad_x)),
        min(h, int(y2 + pad_y)),
    )


def _render(item: dict, snap: Path) -> tuple[str, str]:
    """The item's two pictures: the scene around the plate with the boxes
    drawn, and the plate itself enlarged, with nothing drawn on it."""
    with Image.open(snap) as opened:
        im = opened.convert("RGB")
    px1, py1, px2, py2 = item["det_bbox"]
    pw, ph = px2 - px1, py2 - py1
    plate = im.crop(
        (
            max(0, int(px1 - 0.4 * pw)),
            max(0, int(py1 - 0.8 * ph)),
            min(im.width, int(px2 + 0.4 * pw)),
            min(im.height, int(py2 + 0.8 * ph)),
        )
    )
    scale = max(1.0, min(6.0, PLATE_TARGET_W / max(1, plate.width)))
    plate = plate.resize((round(plate.width * scale), round(plate.height * scale)), Image.LANCZOS)

    region = _context_region(item, im.size)
    context = im.crop(region)
    _draw_boxes(context, item, region[:2], width=max(3, (region[2] - region[0]) // 350))
    context = _fit(context, CONTEXT_MAX)

    stem = f"{item['n']:04d}"
    context.save(IMG_DIR / f"{stem}_ctx.jpg", quality=88)
    plate.save(IMG_DIR / f"{stem}_plate.jpg", quality=92)
    return f"{stem}_ctx.jpg", f"{stem}_plate.jpg"


def _guard_existing(force: bool) -> None:
    """Never overwrite labelled work: with --force the old labels are kept
    beside the new set under a dated name."""
    labels = read_json(LABELS_PATH, {})
    if not labels:
        return
    if not force:
        sys.exit(
            f"{LABELS_PATH} already holds {len(labels)} label(s). The set is meant to stay "
            f"frozen; pass --force to draw a new one (the old labels and manifest are kept "
            f"under a dated name)."
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    LABELS_PATH.rename(LABELS_PATH.with_name(f"labels.{stamp}.json"))
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.rename(MANIFEST_PATH.with_name(f"manifest.{stamp}.json"))
    print(f"kept the previous set as labels.{stamp}.json / manifest.{stamp}.json")


def select(args: argparse.Namespace) -> None:  # noqa: C901 - one linear procedure
    _guard_existing(args.force)
    root = Path(args.output_root)
    gate, gate_source = resolve_plate_gate()
    print(f"plate gate: {gate.describe()} ({gate_source})")
    support = load_plate_support(root)
    polygon = load_road_polygon(args.road_polygon, log_prefix="[audit]")

    pools, skipped = [], {}
    for sd in _sessions(root, args.sessions):
        reason = skip_reason(sd)
        if reason is None:
            pool = session_pool(
                sd,
                gate=gate,
                support=support,
                road_polygon_frac=polygon,
                latitude=args.lat,
                longitude=args.lon,
            )
            reason = None if pool is not None else "no 4K snap can be opened"
        if reason is not None:
            skipped[sd.name] = reason
            print(f"  {sd.name}: skipped ({reason})")
            continue
        pools.append(pool)
        print(
            f"  {sd.name}: {sum(pool.cars.values()):6d} cars, {len(pool.best):5d} best reads, "
            f"{len(pool.colour_rejected):4d} colour / {len(pool.static_rejected):4d} static "
            f"rejected, {len(pool.dvsa):4d} DVSA-labelled tracks"
        )
    if not pools:
        sys.exit("no session can be sampled")

    sizes = SampleSizes(args.n_best, args.n_colour, args.n_static, args.n_dvsa)
    drawn = draw_audit_set(pools, sizes, random.Random(args.seed), usable=_usable(root))

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for old in IMG_DIR.glob("*.jpg"):
        old.unlink()
    items = []
    for n, item in enumerate(drawn.items, 1):
        item = {**item, "n": n}
        item["ctx_img"], item["plate_img"] = _render(item, root / item["session"] / item["image"])
        items.append(item)
        if n % 25 == 0 or n == len(drawn.items):
            print(f"  rendered {n}/{len(drawn.items)}")

    manifest = {
        "version": MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output_root": str(root),
        "seed": args.seed,
        "location": {"lat": round(args.lat, 1), "lon": round(args.lon, 1)},
        "gate": gate.to_json(),
        "sizes": {
            "best": sizes.best,
            "colour_rejected": sizes.colour_rejected,
            "static_rejected": sizes.static_rejected,
            "dvsa": sizes.dvsa,
        },
        "sessions": [p.session for p in pools],
        "skipped": skipped,
        "strata": drawn.strata,
        "domains": drawn.domains,
        "items": items,
    }
    atomic_write_json(MANIFEST_PATH, manifest, indent=1)
    _print_strata(manifest)
    print(
        f"\n{len(items)} items -> {MANIFEST_PATH}\nnext: uv run python .claude/audit_e12.py serve"
    )


def _print_strata(manifest: dict) -> None:
    print(f"\n{'stratum':<46} {'reads':>8} {'drawn':>6}")
    for stratum, info in manifest["strata"].items():
        print(f"{stratum:<46} {info['population']:>8} {info['sampled']:>6}")


# ----------------------------------------------------------------------
# serve

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Plate audit</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root { color-scheme: dark; --bg:#14161a; --panel:#1d2026; --line:#2c313a; --text:#dfe3e8;
          --dim:#98a1ac; --accent:#4caf7d; --warn:#e8b75d; --bad:#e35d5d; }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.45 system-ui,sans-serif; background:var(--bg); color:var(--text); }
  header { display:flex; gap:14px; align-items:center; padding:9px 16px; background:var(--panel);
           position:sticky; top:0; z-index:3; flex-wrap:wrap; border-bottom:1px solid var(--line); }
  #bar { flex:1 1 160px; height:8px; background:var(--line); border-radius:4px; overflow:hidden; }
  #bar div { height:100%; background:var(--accent); width:0; transition:width .15s; }
  header button, .btn { background:var(--panel); color:var(--text); border:1px solid var(--line);
           border-radius:6px; padding:6px 11px; cursor:pointer; font:inherit; }
  header button:hover, .btn:hover { border-color:#4a5260; }
  main { display:grid; grid-template-columns:minmax(0,1fr) 380px; gap:14px; padding:14px; }
  @media (max-width:1100px) { main { grid-template-columns:1fr; } }
  #ctxwrap { overflow:hidden; border:1px solid var(--line); border-radius:6px; background:#000;
             cursor:zoom-in; }
  #ctx { display:block; width:100%; height:auto; transition:transform .12s; }
  #plate { display:block; max-width:100%; border:1px solid var(--line); border-radius:6px;
           background:#000; image-rendering:auto; }
  #legend { color:var(--dim); font-size:12px; margin-top:6px; }
  #legend b { font-weight:600; }
  aside { display:flex; flex-direction:column; gap:12px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }
  .card h3 { margin:0 0 8px; font-size:13px; color:var(--dim); font-weight:600;
             text-transform:uppercase; letter-spacing:.04em; }
  #meta div { display:flex; justify-content:space-between; gap:8px; }
  #meta span:first-child { color:var(--dim); }
  .opt { display:block; width:100%; text-align:left; margin:4px 0; }
  .opt kbd, .key { display:inline-block; min-width:20px; text-align:center; background:var(--line);
           border-radius:4px; padding:0 5px; margin-right:8px; font:600 12px ui-monospace,monospace; }
  .opt.sel { outline:2px solid var(--accent); }
  .step { display:none; } .step.on { display:block; }
  input[type=text] { width:100%; padding:9px 10px; font:600 22px ui-monospace,monospace;
           letter-spacing:.12em; text-transform:uppercase; background:#0e1013; color:var(--text);
           border:1px solid var(--line); border-radius:6px; }
  #note { font:13px system-ui,sans-serif; letter-spacing:0; text-transform:none; }
  .hint { color:var(--dim); font-size:12px; margin-top:6px; }
  .err { color:var(--bad); font-size:13px; min-height:1em; margin-top:6px; }
  .reads { font:600 20px ui-monospace,monospace; letter-spacing:.1em; margin:4px 0 10px; }
  .row { display:flex; gap:6px; align-items:center; margin:6px 0; padding:4px; border-radius:6px; }
  .row.on { background:#262b33; }
  .row .what { flex:1; }
  .row button.sel { outline:2px solid var(--accent); }
  #done { color:var(--accent); }
  .tag { display:inline-block; padding:1px 7px; border-radius:10px; background:var(--line);
         font-size:12px; margin-right:4px; }
  a { color:#7fb4e8; }
</style></head><body>
<header>
  <strong>Plate audit</strong>
  <span id="pos">-</span>
  <div id="bar"><div></div></div>
  <span id="prog">-</span>
  <button onclick="go(-1)" title="Left arrow">&larr; prev</button>
  <button onclick="go(1)" title="Right arrow">next &rarr;</button>
  <button onclick="nextUnlabelled()" title="key N">next unlabelled</button>
</header>
<main>
  <section>
    <div id="ctxwrap"><img id="ctx" alt="The scene around the plate"></div>
    <div id="legend">
      <b style="color:#ff00dc">magenta</b> where the tracker had the car when the snap landed (when recorded) &middot;
      <b style="color:#00c8ff">cyan</b> where plate reading looked for it &middot;
      <b style="color:#3ce63c">green</b> the vehicle the plate crop came from &middot;
      <b style="color:#ff9600">orange</b> the plate &middot; click to zoom &middot;
      <a id="full" target="_blank">whole frame</a>
    </div>
  </section>
  <aside>
    <div class="card"><h3>The plate</h3><img id="plate" alt="The plate, enlarged"></div>
    <div class="card"><h3>The track</h3><div id="meta"></div></div>
    <div class="card" id="work">
      <div class="step" id="s-attr"><h3>1 &middot; Whose plate is in the orange box?</h3><div id="attrs"></div></div>
      <div class="step" id="s-text"><h3>2 &middot; Type the plate</h3>
        <input type="text" id="text" maxlength="10" autocomplete="off" spellcheck="false">
        <div class="hint">Enter to save. <span class="key">.</span> for a character you can't read;
          <span class="key">?</span> alone if none of it is legible.</div>
        <div class="err" id="texterr"></div></div>
      <div class="step" id="s-rec"><h3>2b &middot; The pipeline read it differently</h3>
        <div class="hint">You typed</div><div class="reads" id="mine"></div>
        <div class="hint">The pipeline read</div><div class="reads" id="theirs"></div>
        <button class="btn opt" onclick="reconcile('kept')"><kbd>K</kbd>Keep mine</button>
        <button class="btn opt" onclick="reconcile('ocr_right')"><kbd>O</kbd>The pipeline is right</button>
        <button class="btn opt" onclick="editText()"><kbd>E</kbd>Edit mine</button></div>
      <div class="step" id="s-dvsa"><h3>3 &middot; Does the register fit the tracked car?</h3>
        <div id="dvsarows"></div>
        <div class="hint"><span class="key">Y</span> yes <span class="key">X</span> no
          <span class="key">U</span> can't tell, for the highlighted row.</div></div>
      <div class="step" id="s-done"><h3>Saved</h3><div id="done"></div>
        <div id="answer"></div>
        <button class="btn" style="margin-top:10px" onclick="redo()"><span class="key">R</span>Redo this item</button></div>
      <input type="text" id="note" placeholder="Note (optional, saved with the answer)" style="margin-top:10px">
    </div>
  </aside>
</main>
<script>
const ATTR = __ATTRIBUTIONS__;
const FIELDS = [["make","Make and model"],["colour","Colour"],["body","Body type"]];
let S = {items:[], labels:{}, cur:0}, D = null, zoomed = false;

const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const norm = s => (s || "").toUpperCase().replace(/[^A-Z0-9.]/g, "");
const glyph = s => s.replace(/O/g, "0").replace(/I/g, "1");
function sameText(read, truth) {
  const a = glyph(norm(read)), b = glyph(norm(truth));
  return a.length === b.length && [...b].every((c, i) => c === "." || c === a[i]);
}
const item = () => S.items[S.cur];
const isDvsa = it => it.sample === "dvsa";

async function init() {
  const s = await (await fetch("/api/state")).json();
  S.items = s.items; S.labels = s.labels;
  const box = $("attrs");
  Object.entries(ATTR).forEach(([k, txt], i) => {
    const b = document.createElement("button");
    b.className = "btn opt"; b.id = "a-" + k;
    b.innerHTML = "<kbd>" + (i + 1) + "</kbd>" + esc(txt);
    b.onclick = () => chooseAttr(k);
    box.appendChild(b);
  });
  $("ctxwrap").onclick = ev => {
    const img = $("ctx");
    if (zoomed) { img.style.transform = ""; zoomed = false; return; }
    const r = img.getBoundingClientRect();
    img.style.transformOrigin = ((ev.clientX - r.left) / r.width * 100) + "% " + ((ev.clientY - r.top) / r.height * 100) + "%";
    img.style.transform = "scale(2.6)"; zoomed = true;
  };
  $("text").addEventListener("keydown", ev => { if (ev.key === "Enter") { ev.preventDefault(); submitText(); } });
  S.cur = firstUnlabelled();
  show();
}
function firstUnlabelled() {
  const i = S.items.findIndex(it => !(it.key in S.labels));
  return i < 0 ? 0 : i;
}
function nextUnlabelled() {
  for (let k = 1; k <= S.items.length; k++) {
    const j = (S.cur + k) % S.items.length;
    if (!(S.items[j].key in S.labels)) { S.cur = j; show(); return; }
  }
  show();
}
function go(d) { S.cur = (S.cur + d + S.items.length) % S.items.length; show(); }

function step(name) {
  document.querySelectorAll(".step").forEach(e => e.classList.toggle("on", e.id === "s-" + name));
  D && (D.step = name);
  if (name === "text") setTimeout(() => $("text").focus(), 0);
}

function show() {
  const it = item(); zoomed = false;
  $("ctx").style.transform = "";
  // Hide both pictures until the new ones have loaded: otherwise the last
  // item's car stays on screen and a quick answer goes against it.
  for (const [id, name] of [["ctx", it.ctx_img], ["plate", it.plate_img]]) {
    const img = $(id);
    img.style.visibility = "hidden";
    img.onload = () => { if (img.src.endsWith(encodeURIComponent(name))) img.style.visibility = "visible"; };
    img.src = "/img/" + encodeURIComponent(name);
  }
  $("full").href = "/full/" + it.n;
  const n = Object.keys(S.labels).length;
  $("pos").textContent = "item " + (S.cur + 1) + " of " + S.items.length;
  $("prog").textContent = n + " labelled";
  document.querySelector("#bar div").style.width = (100 * n / S.items.length) + "%";
  const arrow = it.direction === "right to left" ? "&larr; right to left" : "left to right &rarr;";
  const when = (it.time_start || "").replace("T", " ").slice(0, 19);
  $("meta").innerHTML =
    "<div><span>Direction</span><span>" + arrow + "</span></div>" +
    "<div><span>Time</span><span>" + esc(when) + "</span></div>" +
    "<div><span>Track</span><span>" + esc(it.session.replace("session_", "")) + " #" + it.track_id + " snap " + it.snap_index + "</span></div>";
  const saved = S.labels[it.key];
  $("note").value = saved ? (saved.note || "") : "";
  if (saved) { D = null; showSaved(it, saved); return; }
  D = {attribution: null, text: null, text_first: null, reconciled: null, dvsa: {}, row: 0};
  Object.keys(ATTR).forEach(k => $("a-" + k).classList.remove("sel"));
  $("text").value = ""; $("texterr").textContent = "";
  step("attr");
}

function showSaved(it, lab) {
  step("done");
  const tags = ["<span class='tag'>" + esc(it.sample.replace("_", " ")) + "</span>"];
  if (it.colour_suspect) tags.push("<span class='tag'>plate-colour flag</span>");
  if (it.static_suspect) tags.push("<span class='tag'>static-plate flag</span>");
  tags.push("<span class='tag'>" + esc(it.path) + "</span>");
  const right = lab.text == null ? "plate illegible" : (sameText(it.ocr_text, lab.text) ? "the pipeline read it right" : "the pipeline misread it");
  let html = "<div>" + tags.join("") + "</div>" +
    "<div style='margin-top:8px'>" + esc(ATTR[lab.attribution]) + "</div>" +
    (lab.attribution === "not_plate" ? "" :
      "<div class='hint'>You read</div><div class='reads'>" + esc(lab.text ?? "?") + "</div>" +
      "<div class='hint'>The pipeline read (conf " + Number(it.ocr_conf || 0).toFixed(2) + "): " + right + "</div>" +
      "<div class='reads'>" + esc(it.ocr_text) + "</div>");
  if (lab.dvsa) html += "<div class='hint'>Register: " + FIELDS.map(([f, t]) => esc(t) + " " + esc(lab.dvsa[f])).join(", ") + "</div>";
  $("done").textContent = "";
  $("answer").innerHTML = html;
}

function chooseAttr(k) {
  if (!D || D.step !== "attr") return;
  D.attribution = k;
  Object.keys(ATTR).forEach(a => $("a-" + a).classList.toggle("sel", a === k));
  if (k === "not_plate") { afterText(); return; }
  step("text");
}

function submitText() {
  const raw = $("text").value.trim();
  if (raw === "?") { D.text = null; D.text_first = D.text_first ?? null; afterText(); return; }
  const t = norm(raw);
  if (t.length < 2 || t.length > 8) { $("texterr").textContent = "A plate is 2 to 8 characters (or ? if illegible)."; return; }
  $("texterr").textContent = "";
  if (D.text_first == null && !D.edited) D.text_first = t;
  D.text = t;
  if (sameText(item().ocr_text, t)) { D.reconciled = null; afterText(); return; }
  $("mine").textContent = t; $("theirs").textContent = item().ocr_text;
  step("rec");
}
function reconcile(how) {
  if (!D || D.step !== "rec") return;
  D.reconciled = how;
  if (how === "ocr_right") D.text = norm(item().ocr_text);
  afterText();
}
function editText() { if (!D) return; D.edited = true; $("text").value = D.text || ""; step("text"); }

function afterText() {
  if (isDvsa(item())) { dvsaStep(); return; }
  finish();
}
function dvsaStep() {
  const d = item().dvsa || {};
  const what = {
    make: [d.make, d.model, d.year].filter(Boolean).join(" "),
    colour: d.primary_colour || "-",
    body: d.body_type || "(no body type for this model)",
  };
  $("dvsarows").innerHTML = FIELDS.map(([f, t], i) =>
    "<div class='row' id='r-" + f + "'><div class='what'><div class='hint'>" + esc(t) + "</div>" + esc(what[f]) + "</div>" +
    ["yes", "no", "unsure"].map(a => "<button class='btn' id='d-" + f + "-" + a + "' onclick=\"answer('" + f + "','" + a + "')\">" + a + "</button>").join("") +
    "</div>").join("");
  D.row = 0; markRow(); step("dvsa");
}
function markRow() { FIELDS.forEach(([f], i) => $("r-" + f).classList.toggle("on", i === D.row)); }
function answer(f, a) {
  if (!D || D.step !== "dvsa") return;
  D.dvsa[f] = a;
  ["yes", "no", "unsure"].forEach(x => $("d-" + f + "-" + x).classList.toggle("sel", x === a));
  const next = FIELDS.findIndex(([g]) => !(g in D.dvsa));
  if (next < 0) { finish(); return; }
  D.row = next; markRow();
}

async function finish() {
  const it = item();
  const body = {key: it.key, attribution: D.attribution, text: D.text, text_first: D.text_first,
                reconciled: D.reconciled, note: $("note").value, dvsa: isDvsa(it) ? D.dvsa : undefined};
  const r = await fetch("/api/label", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  if (!r.ok) { alert("Not saved: " + await r.text()); return; }
  S.labels[it.key] = await r.json();
  nextUnlabelled();
}
async function redo() {
  const it = item();
  const r = await fetch("/api/label", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({key: it.key, clear: true})});
  if (!r.ok) { alert("Not cleared: " + await r.text()); return; }
  delete S.labels[it.key];
  show();
}

document.addEventListener("keydown", ev => {
  if (ev.target.tagName === "INPUT") return;
  const k = ev.key.toLowerCase();
  if (ev.key === "ArrowLeft") return go(-1);
  if (ev.key === "ArrowRight") return go(1);
  if (k === "n" && (!D || D.step === "attr")) return nextUnlabelled();
  if (!D) { if (k === "r") redo(); return; }
  if (D.step === "attr" && /^[1-9]$/.test(k)) {
    const key = Object.keys(ATTR)[Number(k) - 1];
    if (key) chooseAttr(key);
  } else if (D.step === "rec") {
    if (k === "k") reconcile("kept"); else if (k === "o") reconcile("ocr_right"); else if (k === "e") editText();
  } else if (D.step === "dvsa") {
    const f = FIELDS[D.row][0];
    if (k === "y") answer(f, "yes"); else if (k === "x") answer(f, "no"); else if (k === "u") answer(f, "unsure");
  }
});
init();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    manifest: dict[str, Any] = {}
    labels: dict[str, Any] = {}
    by_key: dict[str, dict[str, Any]] = {}
    by_n: dict[int, dict[str, Any]] = {}
    images: set[str] = set()
    output_root: Path = Path("output")

    def log_message(self, *args: object) -> None:  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: object) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            page = PAGE.replace("__ATTRIBUTIONS__", json.dumps(ATTRIBUTIONS))
            self._send(200, page.encode(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._json(200, {"items": self.manifest["items"], "labels": self.labels})
        elif path.startswith("/img/"):
            name = unquote(path[len("/img/") :])
            if name not in self.images:
                self._send(404, b"not found", "text/plain")
                return
            self._send(200, (IMG_DIR / name).read_bytes(), "image/jpeg")
        elif path.startswith("/full/"):
            self._full(path[len("/full/") :])
        else:
            self._send(404, b"not found", "text/plain")

    def _full(self, n_text: str) -> None:
        """The whole 4K frame with the boxes drawn, rendered on request."""
        try:
            item = self.by_n[int(n_text)]
            with Image.open(self.output_root / item["session"] / item["image"]) as opened:
                im = opened.convert("RGB")
        except (KeyError, ValueError, OSError):
            self._send(404, b"not found", "text/plain")
            return
        _draw_boxes(im, item, (0, 0), width=max(4, im.width // 500))
        buf = io.BytesIO()
        _fit(im, (2000, 1200)).save(buf, format="JPEG", quality=85)
        self._send(200, buf.getvalue(), "image/jpeg")

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/label":
            self._send(404, b"not found", "text/plain")
            return
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            item = self.by_key[payload["key"]]
            if payload.get("clear"):
                self.labels.pop(item["key"], None)
                saved: dict[str, Any] | None = None
            else:
                saved = clean_label(payload, dvsa=item["sample"] == "dvsa")
                saved["labelled_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self.labels[item["key"]] = saved
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self._send(400, str(exc).encode(), "text/plain")
            return
        atomic_write_json(LABELS_PATH, self.labels, indent=1)
        self._json(200, saved)


def _load_manifest() -> dict[str, Any]:
    manifest = read_json(MANIFEST_PATH)
    if not isinstance(manifest, dict) or manifest.get("version") != MANIFEST_VERSION:
        sys.exit(f"no usable {MANIFEST_PATH}: run `select` first")
    return manifest


def serve(args: argparse.Namespace) -> None:
    manifest = _load_manifest()
    Handler.manifest = manifest
    Handler.labels = read_json(LABELS_PATH, {}) or {}
    Handler.by_key = {it["key"]: it for it in manifest["items"]}
    Handler.by_n = {int(it["n"]): it for it in manifest["items"]}
    Handler.images = {it[k] for it in manifest["items"] for k in ("ctx_img", "plate_img")}
    Handler.output_root = Path(args.output_root or manifest["output_root"])
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"audit set: {len(manifest['items'])} items, {len(Handler.labels)} labelled")
    print(f"serving on http://127.0.0.1:{args.port}/  (Ctrl-C to stop)")
    httpd.serve_forever()


# ----------------------------------------------------------------------
# report / rescore


def report(args: argparse.Namespace) -> None:
    manifest = _load_manifest()
    labels = read_json(LABELS_PATH, {}) or {}
    result = build_report(manifest, labels, n_boot=args.n_boot, seed=args.seed)
    print(format_report(result))
    atomic_write_json(
        REPORT_PATH,
        {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **result},
    )
    print(f"\n-> {REPORT_PATH}")


def rescore_cmd(args: argparse.Namespace) -> None:
    manifest = _load_manifest()
    labels = read_json(LABELS_PATH, {}) or {}
    root = Path(args.output_root or manifest["output_root"])
    current: dict[tuple[str, int], Any] = {}
    for session in sorted({it["session"] for it in manifest["items"]}):
        best = kept_best_reads(read_json(SessionFiles(root / session).alpr_by_track, {}))
        for it in manifest["items"]:
            if it["session"] == session:
                current[(session, int(it["track_id"]))] = best.get(int(it["track_id"]))
    result = rescore(manifest, labels, current)
    if not result:
        print("no labelled item has a full plate reading yet")
        return
    names = {
        "known": "Tracks whose own plate the labeller read in full",
        "misattributed": "Tracks whose audited read was another car's plate",
    }
    for kind, block in result.items():
        print(f"{names[kind]} ({block['n']})")
        print(f"  when drawn: {block['then']}")
        print(f"  now:        {block['now']}")
        for change in block["changed"]:
            print(f"    {change['key']}: {change['then']} -> {change['now']}")
    print(
        "\n'same' is the plate the labeller read: right for the first group, still the wrong "
        "car's plate for the second."
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    sel = sub.add_parser("select", help="draw the set and render its pictures")
    sel.add_argument("--output-root", default="output")
    sel.add_argument("--lat", type=float, required=True, help="camera latitude (0.1 deg is enough)")
    sel.add_argument("--lon", type=float, required=True, help="camera longitude, east positive")
    sel.add_argument("--sessions", nargs="*", help="only these session labels (default: all)")
    sel.add_argument("--road-polygon", type=Path, default=Path(".claude/triggers_proposal.json"))
    sel.add_argument("--seed", type=int, default=12)
    sel.add_argument("--n-best", type=int, default=SampleSizes().best)
    sel.add_argument("--n-colour", type=int, default=SampleSizes().colour_rejected)
    sel.add_argument("--n-static", type=int, default=SampleSizes().static_rejected)
    sel.add_argument("--n-dvsa", type=int, default=SampleSizes().dvsa)
    sel.add_argument("--force", action="store_true", help="replace a set that already has labels")
    sel.set_defaults(func=select)

    srv = sub.add_parser("serve", help="the labelling page")
    srv.add_argument("--port", type=int, default=8093)
    srv.add_argument("--output-root", default=None, help="default: the one the set was drawn from")
    srv.set_defaults(func=serve)

    rep = sub.add_parser("report", help="rates from the labels so far")
    rep.add_argument("--n-boot", type=int, default=2000)
    rep.add_argument("--seed", type=int, default=0)
    rep.set_defaults(func=report)

    res = sub.add_parser("rescore", help="the audited tracks' best reads now vs when drawn")
    res.add_argument("--output-root", default=None)
    res.set_defaults(func=rescore_cmd)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
