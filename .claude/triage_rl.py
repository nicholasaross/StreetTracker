"""R->L per-image ALPR failure triage tool.

Phase 1 (--select): pool the failed R->L car snaps across every session in
--output-root (failed = no plate detection, or best OCR conf below
--fail-thresh on the preferred pipeline), sample ~--n of them with
per-track and per-session caps for diversity, pre-render padded vehicle
crops (tracked-vehicle bbox drawn in green, plate detection -- if any --
in orange) into .claude/triage/crops/, and write manifest.json.

Phase 2 (--serve, default when a manifest exists): serve a local
single-page labelling UI on http://127.0.0.1:8091/ that shows one crop at
a time with keyboard buckets 1-6; every click persists atomically to
.claude/triage/labels.json. Resume-safe: reload jumps to the first
unlabelled item.

Buckets (what they tell us):
  1 stale_box     car missed/escaped the green box         -> bbox staleness vs snap
                  (box shows road / only part of car)         latency (software: motion-
                                                              extrapolated hint)
  2 not_visible   occluded or facing away                  -> geometry/band
  3 cut_off       plate clipped by frame/crop edge         -> crop logic (software)
  4 too_small     visible but too far/tiny to resolve      -> optics/zoom or band
  5 smeared       visible but motion-blurred / RS-skewed   -> speed/exposure/band
  6 sharp_unread  human-legible (or nearly) yet unread     -> detector/OCR (software)
  7 junk          no car at all / parked car only / other  -> noise

Judge the plate relative to the GREEN box: that box (+<=30 px pad) is all
the plate detector ever sees. A sharp plate outside the green box is
bucket 1, not bucket 6.

Run from repo root:
  uv run python .claude/triage_rl.py --select          # build sample (once)
  uv run python .claude/triage_rl.py                   # serve the UI
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw

from streettracker.analysis.snap_assets import load_bbox_index, resolve_bbox_hint

TRIAGE_DIR = Path(__file__).resolve().parent / "triage"
CROPS_DIR = TRIAGE_DIR / "crops"
MANIFEST_PATH = TRIAGE_DIR / "manifest.json"
LABELS_PATH = TRIAGE_DIR / "labels.json"

BUCKETS = {
    1: ("stale_box", "Car missed/escaped the green box — box shows road or part of car"),
    2: ("not_visible", "Plate not visible — occluded or facing away"),
    3: ("cut_off", "Plate clipped by frame/crop edge"),
    4: ("too_small", "Visible but too far/tiny to resolve characters"),
    5: ("smeared", "Visible but motion-smeared / blurred"),
    6: ("sharp_unread", "Sharp & human-legible — software should have read it"),
    7: ("junk", "No car at all / parked car only / other"),
}


# ----------------------------------------------------------------------
# Phase 1: selection + crop rendering
# ----------------------------------------------------------------------

def select(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    root = Path(args.output_root)
    pool: list[dict] = []  # candidate failed snaps, lightweight
    session_summaries = []

    for sd in sorted(root.iterdir()):
        if not sd.is_dir() or not sd.name.startswith("session_"):
            continue
        label = sd.name
        alpr_path = sd / f"{label}_alpr.json"
        data_path = sd / f"{label}_data.json"
        if not alpr_path.exists() or not data_path.exists():
            continue
        records = json.loads(data_path.read_text())
        cars_rl = {
            int(r["track_id"]): r
            for r in records
            if r.get("class_name") == "car"
            and r.get("direction") == "right to left"
            and r.get("track_id") is not None
        }
        if not cars_rl:
            continue

        # Per-track rollup: did ANY snap of this track read, eventually?
        best_by_tid: dict[int, dict] = {}
        bt_path = sd / f"{label}_alpr_by_track.json"
        if bt_path.exists():
            bt = json.loads(bt_path.read_text())
            entries = next(iter(bt.values())) if isinstance(bt, dict) else bt
            for t in entries:
                bp = t.get("best_preferred")
                if t.get("track_id") is not None and bp:
                    best_by_tid[int(t["track_id"])] = bp

        fails_by_track: dict[int, list[dict]] = {}
        n_images = n_failed = 0
        for e in json.loads(alpr_path.read_text()):
            tid = e.get("track_id")
            if tid is None or int(tid) not in cars_rl:
                continue
            if e.get("pipeline") not in (None, "preferred"):
                continue  # ignore bespoke rows if a --both run wrote them
            n_images += 1
            conf = e.get("ocr_conf")
            if conf is not None and conf >= args.fail_thresh:
                continue
            n_failed += 1
            fails_by_track.setdefault(int(tid), []).append(e)

        for tid, fails in fails_by_track.items():
            take = rng.sample(fails, min(args.per_track, len(fails)))
            rec = cars_rl[tid]
            bp = best_by_tid.get(tid)
            for e in take:
                pool.append(
                    {
                        "session": label,
                        "session_dir": str(sd),
                        "image": e["image"],
                        "track_id": tid,
                        "snap_index": e.get("snap_index"),
                        "det_bbox": e.get("det_bbox"),
                        "det_conf": e.get("det_conf"),
                        "ocr_text": e.get("ocr_text"),
                        "ocr_conf": e.get("ocr_conf"),
                        "error": e.get("error"),
                        "speed_px_s": rec.get("speed_px_s"),
                        "color": rec.get("color"),
                        "time_start": rec.get("time_start"),
                        "track_best_text": (bp or {}).get("ocr_text"),
                        "track_best_conf": (bp or {}).get("ocr_conf"),
                        "track_best_canonical": (bp or {}).get("canonical_uk_shape"),
                    }
                )
        session_summaries.append((label, n_images, n_failed))

    print(f"{'session':38} {'R->L imgs':>9} {'failed':>7}")
    for label, n_images, n_failed in session_summaries:
        print(f"{label:38} {n_images:>9} {n_failed:>7}")
    print(f"\ncandidate pool after per-track cap ({args.per_track}/track): {len(pool)}")

    rng.shuffle(pool)
    selected: list[dict] = []
    per_session: dict[str, int] = {}
    for cand in pool:
        if len(selected) >= args.n:
            break
        if per_session.get(cand["session"], 0) >= args.per_session:
            continue
        selected.append(cand)
        per_session[cand["session"]] = per_session.get(cand["session"], 0) + 1

    # Render crops, grouped by session so the bbox index loads once each.
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    dropped_no_hint = 0
    by_session: dict[str, list[dict]] = {}
    for cand in selected:
        by_session.setdefault(cand["session"], []).append(cand)

    for label, cands in by_session.items():
        sd = Path(cands[0]["session_dir"])
        bbox_index, sub_size = load_bbox_index(sd)
        for cand in cands:
            img_path = sd / cand["image"]
            if not img_path.exists():
                dropped_no_hint += 1
                continue
            hint = resolve_bbox_hint(
                img_path, cand["track_id"], cand["snap_index"], bbox_index, sub_size
            )
            if hint is None:
                dropped_no_hint += 1
                continue
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                w, h = im.size
                x1, y1, x2, y2 = hint
                pad_x = int((x2 - x1) * 0.35) + 40
                pad_y = int((y2 - y1) * 0.35) + 40
                cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
                crop = im.crop((cx1, cy1, cx2, cy2))
                draw = ImageDraw.Draw(crop)
                draw.rectangle(
                    [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1],
                    outline=(0, 255, 60),
                    width=4,
                )
                db = cand["det_bbox"]
                if db:
                    draw.rectangle(
                        [db[0] - cx1, db[1] - cy1, db[2] - cx1, db[3] - cy1],
                        outline=(255, 150, 0),
                        width=3,
                    )
                crop_name = f"{label}__{cand['image']}"
                crop.save(CROPS_DIR / crop_name, quality=88)
            items.append(
                {
                    **{k: v for k, v in cand.items() if k != "session_dir"},
                    "key": f"{label}/{cand['image']}",
                    "crop": crop_name,
                    "full_rel": str(Path(cand["session_dir"]) / cand["image"]),
                    "cx_frac": round(((x1 + x2) / 2) / w, 4),
                    "cy_frac": round(((y1 + y2) / 2) / h, 4),
                    "bbox_4k": list(hint),
                }
            )

    manifest = {
        "n": len(items),
        "fail_thresh": args.fail_thresh,
        "seed": args.seed,
        "buckets": {str(k): v[0] for k, v in BUCKETS.items()},
        "items": items,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1))
    no_det = sum(1 for i in items if not i["det_bbox"])
    print(f"selected {len(items)} snaps from {len(by_session)} sessions "
          f"({dropped_no_hint} dropped, no bbox hint)")
    print(f"  plate detector found nothing : {no_det}")
    print(f"  plate found but OCR < {args.fail_thresh:.2f}  : {len(items) - no_det}")
    print(f"manifest: {MANIFEST_PATH}")


# ----------------------------------------------------------------------
# Phase 2: labelling server
# ----------------------------------------------------------------------

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>R&rarr;L ALPR failure triage</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font:14px/1.45 system-ui,sans-serif; background:#14161a; color:#dfe3e8; }
  header { display:flex; gap:16px; align-items:center; padding:10px 16px; background:#1d2026;
           position:sticky; top:0; z-index:2; flex-wrap:wrap; }
  #bar { flex:1 1 200px; height:8px; background:#2a2e36; border-radius:4px; overflow:hidden; min-width:120px;}
  #bar div { height:100%; background:#4caf7d; width:0%; transition:width .15s; }
  main { display:flex; flex-direction:column; align-items:center; padding:12px; gap:10px; }
  #imgwrap { max-width:96vw; overflow:hidden; border:1px solid #2a2e36; border-radius:6px;
             cursor:zoom-in; background:#000; }
  #img { display:block; max-width:96vw; max-height:62vh; transition:transform .12s; }
  #meta { display:flex; gap:14px; flex-wrap:wrap; color:#9aa3ad; font-size:13px; justify-content:center;}
  #meta b { color:#dfe3e8; font-weight:600; }
  #buttons { display:grid; grid-template-columns:repeat(3,minmax(220px,1fr)); gap:8px; max-width:900px; width:96vw; }
  button.bk { padding:10px 12px; border:1px solid #2a2e36; border-radius:8px; background:#1d2026;
              color:#dfe3e8; text-align:left; cursor:pointer; font-size:13px; }
  button.bk:hover { border-color:#4a5260; }
  button.bk.sel { outline:2px solid #4caf7d; }
  button.bk kbd { display:inline-block; min-width:18px; text-align:center; background:#2a2e36;
                  border-radius:4px; padding:1px 5px; margin-right:8px; font-weight:700; }
  #counts { display:flex; gap:10px; flex-wrap:wrap; color:#9aa3ad; font-size:12px; }
  #counts span b { color:#dfe3e8; }
  nav { display:flex; gap:8px; align-items:center; }
  nav button { background:#1d2026; color:#dfe3e8; border:1px solid #2a2e36; border-radius:6px;
               padding:6px 12px; cursor:pointer; }
  a { color:#7fb4e8; }
  #done { display:none; background:#1d2026; border:1px solid #2a2e36; border-radius:8px;
          padding:16px 22px; max-width:640px; }
  #done table { border-collapse:collapse; margin-top:8px; }
  #done td { padding:3px 12px 3px 0; }
  .b1{border-left:4px solid #e35d5d}.b2{border-left:4px solid #d98343}.b3{border-left:4px solid #b06de8}
  .b4{border-left:4px solid #5da9e3}.b5{border-left:4px solid #e8b75d}.b6{border-left:4px solid #4caf7d}
  .b7{border-left:4px solid #6b7280}
  #legend { color:#8a93a0; font-size:12px; }
  #legend i.g { color:#3fdf5c; font-style:normal; } #legend i.o { color:#ffa040; font-style:normal; }
</style></head><body>
<header>
  <strong>R&rarr;L failure triage</strong>
  <span id="pos">&ndash;</span>
  <div id="bar"><div></div></div>
  <span id="prog">&ndash;</span>
  <nav>
    <button onclick="nav(-1)" title="left arrow">&larr; prev</button>
    <button onclick="nav(1)" title="right arrow">next &rarr;</button>
    <button onclick="jumpUnlabelled()" title="key: n">next unlabelled</button>
    <button onclick="clearLabel()" title="key: u">clear (U)</button>
  </nav>
</header>
<main>
  <div id="imgwrap"><img id="img" alt="crop"></div>
  <div id="legend"><i class="g">green</i> = tracked-car box &mdash; the plate detector only
    ever sees this box +&le;30px &middot; <i class="o">orange</i> = plate detection (if any)
    &middot; judge the plate relative to the green box &middot; click image to zoom</div>
  <div id="meta"></div>
  <div id="buttons"></div>
  <div id="counts"></div>
  <div id="done"></div>
</main>
<script>
let M=null, L={}, cur=0, zoomed=false;
const BK = {
  1:["stale_box","Car missed/escaped the green box &mdash; box shows road or part of car"],
  2:["not_visible","Plate not visible &mdash; occluded or facing away"],
  3:["cut_off","Plate clipped by frame/crop edge"],
  4:["too_small","Visible but too far/tiny to resolve characters"],
  5:["smeared","Visible but motion-smeared / blurred"],
  6:["sharp_unread","Sharp &amp; human-legible &mdash; software should have read it"],
  7:["junk","No car at all / parked car only / other"],
};
async function init(){
  const s = await (await fetch('/api/state')).json();
  M = s.manifest; L = s.labels;
  const btns = document.getElementById('buttons');
  for (const k of Object.keys(BK)) {
    const b = document.createElement('button');
    b.className = 'bk b'+k; b.id = 'bk'+k;
    b.innerHTML = '<kbd>'+k+'</kbd>'+BK[k][1];
    b.onclick = () => label(parseInt(k));
    btns.appendChild(b);
  }
  document.getElementById('imgwrap').onclick = (ev)=>{
    const img = document.getElementById('img');
    if (zoomed) { img.style.transform=''; zoomed=false; }
    else {
      const r = img.getBoundingClientRect();
      img.style.transformOrigin = ((ev.clientX-r.left)/r.width*100)+'% '+((ev.clientY-r.top)/r.height*100)+'%';
      img.style.transform = 'scale(2.6)'; zoomed=true;
    }
  };
  cur = firstUnlabelled();
  render();
}
function firstUnlabelled(){
  for (let i=0;i<M.items.length;i++) if (!(M.items[i].key in L)) return i;
  return 0;
}
function jumpUnlabelled(){
  for (let i=1;i<=M.items.length;i++){
    const j=(cur+i)%M.items.length;
    if (!(M.items[j].key in L)) { cur=j; render(); return; }
  }
  render();
}
function nav(d){ cur=(cur+d+M.items.length)%M.items.length; render(); }
async function label(k){
  const it=M.items[cur];
  L[it.key]=k;
  await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({key:it.key,bucket:k})});
  jumpUnlabelled();
}
async function clearLabel(){
  const it=M.items[cur];
  delete L[it.key];
  await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({key:it.key,bucket:null})});
  render();
}
function render(){
  const it=M.items[cur]; zoomed=false;
  const img=document.getElementById('img');
  img.style.transform=''; img.src='/crops/'+encodeURIComponent(it.crop);
  document.getElementById('pos').textContent=(cur+1)+' / '+M.items.length;
  const n=Object.keys(L).length;
  document.getElementById('prog').textContent=n+' labelled';
  document.querySelector('#bar div').style.width=(100*n/M.items.length)+'%';
  const det = it.det_bbox ? ('plate det '+(it.det_conf||0).toFixed(2)+
      (it.ocr_text?(' &middot; OCR &ldquo;'+it.ocr_text+'&rdquo; @ '+(it.ocr_conf||0).toFixed(2)):' &middot; OCR none'))
    : 'no plate detection';
  const tb = it.track_best_text ? ('track best: '+it.track_best_text+' @ '+(it.track_best_conf||0).toFixed(2)+
      (it.track_best_canonical?' (canonical)':'')) : 'track never read';
  document.getElementById('meta').innerHTML =
    '<span><b>'+it.session+'</b> &middot; track '+it.track_id+' &middot; snap '+it.snap_index+'</span>'+
    '<span>speed <b>'+Math.round(it.speed_px_s||0)+'</b> px/s &middot; '+(it.color||'?')+'</span>'+
    '<span>'+det+'</span><span>'+tb+'</span>'+
    '<span>box <b>'+(it.bbox_4k?(it.bbox_4k[2]-it.bbox_4k[0])+'&times;'+(it.bbox_4k[3]-it.bbox_4k[1]):'?')+
    '</b> px &middot; pos x=<b>'+(it.cx_frac??'?')+'</b></span>'+
    '<span><a href="/full?i='+cur+'" target="_blank">full 4K frame</a></span>';
  for (const k of Object.keys(BK))
    document.getElementById('bk'+k).classList.toggle('sel', L[it.key]==k);
  const cts={}; for (const v of Object.values(L)) cts[v]=(cts[v]||0)+1;
  document.getElementById('counts').innerHTML = Object.keys(BK).map(k=>
    '<span>'+BK[k][0]+': <b>'+(cts[k]||0)+'</b></span>').join(' ');
  const done=document.getElementById('done');
  if (n>=M.items.length){
    done.style.display='block';
    done.innerHTML='<strong>All '+n+' labelled &mdash; thank you!</strong> Labels are in '+
      '<code>.claude/triage/labels.json</code>.<table>'+ Object.keys(BK).map(k=>
      '<tr><td>'+BK[k][1]+'</td><td><b>'+(cts[k]||0)+'</b></td><td>'+
      Math.round(100*(cts[k]||0)/n)+'%</td></tr>').join('')+'</table>';
  } else done.style.display='none';
}
document.addEventListener('keydown',(e)=>{
  if (e.key>='1'&&e.key<='7') label(parseInt(e.key));
  else if (e.key==='ArrowLeft') nav(-1);
  else if (e.key==='ArrowRight') nav(1);
  else if (e.key==='u'||e.key==='U') clearLabel();
  else if (e.key==='n'||e.key==='N') jumpUnlabelled();
});
init();
</script></body></html>
"""


def _atomic_write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


class Handler(BaseHTTPRequestHandler):
    manifest: dict = {}
    labels: dict = {}
    crop_names: set[str] = set()

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        if u.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif u.path == "/api/state":
            body = json.dumps({"manifest": self.manifest, "labels": self.labels})
            self._send(200, body.encode(), "application/json")
        elif u.path.startswith("/crops/"):
            name = u.path[len("/crops/"):]
            from urllib.parse import unquote
            name = unquote(name)
            if name not in self.crop_names or "/" in name or "\\" in name:
                self._send(404, b"not found", "text/plain")
                return
            self._send(200, (CROPS_DIR / name).read_bytes(), "image/jpeg")
        elif u.path == "/full":
            try:
                i = int(parse_qs(u.query).get("i", ["-1"])[0])
                item = self.manifest["items"][i]
                p = Path(item["full_rel"])
                if not p.is_file() or p.suffix.lower() != ".jpg":
                    raise ValueError
                self._send(200, p.read_bytes(), "image/jpeg")
            except (ValueError, IndexError, KeyError, OSError):
                self._send(404, b"not found", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/label":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(n))
            key = payload["key"]
            bucket = payload["bucket"]
            if not any(it["key"] == key for it in self.manifest["items"]):
                raise ValueError("unknown key")
            if bucket is None:
                self.labels.pop(key, None)
            else:
                bucket = int(bucket)
                if bucket not in BUCKETS:
                    raise ValueError("bad bucket")
                self.labels[key] = bucket
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            self._send(400, b"bad request", "text/plain")
            return
        _atomic_write(LABELS_PATH, self.labels)
        self._send(200, b'{"ok":true}', "application/json")


def serve(args: argparse.Namespace) -> None:
    if not MANIFEST_PATH.exists():
        sys.exit("no manifest — run with --select first")
    Handler.manifest = json.loads(MANIFEST_PATH.read_text())
    Handler.labels = (
        json.loads(LABELS_PATH.read_text()) if LABELS_PATH.exists() else {}
    )
    Handler.crop_names = {it["crop"] for it in Handler.manifest["items"]}
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    n = len(Handler.manifest["items"])
    print(f"triage: {n} items, {len(Handler.labels)} already labelled")
    print(f"serving on http://127.0.0.1:{args.port}/  (Ctrl-C to stop)")
    httpd.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--select", action="store_true", help="(re)build the sample")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--fail-thresh", type=float, default=0.5)
    ap.add_argument("--per-track", type=int, default=2)
    ap.add_argument("--per-session", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args()
    if args.select:
        select(args)
    else:
        serve(args)


if __name__ == "__main__":
    main()
