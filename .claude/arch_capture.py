"""Capture redacted full-page screenshots of the StreetTracker sites (headless Edge + CDP).

Used to build docs/architecture/ (the architecture artifact). Text redaction happens inside the
page before capture: every known plate and OCR variant (from the showcase API), any UK-format
plate string, and every operator tag (from output/showcase_metadata.json) is covered with a
yellow bar. Plates inside photos are handled afterwards by arch_redact.py, using the <img>
rects saved here. The plate and tag lists live in memory only and are never written to disk.

Raw captures go to .claude/arch_work/ (gitignored) because they still hold plates in photos.

    uv run --no-sync python .claude/arch_capture.py [page ...]

Env overrides: ST_SHOWCASE_URL, ST_CONTROL_URL, ST_ORIN_URL, ST_CAR (car page to capture).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import aiohttp

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / ".claude" / "arch_work"
OUT = WORK / "shots"
OUT.mkdir(parents=True, exist_ok=True)
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT = 9333
WIDTH = 1400
DSF = 2
SHOWCASE = os.environ.get("ST_SHOWCASE_URL", "http://127.0.0.1:8090")
CONTROL = os.environ.get("ST_CONTROL_URL", "http://127.0.0.1:8095")
ORIN = os.environ.get("ST_ORIN_URL", "http://orin:8080")


def _pick_car(cars: list[dict]) -> str:
    """An untagged regular visitor with a DVSA make, 4th by visits (the 2026-10-05 choice)."""
    pool = [
        c
        for c in cars
        if not c.get("tagged")
        and not any((c.get("meta") or {}).get(k) for k in ("name", "owner", "notes"))
        and c.get("make")
        and c.get("kind") == "different-day"
        and c.get("classification") == "visitor"
    ]
    pool.sort(key=lambda c: -c["n_visits"])
    return pool[min(3, len(pool) - 1)]["plate"]


def _redaction_lists() -> tuple[list[str], list[str], str]:
    with urllib.request.urlopen(SHOWCASE + "/api/cars", timeout=600) as r:
        cars = json.load(r)
    plates: set[str] = set()
    for c in cars:
        variants = [v[0] if isinstance(v, (list, tuple)) else v for v in c.get("plate_variants") or []]
        for p in [c.get("plate"), *variants]:
            if not p:
                continue
            p = p.replace(" ", "").upper()
            if len(p) >= 5 and any(ch.isalpha() for ch in p) and any(ch.isdigit() for ch in p):
                plates.add(p)
    meta_path = REPO / "output" / "showcase_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    tags = {(v.get(k) or "").strip() for v in meta.values() for k in ("name", "owner", "notes")}
    tags.discard("")
    car = os.environ.get("ST_CAR") or _pick_car(cars)
    return sorted(plates), sorted(tags, key=len, reverse=True), car


PLATES, META, CAR = _redaction_lists()

PAGES = [
    # name, url, max css height, settle seconds
    ("showcase_gallery", f"{SHOWCASE}/", 1700, 14),
    ("showcase_car", f"{SHOWCASE}/car/{CAR}", 1250, 10),
    ("showcase_stats", f"{SHOWCASE}/stats", 6500, 16),
    ("showcase_people", f"{SHOWCASE}/people", 4200, 10),
    ("showcase_schedule", f"{SHOWCASE}/schedule", 3200, 10),
    # The dashboard fires a pull-estimate SSH call on load. Capture it once, not in a loop:
    # repeated loads left a hung `du` that blocked the panel's Orin poller on 2026-10-05.
    ("control_dashboard", f"{CONTROL}/", 1900, 25),
    ("control_training", f"{CONTROL}/training", 2600, 8),  # only useful while a run is active
    ("orin_dashboard", f"{ORIN}/", 2400, 12),
]

REDACT_JS = r"""
(() => {
  const PLATES = new Set(__PLATES__);
  const META = __META__;
  const modern = /\b[A-Z]{2}[0-9]{2}\s?[A-Z]{3}\b/g;
  const tok = /\b[A-Z0-9]{2,8}\b/g;
  const pair = /(?=\b([A-Z0-9]{1,4} [A-Z0-9]{1,4})\b)/g;
  const yellow = 'background:#d9b300;color:transparent;border-radius:3px;text-shadow:none;';
  function ranges(text) {
    const out = [];
    let m;
    modern.lastIndex = 0;
    while ((m = modern.exec(text))) { if (m[0].replace(/\s/g, '') !== 'ST26TRK') out.push([m.index, m.index + m[0].length]); }
    tok.lastIndex = 0;
    while ((m = tok.exec(text))) {
      if (PLATES.has(m[0])) out.push([m.index, m.index + m[0].length]);
    }
    pair.lastIndex = 0;
    while ((m = pair.exec(text))) {
      if (PLATES.has(m[1].replace(/\s/g, ''))) out.push([m.index, m.index + m[1].length]);
      pair.lastIndex = m.index + 1;
    }
    for (const s of META) {
      let i = text.indexOf(s);
      while (i >= 0) { out.push([i, i + s.length]); i = text.indexOf(s, i + s.length); }
    }
    out.sort((a, b) => a[0] - b[0]);
    const merged = [];
    for (const r of out) {
      if (merged.length && r[0] <= merged[merged.length - 1][1]) {
        merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], r[1]);
      } else merged.push(r.slice());
    }
    return merged;
  }
  let n = 0;
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) {
    const text = node.nodeValue;
    if (!text || !text.trim()) continue;
    const parent = node.parentNode;
    if (!parent || parent.closest && parent.closest('script,style')) continue;
    const rs = ranges(text);
    if (!rs.length) continue;
    n += rs.length;
    if (parent.namespaceURI === 'http://www.w3.org/2000/svg') {
      let t = text;
      for (const [a, b] of rs.slice().reverse()) t = t.slice(0, a) + '\u2588'.repeat(b - a) + t.slice(b);
      node.nodeValue = t;
      continue;
    }
    const frag = document.createDocumentFragment();
    let pos = 0;
    for (const [a, b] of rs) {
      if (a > pos) frag.appendChild(document.createTextNode(text.slice(pos, a)));
      const sp = document.createElement('span');
      sp.setAttribute('style', yellow); sp.className = 'zz-red';
      sp.textContent = text.slice(a, b);
      frag.appendChild(sp);
      pos = b;
    }
    if (pos < text.length) frag.appendChild(document.createTextNode(text.slice(pos)));
    parent.replaceChild(frag, node);
  }
  document.querySelectorAll('.plate').forEach(el => { el.classList.add('zz-red'); el.setAttribute('style', (el.getAttribute('style') || '') + ';' + yellow); n++; });
  document.querySelectorAll('input[name=name],input[name=owner],textarea[name=notes]').forEach(el => {
    if (el.value) { el.classList.add('zz-red'); el.style.cssText += ';' + yellow; n++; }
  });
  document.querySelectorAll('input').forEach(el => {
    const v = (el.value || '').replace(/\s/g, '').toUpperCase();
    if (PLATES.has(v) || META.includes(el.value)) { el.classList.add('zz-red'); el.style.cssText += ';' + yellow; n++; }
  });
  if (location.pathname.startsWith('/car/')) {
    document.querySelectorAll('img:not(#hero)').forEach(img => { img.style.filter = 'blur(1.6px)'; });
  }
  let persons = 0;
  document.querySelectorAll('img').forEach(img => {
    const s = (img.getAttribute('src') || '') + ' ' + (img.getAttribute('data-full') || '');
    if (/person_/.test(s)) { img.style.filter = 'blur(10px)'; persons++; }
  });
  return {redacted: n, person_imgs: persons};
})()
"""

RECTS_JS = r"""
(() => {
  const out = [];
  document.querySelectorAll('img').forEach(img => {
    const r = img.getBoundingClientRect();
    if (r.width < 40 || r.height < 30 || !img.naturalWidth) return;
    out.push({x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height,
              src: img.currentSrc || img.src, fit: getComputedStyle(img).objectFit,
              person: /person_/.test(img.src)});
  });
  const red = [];
  document.querySelectorAll('.zz-red').forEach(el => {
    for (const r of el.getClientRects()) red.push([r.left + scrollX, r.top + scrollY, r.right + scrollX, r.bottom + scrollY]);
  });
  return {imgs: out, red: red, height: Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)};
})()
"""


class Cdp:
    def __init__(self, ws):
        self.ws = ws
        self.i = 0

    async def call(self, method, **params):
        self.i += 1
        my = self.i
        await self.ws.send_json({"id": my, "method": method, "params": params})
        while True:
            msg = await self.ws.receive_json(timeout=120)
            if msg.get("id") == my:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def eval(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")


async def shoot(session, name, url, max_h, settle):
    async with session.put(f"http://127.0.0.1:{PORT}/json/new?about:blank") as r:
        target = await r.json(content_type=None)
    async with session.ws_connect(target["webSocketDebuggerUrl"], max_msg_size=0) as ws:
        cdp = Cdp(ws)
        await cdp.call("Page.enable")
        await cdp.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=1000,
                       deviceScaleFactor=DSF, mobile=False)
        await cdp.call("Page.navigate", url=url)
        await asyncio.sleep(settle)
        info = await cdp.eval(RECTS_JS)
        h = int(min(info["height"], max_h))
        # Grow the viewport to the capture height so lazy images below the fold load.
        await cdp.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=h,
                       deviceScaleFactor=DSF, mobile=False)
        await asyncio.sleep(max(4, settle // 2))
        js = REDACT_JS.replace("__PLATES__", json.dumps(PLATES)).replace("__META__", json.dumps(META))
        red = await cdp.eval(js)
        await asyncio.sleep(1.0)
        info = await cdp.eval(RECTS_JS)
        shot = await cdp.call("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                              clip={"x": 0, "y": 0, "width": WIDTH, "height": h, "scale": 1})
        (OUT / f"{name}.png").write_bytes(base64.b64decode(shot["data"]))
        imgs = [i for i in info["imgs"] if i["y"] < h]
        (OUT / f"{name}.json").write_text(json.dumps({"url": url, "height_css": h, "dsf": DSF,
                                                      "redaction": red, "imgs": imgs, "red_rects": info.get("red", [])}, indent=1))
        print(name, "h", h, "redacted", red, "imgs", len(imgs), flush=True)
    async with session.get(f"http://127.0.0.1:{PORT}/json/close/{target['id']}") as r:
        await r.read()


async def main(only):
    prof = WORK / "edgeprof"
    proc = subprocess.Popen([EDGE, "--headless=new", "--disable-gpu", f"--remote-debugging-port={PORT}",
                             f"--user-data-dir={prof}", "--hide-scrollbars", "--no-first-run",
                             "--mute-audio", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        async with aiohttp.ClientSession() as session:
            for _ in range(40):
                try:
                    async with session.get(f"http://127.0.0.1:{PORT}/json/version") as r:
                        await r.read()
                        break
                except aiohttp.ClientError:
                    await asyncio.sleep(0.5)
            for name, url, max_h, settle in PAGES:
                if only and name not in only:
                    continue
                try:
                    await shoot(session, name, url, max_h, settle)
                except Exception as e:  # noqa: BLE001
                    print(name, "FAILED", repr(e), flush=True)
    finally:
        proc.terminate()


if __name__ == "__main__":
    asyncio.run(main(set(sys.argv[1:])))
