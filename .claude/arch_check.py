"""One preview pass of docs/architecture/index.html before republishing the artifact.

Wraps the artifact source in the document skeleton the Artifact publisher adds, then checks
1400 px and 390 px in dark and light: horizontal overflow, broken images, the display font,
console errors, three deep-linked pop-ups and a system-map click. Screenshots go to
.claude/arch_work/check/.

    uv run --no-sync python .claude/arch_check.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
from pathlib import Path

import aiohttp

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / ".claude" / "arch_work"
WORK.mkdir(parents=True, exist_ok=True)
PREVIEW = REPO / "docs" / "architecture" / "_preview.html"  # deleted after the run
PREVIEW.write_text(
    '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
    'content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>'
    + (REPO / "docs" / "architecture" / "index.html").read_text(encoding="utf-8")
    + "</body></html>",
    encoding="utf-8",
)
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT = 9334
URL = PREVIEW.as_uri()
OUT = WORK / "check"
OUT.mkdir(exist_ok=True)


async def main():
    proc = subprocess.Popen([EDGE, "--headless=new", "--disable-gpu", f"--remote-debugging-port={PORT}",
                             f"--user-data-dir={WORK / 'edgeprof2'}", "--hide-scrollbars", "--no-first-run",
                             "--allow-file-access-from-files", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        async with aiohttp.ClientSession() as s:
            for _ in range(40):
                try:
                    async with s.get(f"http://127.0.0.1:{PORT}/json/version") as r:
                        await r.read()
                        break
                except aiohttp.ClientError:
                    await asyncio.sleep(0.5)
            async with s.put(f"http://127.0.0.1:{PORT}/json/new?about:blank") as r:
                t = await r.json(content_type=None)
            async with s.ws_connect(t["webSocketDebuggerUrl"], max_msg_size=0) as ws:
                i = 0
                logs = []

                async def call(method, **params):
                    nonlocal i
                    i += 1
                    my = i
                    await ws.send_json({"id": my, "method": method, "params": params})
                    while True:
                        m = await ws.receive_json(timeout=120)
                        if m.get("method") in ("Runtime.consoleAPICalled", "Runtime.exceptionThrown", "Log.entryAdded"):
                            logs.append(m)
                        if m.get("id") == my:
                            return m.get("result", {})

                async def ev(expr):
                    r = await call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
                    return r.get("result", {}).get("value")

                await call("Page.enable")
                await call("Runtime.enable")
                await call("Log.enable")
                for theme in ("dark", "light"):
                    for w, h, name in ((1400, 900, "desk"), (390, 844, "phone")):
                        await call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": theme}])
                        await call("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=(w < 500))
                        await call("Page.navigate", url=URL)
                        await asyncio.sleep(4)
                        info = await ev("""(() => ({sw: document.scrollingElement.scrollWidth, iw: innerWidth,
                          imgsBroken: [...document.images].filter(i => i.complete && !i.naturalWidth).map(i => i.src.split('/').pop()),
                          font: document.fonts.check('600 20px "Barlow Condensed"'),
                          wide: [...document.querySelectorAll('main *')].filter(e => e.getBoundingClientRect().right > innerWidth + 1 && !e.closest('.scrollbox,.tbl-wrap')).slice(0,5).map(e => e.tagName + '.' + e.className)}))()""")
                        print(theme, name, json.dumps(info))
                        H = await ev("document.scrollingElement.scrollHeight")
                        clipH = min(H, 2600 if name == "desk" else 3000)
                        await call("Emulation.setDeviceMetricsOverride", width=w, height=clipH, deviceScaleFactor=1, mobile=(w < 500))
                        await asyncio.sleep(1.5)
                        shot = await call("Page.captureScreenshot", format="jpeg", quality=70)
                        (OUT / f"{theme}_{name}.jpg").write_bytes(base64.b64decode(shot["data"]))
                # modal + deep link on desktop dark
                await call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "dark"}])
                await call("Emulation.setDeviceMetricsOverride", width=1400, height=900, deviceScaleFactor=1, mobile=False)
                for nid in ("rt-snap", "web-showcase", "en-rescore"):
                    await call("Page.navigate", url=URL + "#" + nid)
                    await asyncio.sleep(3)
                    st = await ev("({open: document.getElementById('dlg').open, title: document.getElementById('dlg-title').textContent})")
                    print("modal", nid, st)
                    shot = await call("Page.captureScreenshot", format="jpeg", quality=72)
                    (OUT / f"modal_{nid}.jpg").write_bytes(base64.b64decode(shot["data"]))
                # map node click
                await call("Page.navigate", url=URL)
                await asyncio.sleep(3)
                r = await ev("document.querySelector('#sysmap [data-n=\"hw-orin\"]').dispatchEvent(new MouseEvent('click', {bubbles:true})), document.getElementById('dlg-title').textContent")
                print("map click ->", r)
                errs = [m for m in logs if m.get("method") == "Runtime.exceptionThrown" or
                        (m.get("method") == "Runtime.consoleAPICalled" and m["params"].get("type") == "error") or
                        (m.get("method") == "Log.entryAdded" and m["params"]["entry"].get("level") == "error")]
                print("errors:", len(errs))
                for e in errs[:8]:
                    print(json.dumps(e)[:400])
    finally:
        proc.terminate()
        PREVIEW.unlink(missing_ok=True)


asyncio.run(main())
