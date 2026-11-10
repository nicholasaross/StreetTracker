"""HTML summary dashboard generator.

Ported from NanoTracker's `generate_html()` (nano_tracker.py ~line 673).
Behavior preserved exactly so live deploys can swap implementations
without retraining users on a new dashboard:

  - Virtualized renderer (server emits JSON; browser renders only the
    visible viewport's worth of rows).
  - Polls ``vehicles.json`` every ``refresh_seconds`` seconds and
    re-renders in place; sort state + scroll position + active tab
    survive refresh.
  - Sibling ``vehicles.json`` written next to the HTML.
  - Tiny ``index.html`` redirector so ``http://<host>:<port>/`` lands on
    the latest summary without having to know the timestamped filename.

String concatenation is intentional (NOT ``str.format`` / f-strings inside
the JS body): CSS / JS braces collide with format placeholders. The few
runtime values that need interpolation are concatenated explicitly.
"""

from __future__ import annotations

import html as html_mod
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from streettracker.common.fileio import atomic_write_text
from streettracker.common.schema import TrackRecord

# Inline ST26 TRK number-plate brand mark — the project logo, shared with the
# showcase/control headers (docs/assets/st26trk.svg is the canonical source).
# Inlined so the generated summary stays a single self-contained file.
ST26_LOGO_SVG = (
    '<svg style="height:26px;width:auto;vertical-align:-5px;margin-right:10px" '
    'viewBox="0 0 168 44" role="img" aria-label="ST26 TRK number plate" '
    'xmlns="http://www.w3.org/2000/svg">'
    '<rect x="1.5" y="1.5" width="165" height="41" rx="5" fill="#F2C200" '
    'stroke="#1B2026" stroke-opacity="0.55" stroke-width="3"/>'
    '<text x="84" y="29.5" text-anchor="middle" '
    "font-family=\"'Cascadia Code','Consolas','DejaVu Sans Mono',monospace\" "
    'font-size="17" font-weight="700" letter-spacing="2.4" fill="#1B2026">ST26 TRK</text>'
    "</svg>"
)

# Matching browser-tab icon (URL-encoded inline SVG; same mark as the showcase).
ST26_FAVICON_LINK = (
    '<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,'
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
    "<rect x='2' y='7' width='28' height='18' rx='3.5' fill='%23F2C200' "
    "stroke='%231B2026' stroke-width='2'/>"
    "<text x='16' y='21' text-anchor='middle' font-family='monospace' "
    "font-size='13' font-weight='700' fill='%231B2026'>ST</text></svg>\">"
)


def summary_row(record: TrackRecord | dict[str, Any]) -> dict[str, Any]:
    """The fields the summary page shows for one track, in the order the
    page's row payload has always carried them."""
    if isinstance(record, TrackRecord):
        # Read the attributes directly. ``to_json_dict`` deep-copies the whole
        # record, bbox lists included, to pick out a dozen scalars.
        return {
            "track_id": record.track_id,
            "class_name": record.class_name,
            "color": record.color,
            "time_start": record.time_start,
            "time_start_unix": record.time_start_unix,
            "duration_visible": record.duration_visible,
            "direction": record.direction,
            "speed_px_s": record.speed_px_s,
            "lane": record.lane,
            "avg_confidence": record.avg_confidence,
            "asset_prefix": record.asset_prefix,
            "main_snaps": list(record.main_snaps),
        }
    v = record
    return {
        "track_id": v["track_id"],
        "class_name": v["class_name"],
        "color": v["color"],
        "time_start": v["time_start"],
        "time_start_unix": v["time_start_unix"],
        "duration_visible": v["duration_visible"],
        "direction": v["direction"],
        "speed_px_s": v["speed_px_s"],
        "lane": v["lane"],
        "avg_confidence": v["avg_confidence"],
        "asset_prefix": v.get("asset_prefix", "vehicle"),
        "main_snaps": list(v.get("main_snaps", [])),
    }


class SummaryRows:
    """The summary page's rows, built up one finalised track at a time.

    Each row is serialised once, when it is added, so producing the page's
    payload is a string join whatever the session's length. The live runtime
    rebuilds the page every time the street goes quiet; when it re-derived
    every row from every record on each rebuild, that took 0.5 s plus 0.24 ms
    per record on the Orin, on the event loop, and no frame was processed
    meanwhile (``.claude/loop_stall_check.py``, 2026-11-10).

    A finalised record never changes, so a row never needs rewriting.
    """

    __slots__ = ("_encoded", "class_counts")

    def __init__(self, records: Iterable[TrackRecord | dict[str, Any]] = ()) -> None:
        self._encoded: list[str] = []
        self.class_counts: dict[str, int] = {}
        for record in records:
            self.add(record)

    def add(self, record: TrackRecord | dict[str, Any]) -> None:
        row = summary_row(record)
        self._encoded.append(json.dumps(row, separators=(",", ":")))
        name = row["class_name"]
        self.class_counts[name] = self.class_counts.get(name, 0) + 1

    def __len__(self) -> int:
        return len(self._encoded)

    def to_json(self) -> str:
        """The rows as one JSON array: what ``vehicles.json`` holds and the
        page embeds."""
        return "[" + ",".join(self._encoded) + "]"

    def summary_text(self) -> str:
        n = len(self._encoded)
        if not n:
            return "No detections yet"
        parts = [
            f"{c} {name}{'s' if c != 1 else ''}" for name, c in sorted(self.class_counts.items())
        ]
        return f"{n} entr{'ies' if n != 1 else 'y'}: {', '.join(parts)}"


@dataclass(frozen=True, slots=True)
class SummarySnapshot:
    """One rendering of the summary, as immutable strings: safe to hand to
    another thread to write while the runtime carries on."""

    data_json: str  # -> vehicles.json
    page: str  # -> <session>_summary.html


def generate_html(
    records: Iterable[TrackRecord | dict[str, Any]],
    output_dir: Path,
    html_path: Path,
    session_label: str,
    meta: dict[str, Any],
    refresh_seconds: int = 15,
) -> None:
    """Render the session dashboard to ``html_path`` and the row payload
    to ``output_dir / "vehicles.json"``, from a complete list of records.

    Also writes / updates an ``index.html`` redirector so the dashboard
    is reachable at the directory root. The live runtime keeps a
    :class:`SummaryRows` instead and calls :func:`render_summary` +
    :func:`write_summary` itself.
    """
    snapshot = render_summary(SummaryRows(records), session_label, meta, refresh_seconds)
    write_summary(snapshot, output_dir, html_path)


def write_summary(snapshot: SummarySnapshot, output_dir: Path, html_path: Path) -> None:
    """Write one rendering to disk. Each file is replaced in one step, so the
    dashboard's server and a browser polling ``vehicles.json`` never read a
    half-written file. Blocking; the runtime calls it from a worker thread."""
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_dir / "vehicles.json", snapshot.data_json)
    atomic_write_text(html_path, snapshot.page)

    # Index redirector so http://<host>:<port>/ lands on the latest summary
    # without needing to know the timestamped filename.
    index_path = output_dir / "index.html"
    if not index_path.exists() or index_path.stat().st_mtime < html_path.stat().st_mtime - 1:
        atomic_write_text(
            index_path,
            f'<!DOCTYPE html><meta http-equiv="refresh" content="0; url={html_path.name}">',
        )


def render_summary(
    rows: SummaryRows,
    session_label: str,
    meta: dict[str, Any],
    refresh_seconds: int = 15,
) -> SummarySnapshot:
    """Build the page and its row payload from the accumulated rows. Cheap:
    no per-row work beyond joining strings already serialised."""
    summary_text = rows.summary_text()
    meta_kv = " · ".join(f"{k}: {v}" for k, v in sorted(meta.items()))
    data_json = rows.to_json()

    poll_ms = int(max(refresh_seconds, 0)) * 1000
    page = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        + "<title>StreetTracker — "
        + html_mod.escape(session_label)
        + "</title>"
        + ST26_FAVICON_LINK
        + "<style>"
        + "*{margin:0;padding:0;box-sizing:border-box}"
        + 'body{background:#1a1a1a;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,monospace;padding:24px}'
        + "h1{font-size:1.4rem;margin-bottom:8px}"
        + ".summary{font-size:0.9rem;color:#aaa;margin-bottom:8px}"
        + ".meta{font-size:0.75rem;color:#777;margin-bottom:20px;font-family:monospace}"
        + ".thead,.row{display:grid;grid-template-columns:200px 56px 76px 76px 100px 80px 110px 96px 76px 80px;gap:10px;padding:8px 12px;align-items:center;font-size:0.85rem}"
        + ".thead{background:#252525;color:#ccc;font-weight:600;border-bottom:1px solid #444;position:sticky;top:0;z-index:2}"
        + ".thead span{cursor:pointer;user-select:none}"
        + ".thead span:hover{color:#fff}"
        + '.thead span.sk-asc::after{content:" \\25B2"}'
        + '.thead span.sk-desc::after{content:" \\25BC"}'
        + ".vp{height:85vh;overflow-y:auto;border:1px solid #333;position:relative}"
        + ".spacer{position:relative}"
        + ".row{position:absolute;left:0;right:0;border-bottom:1px solid #2a2a2a}"
        + ".row:hover{background:#252525}"
        + ".row img{max-width:80px;max-height:54px;border-radius:4px;display:block}"
        + ".no-img{width:80px;height:54px;background:#333;border-radius:4px;display:flex;align-items:center;justify-content:center;color:#666;font-size:0.7rem}"
        + ".thumb-cell{display:flex;flex-direction:row;align-items:flex-start;gap:8px;width:200px}"
        + ".thumb-cell>a:first-child{display:block;line-height:0;flex:0 0 80px}"
        + ".snap-row{display:flex;flex-wrap:wrap;gap:3px;min-height:14px;flex:1;align-content:flex-start}"
        + ".snap-badge{display:inline-block;background:#0a84ff;color:#fff;font-size:0.6rem;font-weight:700;padding:1px 5px;border-radius:3px;text-decoration:none;line-height:1.3;min-width:12px;text-align:center}"
        + ".snap-badge:hover{background:#1f95ff}"
        + ".snap-badge.label{background:transparent;color:#888;padding:1px 0;font-weight:400}"
        + ".tabs{display:flex;gap:4px;margin-bottom:0;border-bottom:1px solid #333}"
        + ".tab{padding:8px 16px;background:#252525;color:#aaa;border:1px solid #333;border-bottom:none;border-radius:4px 4px 0 0;cursor:pointer;font-size:0.85rem;user-select:none;font-weight:600}"
        + ".tab:hover{color:#fff}"
        + ".tab.active{background:#1a1a1a;color:#e0e0e0;border-color:#444;position:relative;top:1px}"
        + ".tab .ct{color:#666;font-weight:400;margin-left:6px;font-size:0.8rem}"
        + ".tab.active .ct{color:#888}"
        + "</style></head><body>"
        + "<h1>"
        + ST26_LOGO_SVG
        + "StreetTracker Summary</h1>"
        + '<div class="summary">Session: '
        + html_mod.escape(session_label)
        + ' &mdash; <span id="summary-text">'
        + html_mod.escape(summary_text)
        + "</span></div>"
        + '<div class="meta">'
        + html_mod.escape(meta_kv)
        + "</div>"
        + '<div class="tabs" id="tabs"></div>'
        + '<div class="vp" id="vp">'
        + '  <div class="thead">'
        + "    <span>Thumbnail</span>"
        + '    <span data-sk="track_id">ID</span>'
        + '    <span data-sk="class_name">Type</span>'
        + '    <span data-sk="color">Color</span>'
        + '    <span data-sk="time_start_unix" class="sk-desc">Time</span>'
        + '    <span data-sk="duration_visible">Duration</span>'
        + '    <span data-sk="direction">Direction</span>'
        + '    <span data-sk="speed_px_s">Speed</span>'
        + '    <span data-sk="lane">Lane</span>'
        + '    <span data-sk="avg_confidence">Conf</span>'
        + "  </div>"
        + '  <div class="spacer" id="spacer"><div id="rows"></div></div>'
        + "</div>"
        + '<script id="vehicles-data" type="application/json">'
        + data_json
        + "</script>"
        + "<script>"
        + 'let DATA=JSON.parse(document.getElementById("vehicles-data").textContent);'
        + "const POLL_MS="
        + str(poll_ms)
        + ";"
        + "const ROW_H=88;"
        + 'const TABS=[{label:"Cars",cls:"car"},{label:"People",cls:"person"}];'
        + "function readTabHash(){const m=location.hash.match(/tab=([^&]+)/);return m?decodeURIComponent(m[1]):null}"
        + "let activeTab=readTabHash()||TABS[0].cls;"
        + 'let sortKey="time_start_unix",sortAsc=false;'
        + "let sorted=[];"
        + 'const VP=document.getElementById("vp"),SPACER=document.getElementById("spacer"),ROWS=document.getElementById("rows"),TABBAR=document.getElementById("tabs");'
        + 'function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",\'"\':"&quot;"}[c]))}'
        + 'function sortData(){sorted.sort((a,b)=>{let va=a[sortKey],vb=b[sortKey];if(typeof va==="number")return sortAsc?va-vb:vb-va;va=String(va);vb=String(vb);return sortAsc?va.localeCompare(vb):vb.localeCompare(va)})}'
        + "function rowHtml(v,i){"
        + 'const t=(v.time_start||"").substr(11,8);'
        + 'const pfx=v.asset_prefix||"vehicle";'
        + 'const thumb=pfx+"_"+v.track_id+".jpg";'
        + 'const hq=pfx+"_"+v.track_id+"_hq.jpg";'
        + "const snaps=v.main_snaps||[];"
        + "const snapBadges=snaps.length>0"
        + '?`<span class="snap-badge label">4K</span>`+snaps.map(n=>`<a class="snap-badge" href="${pfx}_${v.track_id}_main_${n}.jpg" target="_blank" title="main-stream snapshot ${n} of ${snaps.length}">${n}</a>`).join("")'
        + ':"";'
        + 'return `<div class="row" style="top:${i*ROW_H}px">'
        + '<span class="thumb-cell">'
        + '<a href="${hq}" target="_blank" title="open full-quality crop"><img src="${thumb}" loading="lazy"></a>'
        + '<span class="snap-row">${snapBadges}</span>'
        + "</span>"
        + "<span>#${v.track_id}</span>"
        + "<span>${esc(v.class_name)}</span>"
        + "<span>${esc(v.color)}</span>"
        + '<span title="${esc(v.time_start)}">${t}</span>'
        + "<span>${v.duration_visible}s</span>"
        + "<span>${esc(v.direction)}</span>"
        + "<span>${v.speed_px_s} px/s</span>"
        + "<span>${esc(v.lane)}</span>"
        + "<span>${v.avg_confidence.toFixed(3)}</span>"
        + "</div>`"
        + "}"
        + "function render(){"
        + 'SPACER.style.height=(sorted.length*ROW_H)+"px";'
        + "const top=VP.scrollTop,visH=VP.clientHeight;"
        + "const start=Math.max(0,Math.floor(top/ROW_H)-5);"
        + "const end=Math.min(sorted.length,Math.ceil((top+visH)/ROW_H)+5);"
        + 'let h="";for(let i=start;i<end;i++)h+=rowHtml(sorted[i],i);'
        + "ROWS.innerHTML=h;"
        + "}"
        + 'document.querySelectorAll(".thead span[data-sk]").forEach(s=>{'
        + 's.addEventListener("click",()=>{'
        + "const k=s.dataset.sk;"
        + "if(sortKey===k)sortAsc=!sortAsc;else{sortKey=k;sortAsc=true}"
        + 'document.querySelectorAll(".thead span").forEach(x=>x.classList.remove("sk-asc","sk-desc"));'
        + 's.classList.add(sortAsc?"sk-asc":"sk-desc");'
        + "sortData();render();"
        + "});"
        + "});"
        + 'VP.addEventListener("scroll",render);'
        + 'window.addEventListener("resize",render);'
        + "function updateSummary(){"
        + "const counts={};"
        + "DATA.forEach(v=>{counts[v.class_name]=(counts[v.class_name]||0)+1});"
        + 'const parts=Object.keys(counts).sort().map(n=>{const c=counts[n];return c+" "+n+(c!==1?"s":"")});'
        + "const n=DATA.length;"
        + 'const txt=n>0?(n+" detection"+(n!==1?"s":"")+": "+parts.join(", ")):"No detections yet";'
        + 'const el=document.getElementById("summary-text");if(el)el.textContent=txt;'
        + "}"
        + "function renderTabs(){"
        + "const counts={};"
        + "DATA.forEach(v=>{counts[v.class_name]=(counts[v.class_name]||0)+1});"
        + "TABBAR.innerHTML=TABS.map(t=>{"
        + "const c=counts[t.cls]||0;"
        + 'const cls="tab"+(t.cls===activeTab?" active":"");'
        + 'return `<span class="${cls}" data-cls="${t.cls}">${t.label}<span class="ct">${c}</span></span>`;'
        + '}).join("");'
        + 'TABBAR.querySelectorAll(".tab").forEach(el=>{'
        + 'el.addEventListener("click",()=>setTab(el.dataset.cls));'
        + "});"
        + "}"
        + "function setTab(cls){"
        + "if(cls===activeTab)return;"
        + "activeTab=cls;"
        + 'history.replaceState(null,"","#tab="+encodeURIComponent(cls));'
        + "renderTabs();"
        + "applyFilterSortRender(true);"
        + "}"
        + "function applyFilterSortRender(resetScroll){"
        + "sorted=DATA.filter(v=>v.class_name===activeTab);"
        + "sortData();"
        + "if(resetScroll)VP.scrollTop=0;"
        + "render();"
        + "}"
        + "renderTabs();applyFilterSortRender(false);updateSummary();"
        + "if(POLL_MS>0){"
        + "setInterval(()=>{"
        + 'fetch("vehicles.json?t="+Date.now()).then(r=>r.ok?r.json():null).then(d=>{'
        + "if(!d)return;"
        + "DATA=d;renderTabs();applyFilterSortRender(false);updateSummary();"
        + "}).catch(()=>{});"
        + "},POLL_MS);"
        + "}"
        + 'window.addEventListener("hashchange",()=>{'
        + "const t=readTabHash();if(t&&t!==activeTab){activeTab=t;renderTabs();applyFilterSortRender(true);}"
        + "});"
        + "</script>"
        + "</body></html>"
    )
    return SummarySnapshot(data_json=data_json, page=page)
