"""Capture night / IR-mode footage for the R16 test (docs/data_integrity_review.md).

Runs ON THE ORIN, detached, from the repo checkout (it reads the live
``configs/camera.json`` for the camera address and the read-only account, so
no credentials pass through anything else):

    cd ~/streettracker && nohup setsid ~/.local/bin/uv run --no-sync \
        python ~/ir_test/ir_capture.py --out ~/ir_test \
        > ~/ir_test/capture.log 2>&1 < /dev/null &

From ``--from`` (default 18:30) it polls one 4K snap a minute. Once three
polls in a row are IR (the camera's black-and-white mode, judged with the
runtime's own ``is_ir_frame``), it records ``--minutes`` of the sub-stream
with ffmpeg (stream copy, no re-encode) and, alongside, a 4K snap every
``--snap-interval`` seconds. At ``--morning`` it repeats for
``--morning-minutes`` if the camera is still in IR (the morning-overrun case);
if not, it skips. It writes ``status.json`` as it goes and exits after the
morning window.

Output (``--out``): ``evening/`` and ``morning/`` each with ``sub.mp4``,
``snaps/snap_<unix_ms>.jpg``, and ``capture.json`` (window start/end, the
ffmpeg start time, every snap's request/landed times). Plus ``isp.json`` (the
camera's GetIsp, read-only) when the account may read it.

With ``--windows`` it instead records at fixed times whatever the camera's
mode (used 2026-10-04, when the camera had been forced to colour at night):
each window gets its own ``<MMDD_HHMM>/`` directory and records whether the
camera was in IR at its start.

Analysis happens on the dev box after a pull (`.claude/ir_capture_detect.py`,
`ir_capture_report.py` and `ir_capture_plates.py`); nothing here runs inference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path.home() / "streettracker" / "src"))
from streettracker.cli.run import _build_rtsp_url  # noqa: E402
from streettracker.common.config import StreetTrackerConfig  # noqa: E402
from streettracker.device.ir_detector import is_ir_frame  # noqa: E402
from streettracker.device.snapshotter import build_snap_url  # noqa: E402

_POLL_S = 60
_IR_POLLS = 3  # consecutive IR polls before capturing


def _log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def _status(out: Path, **kw: Any) -> None:
    kw["at"] = datetime.now().isoformat(timespec="seconds")
    (out / "status.json").write_text(json.dumps(kw, indent=1))


def _fetch(url: str, timeout: float = 10.0) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read()
        return data if data[:3] == b"\xff\xd8\xff" else None
    except Exception as e:  # network blips are expected; keep going
        _log(f"snap failed: {type(e).__name__}: {e}")
        return None


def _is_ir(jpeg: bytes) -> bool | None:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else bool(is_ir_frame(img))


def _sleep_until(t: datetime) -> None:
    while (left := (t - datetime.now()).total_seconds()) > 0:
        time.sleep(min(left, 300))


def _get_isp(cfg: StreetTrackerConfig, out: Path) -> None:
    """Save the camera's ISP settings (dayNight mode + threshold), read-only."""
    q = urllib.parse.urlencode(
        {"cmd": "GetIsp", "user": cfg.camera.username, "password": cfg.camera.password}
    )
    url = f"http://{cfg.camera.ip}:{cfg.ports.http}/cgi-bin/api.cgi?{q}"
    body = json.dumps([{"cmd": "GetIsp", "action": 1, "param": {"channel": 0}}]).encode()
    try:
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            (out / "isp.json").write_bytes(r.read())
        _log("saved isp.json")
    except Exception as e:
        _log(f"GetIsp failed (fine if the account can't read it): {e}")


def capture(
    out: Path, rtsp: str, snap_url: str, minutes: float, snap_interval: float, label: str
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "snaps").mkdir(exist_ok=True)
    dur = int(minutes * 60)
    meta: dict[str, Any] = {"label": label, "minutes": minutes, "snaps": []}
    meta["ffmpeg_start_unix"] = time.time()
    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-i",
            rtsp,
            "-c",
            "copy",
            "-t",
            str(dur),
            "-y",
            str(out / "sub.mp4"),
        ],
        stdout=subprocess.DEVNULL,
        stderr=open(out / "ffmpeg.log", "w"),  # noqa: SIM115
    )
    _log(f"[{label}] capturing {minutes} min (ffmpeg pid {ff.pid})")
    end = time.time() + dur
    n = 0
    while time.time() < end:
        t0 = time.time()
        data = _fetch(snap_url)
        t1 = time.time()
        if data:
            name = f"snap_{int(t0 * 1000)}.jpg"
            (out / "snaps" / name).write_bytes(data)
            meta["snaps"].append({"file": name, "request_unix": t0, "landed_unix": t1})
            n += 1
        if n % 50 == 0:
            _status(out.parent, phase=f"{label} capture", snaps=n, until=end)
        time.sleep(max(0.0, snap_interval - (time.time() - t0)))
    try:
        ff.wait(timeout=120)
    except subprocess.TimeoutExpired:
        ff.terminate()
    meta["end_unix"] = time.time()
    meta["ffmpeg_returncode"] = ff.returncode
    (out / "capture.json").write_text(json.dumps(meta, indent=1))
    _log(f"[{label}] done: {n} snaps, ffmpeg rc {ff.returncode}")


def wait_for_ir(snap_url: str, out: Path, deadline: datetime) -> bool:
    streak = 0
    while datetime.now() < deadline:
        data = _fetch(snap_url)
        ir = _is_ir(data) if data else None
        streak = streak + 1 if ir else 0
        _status(out, phase="waiting for IR", last_poll_ir=ir, streak=streak)
        if streak >= _IR_POLLS:
            _log("camera is in IR mode")
            return True
        time.sleep(_POLL_S)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path.home() / "streettracker/configs/camera.json"))
    ap.add_argument("--out", type=Path, default=Path.home() / "ir_test")
    ap.add_argument("--from", dest="start", default="18:30", help="start polling (HH:MM)")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--snap-interval", type=float, default=2.0)
    ap.add_argument("--morning", default="07:00", help="morning window start (HH:MM), '' to skip")
    ap.add_argument("--morning-minutes", type=float, default=20.0)
    ap.add_argument(
        "--windows",
        default="",
        help="fixed windows instead of waiting for IR, e.g. '18:30=20,21:30=30,07:00=20' "
        "(each runs at its next occurrence after the previous one; mode recorded, not required)",
    )
    args = ap.parse_args()

    cfg = StreetTrackerConfig.from_json_file(args.config)
    rtsp = _build_rtsp_url(cfg, cfg.nano.preferred_stream)
    cam = cfg.camera
    snap_url = build_snap_url(cam.ip, cfg.ports.http, cam.username, cam.password)
    args.out.mkdir(parents=True, exist_ok=True)
    _get_isp(cfg, args.out)

    if args.windows:
        prev_end = datetime.now()
        for spec in args.windows.split(","):
            hhmm, minutes = spec.split("=")
            hh, mm = map(int, hhmm.split(":"))
            at = prev_end.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if at < prev_end:
                at += timedelta(days=1)
            label = f"{at:%m%d_%H%M}"
            _status(args.out, phase=f"sleeping until {label}", until=at.isoformat())
            _log(f"next window {label} ({minutes} min)")
            _sleep_until(at)
            data = _fetch(snap_url)
            ir = _is_ir(data) if data else None
            _log(f"[{label}] camera in IR: {ir}")
            capture(args.out / label, rtsp, snap_url, float(minutes), args.snap_interval, label)
            meta_path = args.out / label / "capture.json"
            meta = json.loads(meta_path.read_text())
            meta["ir_at_start"] = ir
            meta_path.write_text(json.dumps(meta, indent=1))
            prev_end = datetime.now()
        _status(args.out, phase="finished")
        _log("finished")
        return 0

    now = datetime.now()
    hh, mm = map(int, args.start.split(":"))
    start = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if start < now - timedelta(hours=6):
        start += timedelta(days=1)
    _status(args.out, phase="sleeping", until=start.isoformat())
    _log(f"polling from {start:%Y-%m-%d %H:%M}")
    _sleep_until(start)
    if wait_for_ir(snap_url, args.out, start + timedelta(hours=8)):
        capture(args.out / "evening", rtsp, snap_url, args.minutes, args.snap_interval, "evening")
    else:
        _log("no IR before the deadline; skipping the evening window")

    if args.morning:
        hh, mm = map(int, args.morning.split(":"))
        morning = datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
        if morning < datetime.now():
            morning += timedelta(days=1)
        _status(args.out, phase="sleeping until morning", until=morning.isoformat())
        _sleep_until(morning)
        data = _fetch(snap_url)
        if data and _is_ir(data):
            capture(
                args.out / "morning",
                rtsp,
                snap_url,
                args.morning_minutes,
                args.snap_interval,
                "morning",
            )
        else:
            _log("camera not in IR at the morning window; skipping it")
    _status(args.out, phase="finished")
    _log("finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
