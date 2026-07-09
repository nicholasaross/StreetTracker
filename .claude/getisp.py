"""Read-only verifier: print the Reolink's CURRENT exposure/shutter and
score the Anti-Smearing revert -- WITHOUT needing the admin password.

Uses the tracker's own Reolink creds from camera.json (token login;
GetIsp is allowed for the read-only account). Prints exposure / shutter
/ gain and a PASS/FAIL verdict. Never prints the password or the URL.

  PASS = exposure=Auto AND shutter.max=125   (pre-experiment / reverted)
  FAIL = exposure=Anti-Smearing OR shutter.max=32  (experiment still live)

Run ON THE ORIN (reads ~/streettracker/configs/camera.json); no stdin:
    ssh orin 'python3 -' < .claude/getisp.py
or:
    python3 /tmp/getisp.py [path/to/camera.json]
"""

import json
import os
import sys
import urllib.parse
import urllib.request

DEFAULT_CFG = os.path.expanduser("~/streettracker/configs/camera.json")
TIMEOUT = 10


def _post(url, body):
    """POST JSON; return parsed list, or a synthetic error dict on
    failure. Never surfaces the URL (it carries a token)."""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001 -- sanitised, no URL/pw leak
        return [{"code": 1, "error": {"detail": type(e).__name__}}]


def main(argv) -> int:
    cfg_path = argv[0] if argv else DEFAULT_CFG
    try:
        cfg = json.load(open(cfg_path))
    except Exception as e:  # noqa: BLE001
        print(f"cannot read camera.json at {cfg_path}: {type(e).__name__}")
        return 2
    cam = cfg["camera"]
    ip, user, pw = cam["ip"], cam["username"], cam["password"]
    port = cfg.get("ports", {}).get("http", 80)
    base = f"http://{ip}:{port}/cgi-bin/api.cgi"
    print(f"camera {ip}:{port}  account={user} (password not shown)")

    # Token login (proven path; same as setexp.py), then GetIsp.
    lr = _post(
        f"{base}?cmd=Login",
        [{"cmd": "Login", "action": 0,
          "param": {"User": {"userName": user, "password": pw}}}],
    )
    if lr[0].get("code") != 0:
        print("LOGIN FAILED:", json.dumps(lr[0].get("error", lr[0])))
        return 1
    token = lr[0]["value"]["Token"]["name"]

    vr = _post(
        f"{base}?cmd=GetIsp&token={urllib.parse.quote(token)}",
        [{"cmd": "GetIsp", "action": 0, "param": {"channel": 0}}],
    )
    try:
        _post(f"{base}?cmd=Logout&token={urllib.parse.quote(token)}",
              [{"cmd": "Logout", "action": 0, "param": {}}])
    except Exception:
        pass

    if vr[0].get("code") != 0:
        print("GetIsp FAILED:", json.dumps(vr[0].get("error", vr[0])))
        return 1
    isp = vr[0]["value"]["Isp"]
    exp, sh, gain = isp.get("exposure"), isp.get("shutter"), isp.get("gain")
    print(f"exposure={exp}  shutter={sh}  gain={gain}")

    sh_max = sh.get("max") if isinstance(sh, dict) else None
    if exp == "Auto" and sh_max == 125:
        print("PASS: reverted to Auto / shutter max 125 (pre-experiment state).")
        return 0
    if exp == "Anti-Smearing" or sh_max == 32:
        print("FAIL: still Anti-Smearing / shutter max 32 -- revert did NOT take effect.")
        return 3
    print(f"UNEXPECTED: exposure={exp}, shutter.max={sh_max} "
          "(neither the revert target Auto/125 nor the experiment).")
    return 4


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
