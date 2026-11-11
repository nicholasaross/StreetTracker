# Deploying and recovering

Procedures for a wiped or new Orin, what to back up, how to rebuild the dev box, and the
JetPack 7 plan. The day-to-day deploy rules (the schema-additive order, config backups) stay
in [`CLAUDE.md`](../CLAUDE.md#deploying-to-the-orin).

Moved from `CLAUDE.md` on 2026-11-11 with the text unchanged.

## Fresh deployment procedure

Canonical reference for any redeploy on a wiped Orin (new chassis, SD
swap, or JetPack upgrade — see [JetPack 7 upgrade plan](#jetpack-7-upgrade-plan)
for JP7 deltas).

### Backups to keep off-device

Three things are not in the repo and not portable from a fresh OS:

| File                                         | Why                                                                                            |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `~/streettracker/configs/camera.json`        | Reolink credentials + operator-traced road polygon + per-install snap_gate tuning. Gitignored. |
| `/etc/sudoers.d/streettracker-svc`           | Lets the `streettracker` user run `systemctl * streettracker.service` without password prompt. |
| `~/.ssh/authorized_keys` for `streettracker` | SSH keys for admin.                                                                            |

The TRT engine (`yolov8m.engine`) is deliberately _not_ on this list —
it's GPU-architecture-bound and must be rebuilt on the target device.

### Dev-box disaster recovery

The table above is **Orin-side only**. The dev box (this analysis
machine) holds a separate, mostly-irreplaceable set. As of 2026-07-09
the durable, non-sensitive pieces are pushed so a fresh `git clone`
recovers them (the repo is **public**, so secrets and plate/person
imagery are deliberately kept out):

| Artifact                                                                                                                                                                          | Where it now lives                                                   | Recreate cost if lost                                                                    |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Operator-traced **scene geometry** (`.claude/{ghost_mask,snap_gate,triggers_proposal,road_polygon_user,road_zones,road_polygon,reolink_isp_current}.json` + `sketch_me_done.png`) | **In repo** (gitignore negations)                                    | Irreplaceable without the physical camera + operator re-sketch + weeks of band re-tuning |
| **Analysis / measurement scripts** (`.claude/*.py` — verdict/band/eval/bakeoff/coverage)                                                                                          | **In repo**                                                          | Re-derive methodology from scratch                                                       |
| **Small inference models** (`bodytype_b0.pt` 16 MB, ALPR `license_plate_detector.pt` 6 MB) + all `*.meta.json` sidecars                                                           | **In repo**                                                          | 12 h train / hard to reacquire                                                           |
| **Production make model** `makemodel_b0.pt` (164 MB, B6)                                                                                                                          | **GitHub Release `models-2026-10-26`** holds production `uk_make_0929_b6` (SHA-256 `f660a9eb…`); `models-2026-07-09` holds the older 0707 B6. Promotions from 2026-10-26 on publish `makemodel-<run>` automatically | ~1.5 days train on a months-built corpus                                                 |

Still **NOT** in GitHub — keep a private/external backup (too big, or
PII, or secret):

- **Training corpora** `runs/uk_crops_0924_576` (make — the plate-anchored
  corpus the production model trained on; also carries body-type + colour
  labels) + the legacy `runs/uk_crops_0707_576` / `runs/uk_crops_0708_bt_384`
  — the distilled datasets; source Orin sessions prune at 7 d, so these
  can't be re-mined once the box dies.
- **Per-session derived JSON** `output/*_{data,alpr,dvsa_labels,vehicles,makemodel,bodytype,people}.json`
  — all the ANPR/DVSA/people labels (plate PII); tar the non-image files.
- **Secrets**: `configs/dvsa.json` (DVSA API key), any `camera.json`
  (Reolink pw) — encrypted store / password manager, never the public repo.
- **Camera-ISP helpers** `.claude/setexp*.py` — re-ignored (hardcode the
  LAN IP + admin account); recover from the private backup.

The `output/` 4K images and `.claude/v*_main_*.jpg` debug snaps are
disposable (raw imagery, plate/person PII) — external backup only if the
imagery itself matters, not for tooling recovery.

### One-time host setup

```bash
# As an admin user on the freshly-flashed Orin:
sudo useradd -m -s /bin/bash streettracker
sudo usermod -aG video,sudo,systemd-journal streettracker
# video for nvidia; systemd-journal so streettracker can read its own
# service logs (without this, `journalctl -u streettracker` looks
# empty rather than erroring).
sudo mkdir -p /home/streettracker/.ssh
sudo cp ~/.ssh/authorized_keys /home/streettracker/.ssh/
sudo chown -R streettracker:streettracker /home/streettracker/.ssh
sudo chmod 700 /home/streettracker/.ssh
sudo chmod 600 /home/streettracker/.ssh/authorized_keys

# Scoped NOPASSWD for service management.
echo "streettracker ALL=(ALL) NOPASSWD: /bin/systemctl * streettracker.service" \
  | sudo tee /etc/sudoers.d/streettracker-svc
sudo chmod 440 /etc/sudoers.d/streettracker-svc
```

### Repo + venv setup

```bash
ssh streettracker@orin
git clone https://github.com/nicholasaross/StreetTracker.git ~/streettracker
cd ~/streettracker
scripts/setup_orin.sh         # idempotent: apt deps + uv + uv sync +
                              # tensorrt symlinks + bashrc LD_LIBRARY_PATH
                              # + nvpmodel
```

Handles both JP6 (Ubuntu 22.04 / Py 3.10) and JP7 (Ubuntu 24.04 / Py
3.12) via uv's `.python-version` honouring.

**Recovery from a venv rebuild.** A `uv sync` that recreates `.venv`
(e.g. after a `.python-version` change) wipes the system-tensorrt
symlinks. The systemd service then fails with `ModuleNotFoundError: No
module named 'tensorrt'` and restart-loops. Recovery:

```bash
ssh streettracker@orin "cd ~/streettracker && scripts/setup_orin.sh --symlinks-only && sudo -n systemctl restart streettracker.service"
```

No sudo/apt/network needed — `--symlinks-only` runs the symlink step in
isolation. Hit this on 2026-05-25 during the 3.10 → 3.11 → 3.10 pin
churn, hence the dedicated flag.

### Per-deploy files

```bash
scp <backup>/camera.json streettracker@orin:~/streettracker/configs/

cd ~/streettracker
uv run streettracker export-engine yolov8m.pt
# ~5 min on Orin Nano Super. The output filename must match
# `inference.engine_path` in camera.json — the load-time validator
# catches mismatches with a clear JSON-path error.
```

### Systemd install + smoke

```bash
sudo cp scripts/systemd/streettracker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo -n systemctl enable --now streettracker.service
journalctl -u streettracker -f                  # Ctrl-C once frames flow
curl http://orin:8080/                          # "no sessions yet" page
                                                # or latest summary redirect
```

Roll back: `sudo -n systemctl disable --now streettracker.service`. SD
card unchanged; next `enable --now` starts a fresh session.

## JetPack 7 upgrade plan

NVIDIA's JP7 for Orin Nano has shipped as **JetPack 7.2** (Jetson Linux
39.2, CUDA 13.2.1, TensorRT 10.16.2, Ubuntu 24.04, Py 3.12 native;
verified 2026-06-07). Before flashing, five repo edits have to land or
the redeploy will fail at `uv sync` — and the hard prerequisite gating
all of it is **torch-wheel availability** (see the go/no-go gate below):

| What                     | Where                                               | JP6 today                                                      | JP7 target                                                                                                                                                          |
| ------------------------ | --------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python pin               | `.python-version`                                   | `3.10`                                                         | `3.12`                                                                                                                                                              |
| Jetson torch wheel index | `pyproject.toml` `[[tool.uv.index]]`                | `https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/`             | `https://pypi.jetson-ai-lab.io/jp7/cu13x/+simple/` (CUDA 13.2 → `cu132`/`cu130`; **not published as of 2026-06-07** — gate-check via `scripts/check_jp7_wheel.ps1`) |
| cuDSS dep                | `pyproject.toml` `nvidia-cudss-cu12`                | `cu12`                                                         | `cu13` if JP7 ships CUDA 13.x (confirm)                                                                                                                             |
| CUDA lib path            | `scripts/setup_orin.sh` `CUDA_LIB=...`              | `/usr/local/cuda-12.6/lib64`                                   | auto-detect or hardcode `cuda-13.x`                                                                                                                                 |
| Systemd LD_LIBRARY_PATH  | `scripts/systemd/streettracker.service` Environment | `cuda-12.6/lib64` + `python3.10/site-packages/nvidia/cu12/lib` | both paths shift                                                                                                                                                    |

**Go/no-go gate.** JP7.2 is out, but **don't flash until the wheel gate
is green**: run `scripts/check_jp7_wheel.ps1` from the dev box (exit 0 =
a `jp7`/CUDA-13.x `cp312`+`aarch64` torch **and** torchvision wheel is
published on jetson-ai-lab — the only `sm_87`-safe source; the official
PyTorch SBSA `cu126` wheel has `sm_87` gaps → silent NaN). **NOT YET as
of 2026-06-07** (`jp7/` 404s); stay on JP6 meanwhile — the live service
is safe (`--no-sync` blocks resync against the decaying JP6 index).

**When the gate is green.** Spin up `claude/jp7-readiness`, make the five
edits, then flash and run the [Fresh deployment
procedure](#fresh-deployment-procedure). **Caveat — there is only ONE
Orin, no separate test box** (the earlier "verify on a test Orin" plan is
infeasible): flash JP7.2 onto a _second_ NVMe and keep the current JP6
drive physically intact as the rollback + `output/` backstop; a
major-version flash also rewrites the module QSPI bootloader, so
reverting = re-flash QSPI to JP6 + reinsert the old drive (downtime, not
a swap). PR #15 hardening (forgiving stream-name lookup + engine-path
validation) means small config drift won't block the redeploy.
