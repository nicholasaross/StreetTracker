# Snap gate: geometry, artefacts and procedures

The live geometry table, the files in `.claude/` that define the scene, how to move a trigger
or re-sketch the road, and the planner's modes. [`CLAUDE.md`](../CLAUDE.md#snap-gate-road-polygon--axis-triggers)
keeps the live config block, the rules for editing it and how pipeline mode works. The
reasons behind each value are in [`history/anpr_tuning.md`](history/anpr_tuning.md).

Moved from `CLAUDE.md` on 2026-11-11 with the text unchanged.

## Live geometry (current values)

| Field                               | Value                                                                | Origin                                                                                                                                                                                                           |
| ----------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Polygon vertices                    | 23 fractional `(x, y)`                                               | operator sketch, `.claude/sketch_me_done.png`                                                                                                                                                                    |
| `t_usable_frac`                     | `[0.10, 0.45]`                                                       | Step 11 trim (was `[0.10, 0.67]`)                                                                                                                                                                                |
| `trigger_t_prime`                   | `[0.05, 0.20, 0.35, 0.65, 0.80, 0.95]` (in usable-band coords)       | 3 forward + 3 reverse triggers                                                                                                                                                                                   |
| `trigger_directions`                | `["forward", "forward", "forward", "reverse", "reverse", "reverse"]` | asymmetric: forward = R→L approach, reverse = L→R depart                                                                                                                                                         |
| `pipeline_interval_ms`              | `400`                                                                | base cadence                                                                                                                                                                                                     |
| `pipeline_max_per_track`            | `15`                                                                 | per-track cap                                                                                                                                                                                                    |
| `pipeline_interval_ms_by_direction` | `{}` (empty — uniform 400 ms both directions)                        | Step 13a deployed `{forward:300,reverse:400}` 2026-05-27 but it's **not in the live config** (dropped, likely the 2026-06-13 splice). Not restored — premise falsified (R→L camera-capped ~15-20 %, 2026-06-19). |
| `pipeline_t_usable_by_direction`    | `{forward:[0.10,0.20], reverse:[0.30,0.60]}`                         | per-direction pipeline bands; reverse lower edge nudged 0.25→0.30 on 2026-06-19 (validated 2026-06-22: L→R per-car 92.3 %). See [config layout](../CLAUDE.md#snap-gate-config-layout).                                       |
| Ghost mask                          | rect `[750, 700, 960, 860]` at `source_size=[4512, 2512]`            | covers parked `FD61PVX` car at 4K `(853, 780)` (`t_norm=0.593`)                                                                                                                                                  |

**`trigger_t_prime` values are in `[0, 1]` of the USABLE band, not
global** — so they automatically shrink with `t_usable_frac`. The
"reverse" triggers (placed for L→R entry) now fire at absolute t_norm
0.33 / 0.38 / 0.43 instead of 0.47 / 0.56 / 0.65 — moves L→R fires out
of the FD61PVX zone (intended secondary benefit of the Step 11 trim).

L→R = rear plate visible, R→L = front plate visible (visually confirmed
in 27.3h soak samples).

## Artifacts in `.claude/` (mostly per-install/gitignored)

Most of `.claude/` is per-install scratch and stays ignored, but since
2026-07-09 the **scene geometry** (`ghost_mask.json`, `snap_gate.json`,
`triggers_proposal.json`, `road_polygon_user.json`, `road_zones.json`,
`road_polygon.json`, `reolink_isp_current.json`, `sketch_me_done.png`)
and the **analysis scripts** (`.claude/*.py`, except the `setexp*.py`
camera-ISP helpers which hardcode the LAN IP + admin account) are
**tracked** via gitignore negations — see [Dev-box disaster
recovery](deployment.md#dev-box-disaster-recovery). The `.jpg` overlays / debug snaps
below stay ignored (regenerate from the tracked JSON).

| File                                                 | Purpose                                                                                                                                                                     |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `live_frame.jpg`                                     | Reference 4K snap of the live scene (fetched directly from Reolink).                                                                                                        |
| `sketch_me.png`                                      | 1200×668 downscale of the live frame, given to the operator to sketch on.                                                                                                   |
| `sketch_me_done.png`                                 | Operator's returned sketch — magenta (`#FF00FF`) outline of the visible road tarmac.                                                                                        |
| `road_polygon_user.json`                             | Polygon vertices extracted from the magenta sketch (fractional coords, resolution-independent).                                                                             |
| `triggers_proposal.json`                             | Full trigger spec: polygon vertices, principal axis, centroid, raw t-range, `t_usable_orig`, `triggers_tprime`, `trigger_directions`, `trigger_labels`.                     |
| `triggers_proposal.jpg`                              | Visual overlay on the live frame — yellow road outline, dimmed excluded regions, coloured trigger lines. Re-render via `uv run python .claude/_render_triggers_overlay.py`. |
| `snap_gate.json`                                     | Subset of `triggers_proposal.json` for shipping to `camera.json` under `snapshot.snap_gate`.                                                                                |
| `ghost_mask.json`                                    | Parked-car rects + `source_size`, fed to `alpr-run --ghost-mask`.                                                                                                           |
| `aggregate_step{8,10,11}.py`, `measure_consensus.py` | Re-producible measurement scripts for each step's table.                                                                                                                    |

## Adjusting the snap_gate

Most operator changes ("move T2 toward camera", "drop T1", "add a
trigger between T2 and T3") only need t' values:

1. Edit `triggers_tprime` (and optionally `t_usable_orig`) in
   `.claude/triggers_proposal.json`. Values in `[0, 1]` over the usable
   band — t'=0 = distant edge, t'=1 = near edge.
2. Re-render: `uv run python .claude/_render_triggers_overlay.py`.
3. Regenerate `.claude/snap_gate.json` from the JSON
   (`polygon_frac=vertices_frac`, `trigger_t_prime=triggers_tprime`,
   `t_usable_frac=t_usable_orig`, `trigger_directions` if used).
4. Pull live config, splice the new fields in, scp back, restart:

```bash
scp streettracker@orin:~/streettracker/configs/camera.json /tmp/orin_live.json
# Edit (Python or jq) -- replace the snap_gate value-changes only;
# preserve pipeline_interval_ms / pipeline_max_per_track /
# pipeline_interval_ms_by_direction.
ssh streettracker@orin "cp ~/streettracker/configs/camera.json ~/streettracker/configs/camera.json.bak.$(date -u +%Y%m%dT%H%M%SZ)"
scp /tmp/orin_proposed.json streettracker@orin:~/streettracker/configs/camera.json
ssh streettracker@orin "sudo -n systemctl restart streettracker.service"
ssh streettracker@orin "journalctl -u streettracker.service --since '30 seconds ago' --no-pager | tail -15"
```

For schema-additive changes (new key in `SnapGateSpec`), see
[Schema-additive config changes](../CLAUDE.md#schema-additive-config-changes).

## Re-sketching the road (camera moved, new view)

1. Pull a fresh 4K frame via the Reolink HTTP `cmd=Snap` endpoint →
   `.claude/live_frame.jpg`.
2. Downscale to 1200px wide → `.claude/sketch_me.png`.
3. Operator traces the visible road tarmac as a closed magenta
   (`#FF00FF`) outline → `.claude/sketch_me_done.png`.
4. Extract polygon: detect magenta pixels, dilate + fill, take the
   largest connected component, walk boundary radially from centroid,
   Ramer-Douglas-Peucker simplify (`eps≈6`) →
   `.claude/road_polygon_user.json`.
5. Compute principal axis (PCA on polygon vertices in pixel coords),
   propose initial triggers + `t_usable` band, render overlay,
   iterate with operator until approved. Continue from "Adjusting"
   step 3.

## Live planner behaviour

| `SnapPlannerConfig.road_gate`                       | Mode                        | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| --------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `None`, `right_half_only=True`                      | Right-half zone-thirds      | Pre-polygon fallback. Not used live.                                                                                                                                                                                                                                                                                                                                                                                                       |
| `None`, `right_half_only=False`                     | Legacy peak/decay           | Benchmark only.                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `RoadGateConfig(...)` w/ `pipeline_interval_ms = 0` | Road polygon + trigger-only | Crossing semantics: each frame compares bbox-centre's t' with previous frame's; a not-yet-fired trigger between them fires _if its direction tag matches the motion sign_. After a fire, `prev_t_prime` advances to the trigger's t' so subsequent triggers in the same motion remain detectable one-per-frame. Asymmetric triggers let R→L and L→R each have their own early-capture trigger without one direction consuming the other's. |
| `RoadGateConfig(...)` w/ `pipeline_interval_ms > 0` | + pipeline mode             | **Live deployment since 2026-05-23.** Trigger crossings as above. _Additionally_, while a track sits inside `t_usable`, fire a snap every `pipeline_interval_ms` (or per-direction override), capped at `pipeline_max_per_track`. Pipeline fires share `max_concurrent` with trigger fires; dominate by ~10×. See `consider_pipeline` in `snap_planner.py`.                                                                                |
