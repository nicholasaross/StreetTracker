"""Re-render .claude/triggers_proposal.jpg from .claude/triggers_proposal.json.

One-shot helper. Loads the live frame + the trigger spec, draws the road
polygon outline (yellow), dims regions outside the usable t-band, and
overlays each trigger as a perpendicular line + circle at the bbox-centre
height projected back onto the principal axis. Output goes back to
.claude/triggers_proposal.jpg.

Run from repo root:
    uv run python .claude/_render_triggers_overlay.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CLAUDE_DIR = Path(__file__).resolve().parent
SPEC_PATH = CLAUDE_DIR / "triggers_proposal.json"
FRAME_PATH = CLAUDE_DIR / "live_frame.jpg"
OUT_PATH = CLAUDE_DIR / "triggers_proposal.jpg"

# BGR for cv2.
COLOR_POLY = (0, 255, 255)        # yellow
COLOR_DIM = (0, 0, 0)
# Direction-coded palettes -- warm hues for forward (approach side),
# cool hues for reverse (departure side). Both ranges go from light/
# saturated for the "early" trigger (first crossed by that direction)
# to muted for the "late" trigger.
FORWARD_PALETTE_BGR = [
    (0, 80, 255),     # red-orange (F1 early)
    (0, 165, 255),    # orange (F2 mid)
    (0, 220, 255),    # amber (F3 late)
]
REVERSE_PALETTE_BGR = [
    (180, 220, 0),    # cyan-teal (R3 late, list-index 3)
    (255, 180, 0),    # azure (R2 mid)
    (255, 80, 80),    # blue (R1 early, list-index 5)
]
BOTH_COLOR_BGR = (255, 0, 255)    # magenta


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text())
    frame = cv2.imread(str(FRAME_PATH))
    if frame is None:
        raise SystemExit(f"could not read {FRAME_PATH}")
    h, w = frame.shape[:2]

    poly_px = np.array(
        [[fx * w, fy * h] for fx, fy in spec["vertices_frac"]], dtype=np.float32
    )

    # PCA axis as recorded in the spec (already oriented "near"-ward).
    ax, ay = spec["main_axis_xy"]
    cx_frac, cy_frac = spec["centroid_frac"]
    cx, cy = cx_frac * w, cy_frac * h
    t_min, t_max = spec["t_min"], spec["t_max"]
    t_usable_lo, t_usable_hi = spec["t_usable_orig"]
    triggers = spec["triggers_tprime"]
    directions = spec.get("trigger_directions") or ["both"] * len(triggers)
    labels = spec.get("trigger_labels", [f"T{i+1}" for i in range(len(triggers))])

    # Assign palette slots per direction so each direction's "early"
    # trigger gets the most saturated colour.
    fwd_indices = [i for i, d in enumerate(directions) if d == "forward"]
    rev_indices = [i for i, d in enumerate(directions) if d == "reverse"]
    colors: list[tuple[int, int, int]] = [BOTH_COLOR_BGR] * len(triggers)
    # Forward palette: smallest t' (earliest crossed by forward motion)
    # is most saturated.
    fwd_sorted = sorted(fwd_indices, key=lambda i: triggers[i])
    for slot, i in enumerate(fwd_sorted):
        colors[i] = FORWARD_PALETTE_BGR[min(slot, len(FORWARD_PALETTE_BGR) - 1)]
    # Reverse palette: largest t' (earliest crossed by reverse motion)
    # is most saturated.
    rev_sorted = sorted(rev_indices, key=lambda i: -triggers[i])
    for slot, i in enumerate(rev_sorted):
        colors[i] = REVERSE_PALETTE_BGR[min(slot, len(REVERSE_PALETTE_BGR) - 1)]

    overlay = frame.copy()

    # Dim regions of the polygon outside the usable t-band by drawing
    # two semi-transparent polygons over the distant tip and the near
    # tail. Cheap version: just draw the full polygon dim, then redraw
    # the usable slice at full brightness. Even simpler: skip the dim
    # and just outline the usable band with a different colour. We do
    # the simple outline approach for clarity.
    cv2.polylines(overlay, [poly_px.astype(np.int32)], True, COLOR_POLY, 2)

    # Perpendicular axis (rotate 90deg).
    px_dir = -ay
    py_dir = ax
    # Line half-length: large enough to cross the polygon -- use the
    # frame diagonal.
    half_len = (w * w + h * h) ** 0.5

    def t_prime_to_raw(tp: float) -> float:
        return t_min + (t_usable_lo + tp * (t_usable_hi - t_usable_lo)) * (t_max - t_min)

    # Draw the usable-band boundaries as faint white dashed lines so
    # the operator can see where the trim sits.
    for boundary_tp, name in [(0.0, "lo"), (1.0, "hi")]:
        t_raw = t_prime_to_raw(boundary_tp)
        bx = cx + ax * t_raw
        by = cy + ay * t_raw
        p1 = (int(bx - px_dir * half_len), int(by - py_dir * half_len))
        p2 = (int(bx + px_dir * half_len), int(by + py_dir * half_len))
        cv2.line(overlay, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)

    # Triggers.
    direction_glyph = {"forward": "F", "reverse": "R", "both": "B"}
    for i, tp in enumerate(triggers):
        color = colors[i]
        direction = directions[i]
        t_raw = t_prime_to_raw(tp)
        bx = cx + ax * t_raw
        by = cy + ay * t_raw
        p1 = (int(bx - px_dir * half_len), int(by - py_dir * half_len))
        p2 = (int(bx + px_dir * half_len), int(by + py_dir * half_len))
        cv2.line(overlay, p1, p2, color, 4, cv2.LINE_AA)
        cv2.circle(overlay, (int(bx), int(by)), 16, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (int(bx), int(by)), 16, (0, 0, 0), 2, cv2.LINE_AA)
        # Direction glyph centred in the circle.
        glyph = direction_glyph[direction]
        (gw, gh), _ = cv2.getTextSize(glyph, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(
            overlay, glyph,
            (int(bx) - gw // 2, int(by) + gh // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
        )

    # Title block.
    n_fwd = sum(1 for d in directions if d == "forward")
    n_rev = sum(1 for d in directions if d == "reverse")
    n_both = sum(1 for d in directions if d == "both")
    title = (
        f"TRIGGERS  forward={n_fwd}  reverse={n_rev}  both={n_both}  "
        f"t_usable={spec['t_usable_orig']}"
    )
    sub_pairs = [f"{lbl}={tp}" for lbl, tp in zip(labels, triggers)]
    sub_lines = [
        "  ".join(sub_pairs[: len(sub_pairs) // 2 or 1]),
        "  ".join(sub_pairs[len(sub_pairs) // 2 or 1:]),
    ]
    pad = 8
    text_h = 20
    n_lines = 1 + len(sub_lines)
    box_h = n_lines * text_h + (n_lines + 1) * pad
    cv2.rectangle(overlay, (0, 0), (w, box_h), (0, 0, 0), -1)
    cv2.putText(
        overlay, title, (pad, pad + text_h - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    for line_i, line in enumerate(sub_lines):
        y = (line_i + 2) * pad + (line_i + 2) * text_h - 6
        cv2.putText(
            overlay, line, (pad, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1, cv2.LINE_AA,
        )

    cv2.imwrite(str(OUT_PATH), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
