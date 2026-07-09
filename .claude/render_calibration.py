"""Render the monitored road segment + travel axis onto the live 4K frame,
so the operator can supply the real-world length for speed (px/s -> mph)
calibration. Output: .claude/calibration_measure.jpg.
"""

import json
import pathlib

import cv2
import numpy as np

base = pathlib.Path(".claude")
prop = json.loads((base / "triggers_proposal.json").read_text())
img = cv2.imread(str(base / "live_frame.jpg"))
H, W = img.shape[:2]

# Road polygon: fractional verts -> frame px. Faint fill + bright outline.
verts = np.array([[round(x * W), round(y * H)] for x, y in prop["vertices_frac"]], np.int32)
overlay = img.copy()
cv2.fillPoly(overlay, [verts], (0, 200, 200))
img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
cv2.polylines(img, [verts], True, (0, 255, 255), 6)

# Travel axis (principal axis through centroid). Geometry is in source_size
# (1200x668) px -> fractional -> current frame px.
sw, sh = prop["source_size"]
cx, cy = prop["centroid_frac"][0] * sw, prop["centroid_frac"][1] * sh
ax, ay = prop["main_axis_xy"]


def to_frame(t: float) -> tuple[int, int]:
    return round((cx + t * ax) / sw * W), round((cy + t * ay) / sh * H)


A, B = to_frame(prop["t_min"]), to_frame(prop["t_max"])
ok, p1, p2 = cv2.clipLine((0, 0, W, H), A, B)
if ok:
    cv2.arrowedLine(img, p1, p2, (0, 0, 255), 7, tipLength=0.03)
    cv2.arrowedLine(img, p2, p1, (0, 0, 255), 7, tipLength=0.03)
    for p in (p1, p2):
        cv2.circle(img, p, 18, (0, 0, 255), -1)


def label(txt: str, y: int) -> None:
    cv2.putText(img, txt, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 0), 9)
    cv2.putText(img, txt, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3)


label("YELLOW = monitored road segment.  RED = direction of travel.", H - 95)
label("Tell me the real length of this road stretch (metres).", H - 45)

out = base / "calibration_measure.jpg"
cv2.imwrite(str(out), img)
print(f"wrote {out}  ({W}x{H})")
