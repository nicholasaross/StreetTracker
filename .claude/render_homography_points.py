"""Render 4 road-plane corner points (a rectangle on the tarmac) onto the
live frame, for the operator to measure real WIDTH + LENGTH -> homography
for perspective-correct speed. Output: .claude/homography_measure.jpg.
"""

import json
import pathlib

import cv2
import numpy as np

base = pathlib.Path(".claude")
prop = json.loads((base / "triggers_proposal.json").read_text())
img = cv2.imread(str(base / "live_frame.jpg"))
H, W = img.shape[:2]
sw, sh = prop["source_size"]
cx, cy = prop["centroid_frac"][0] * sw, prop["centroid_frac"][1] * sh
ax, ay = prop["main_axis_xy"]
perp = (-ay, ax)

# faint road polygon for context
verts_img = np.array([[round(x * W), round(y * H)] for x, y in prop["vertices_frac"]], np.int32)
cv2.polylines(img, [verts_img], True, (0, 220, 220), 3)

# vertices in source px -> (t along axis, s across axis)
V = [(x * sw, y * sh) for x, y in prop["vertices_frac"]]
TS = [((p[0] - cx) * ax + (p[1] - cy) * ay, (p[0] - cx) * perp[0] + (p[1] - cy) * perp[1]) for p in V]
tmed = sorted(t for t, _ in TS)[len(TS) // 2]
near = [(s, p) for (t, s), p in zip(TS, V) if t <= tmed]
far = [(s, p) for (t, s), p in zip(TS, V) if t > tmed]
NL = min(near)[1]; NR = max(near)[1]   # near-left / near-right kerb
FL = min(far)[1]; FR = max(far)[1]      # far-left / far-right kerb


def to_img(p):
    return round(p[0] / sw * W), round(p[1] / sh * H)


corners = [("P1", NL), ("P2", NR), ("P3", FR), ("P4", FL)]  # clockwise
quad = np.array([to_img(p) for _, p in corners], np.int32)
cv2.polylines(img, [quad], True, (0, 0, 255), 5)
for lbl, p in corners:
    q = to_img(p)
    cv2.circle(img, q, 22, (0, 255, 255), -1)
    cv2.circle(img, q, 22, (0, 0, 0), 3)
    cv2.putText(img, lbl, (q[0] + 26, q[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 0, 0), 8)
    cv2.putText(img, lbl, (q[0] + 26, q[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 255, 255), 3)


def lab(t, y):
    cv2.putText(img, t, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 8)
    cv2.putText(img, t, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)


lab("Red box approximates a rectangle on the road. On Google Maps satellite, measure:", H - 130)
lab("WIDTH  = P1->P2  (kerb to kerb)", H - 82)
lab("LENGTH = P1->P4  (along the road)   ... tell me both, in metres.", H - 38)

out = base / "homography_measure.jpg"
cv2.imwrite(str(out), img)
print(f"wrote {out}  ({W}x{H})")
