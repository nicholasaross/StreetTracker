"""Merge snap_gate.json into camera_config.json's snapshot.snap_gate.

Designed to run on the Nano (Python 3.6 compatible). Reads both files
from the working directory, replaces only the snap_gate key, and
re-writes camera_config.json with 4-space indent + trailing newline so
the diff is minimal versus the operator's hand-edited convention.

Intended invocation:
    python3 _merge_snap_gate.py
"""

from __future__ import print_function

import json
from collections import OrderedDict


def main():
    with open("camera_config.json") as f:
        cfg = json.load(f, object_pairs_hook=OrderedDict)
    with open("snap_gate.json") as f:
        sg = json.load(f, object_pairs_hook=OrderedDict)

    if "snapshot" not in cfg:
        raise SystemExit("camera_config.json missing top-level 'snapshot' key")

    cfg["snapshot"]["snap_gate"] = sg

    with open("camera_config.json", "w") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")

    new_sg = cfg["snapshot"]["snap_gate"]
    print("merged snap_gate into camera_config.json")
    print("  trigger_t_prime    =", new_sg.get("trigger_t_prime"))
    print("  trigger_directions =", new_sg.get("trigger_directions"))
    print("  t_usable_frac      =", new_sg.get("t_usable_frac"))
    print("  polygon_frac len   =", len(new_sg.get("polygon_frac", [])))


if __name__ == "__main__":
    main()
