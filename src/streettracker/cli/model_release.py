"""``streettracker model-release`` -- back the production make model up as a
GitHub Release.

The make model (EfficientNet-B6@528, ~165 MB) is over GitHub's 100 MB
per-file limit, so ``.gitignore`` keeps it out of the repo, and a Release is
its only off-box copy. The panel's **promote** playbook runs this as its last
step. Before 2026-10-26 nothing did, and production ran a month ahead of its
only backup.

The repository is **public**, so this refuses to upload a checkpoint whose
non-weight contents hold anything plate-shaped or a filesystem path, and
refuses a sidecar holding anything plate-shaped. Today's checkpoints carry
only the weights plus arch, make names, input size, epoch, val score and crop
settings.

Idempotent. The tag is ``makemodel-<source_run>`` from the sidecar. A release
whose asset already matches the model's SHA-256 is left alone, and a
half-finished one (a failed upload) is completed. A tag holding *different*
bytes is refused, never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from streettracker.control.introspect import model_sidecar_path, production_model_path

# A current-format UK plate (AB12CDE), with or without the space. Checked on
# every string in the checkpoint's non-tensor contents and in the sidecar.
_PLATE_RE = re.compile(r"\b[A-Z]{2}[0-9]{2}\s?[A-Z]{3}\b")
# Drive-letter or rooted paths; a dev-box path in the checkpoint would leak the
# box's layout and probably a username.
_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|/(?:home|Users|mnt|root)/)")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unsafe_strings(obj: Any, *, check_paths: bool) -> list[str]:
    """Plate-shaped (and, with ``check_paths``, path-like) strings anywhere in
    ``obj``'s non-tensor contents: dict keys and values, lists, tuples."""
    found: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, str):
            if _PLATE_RE.search(x.upper()) or (check_paths and _PATH_RE.match(x)):
                found.append(x)
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(k)
                walk(v)
        elif isinstance(x, (list, tuple, set)):
            for v in x:
                walk(v)

    walk(obj)
    return found


def _load_checkpoint(path: Path) -> Any:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _gh(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["gh", *args], capture_output=True, text=True)


def release_notes(sidecar: dict[str, Any], model: Path, sha: str, size: int) -> str:
    run = sidecar.get("source_run") or "?"
    acc = sidecar.get("val_make_top1")
    corpus = sidecar.get("trained_corpus") or {}
    lines = [
        f"Disaster-recovery copy of the production UK make classifier `{run}`, "
        f"promoted {sidecar.get('promoted_at', '?')} by the control panel. At "
        f"{size / 1e6:.1f} MB it's too big to commit to the repo, so it lives here.",
        "",
        "| Asset | Role |",
        "|---|---|",
        f"| `{model.name}` | UK make classifier, {sidecar.get('arch', '?')}"
        f"@{sidecar.get('input_size', '?')}, {sidecar.get('n_makes', '?')} makes |",
        f"| `{model_sidecar_path(model).name}` | Sidecar: arch, input size, crop mode, "
        "training-corpus fingerprint, head-to-head |",
        "",
    ]
    if corpus:
        lines.append(
            f"**Training data:** `{corpus.get('name')}`, {corpus.get('n_cars', '?')} cars, "
            f"{corpus.get('n_crops', '?')} crops, {corpus.get('n_makes', '?')} makes."
            + (f" Val make@1 {acc:.3f}." if isinstance(acc, float) else "")
        )
    h2h = sidecar.get("head_to_head") or {}
    delta = h2h.get("delta") or {}
    diff = delta.get("candidate_minus_production")
    if isinstance(diff, (int, float)):
        ci = delta.get("ci95") or []
        ci_txt = (
            f" (95 % CI {ci[0] * 100:+.1f}..{ci[1] * 100:+.1f})"
            if len(ci) == 2 and None not in ci
            else ""
        )
        lines.append(
            f"**Head-to-head recorded at promotion:** {diff * 100:+.1f} pp per track vs the "
            f"previous production model on {h2h.get('n_cars', '?')} held-out cars{ci_txt}."
        )
    lines += [
        "",
        f"**To restore:** put `{model.name}` in `src/streettracker/analysis/makemodel/models/`. "
        "Inference reads arch, input size and crop mode from the checkpoint.",
        "",
        f"SHA-256 of `{model.name}`: `{sha}`",
    ]
    return "\n".join(lines) + "\n"


def _release_assets(tag: str) -> tuple[bool, bool, dict[str, dict[str, Any]]] | None:
    """``(exists, is_draft, {asset name: asset})``, or ``None`` when gh fails
    for a reason other than "no such release"."""
    proc = _gh("release", "view", tag, "--json", "assets,isDraft")
    if proc.returncode != 0:
        if "release not found" in (proc.stderr + proc.stdout).lower():
            return False, False, {}
        print(f"[model-release] gh release view failed: {proc.stderr.strip()}", file=sys.stderr)
        return None
    data = json.loads(proc.stdout or "{}")
    return True, bool(data.get("isDraft")), {a["name"]: a for a in data.get("assets", [])}


def _asset_matches(asset: dict[str, Any] | None, sha: str, size: int) -> bool:
    if not asset or asset.get("state") != "uploaded" or asset.get("size") != size:
        return False
    digest = asset.get("digest")
    # GitHub reports digests for assets uploaded since 2025; size is the
    # fallback check for any that predate it.
    return digest is None or digest == f"sha256:{sha}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="streettracker model-release",
        description="Publish the production make model as a (public) GitHub Release.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="checkpoint to publish (default: the production makemodel_b0.pt)",
    )
    parser.add_argument("--tag", default=None, help="release tag (default makemodel-<source_run>)")
    parser.add_argument("--target", default="main", help="branch or commit the tag points at")
    parser.add_argument(
        "--dry-run", action="store_true", help="run every check, print the notes, upload nothing"
    )
    args = parser.parse_args(argv)

    model: Path = args.model or production_model_path()
    sidecar_path = model_sidecar_path(model)
    if not model.is_file():
        print(f"[model-release] no model at {model}", file=sys.stderr)
        return 1
    if not sidecar_path.is_file():
        print(f"[model-release] no sidecar at {sidecar_path}", file=sys.stderr)
        return 1
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    run = sidecar.get("source_run")
    tag = args.tag or (f"makemodel-{run}" if run else None)
    if not tag:
        print("[model-release] the sidecar names no source_run; pass --tag", file=sys.stderr)
        return 1

    size = model.stat().st_size
    sha = file_sha256(model)
    print(f"[model-release] {model.name}: {size / 1e6:.1f} MB, sha256 {sha}")
    print(f"[model-release] tag {tag} (run {run or '?'})")

    # The repo is public: nothing plate-shaped (or path-like) may ride along.
    bad = unsafe_strings(_load_checkpoint(model), check_paths=True)
    bad += unsafe_strings(sidecar, check_paths=False)
    if bad:
        shown = ", ".join(repr(s[:40]) for s in bad[:5])
        print(
            f"[model-release] REFUSED: {len(bad)} plate-shaped or path-like string(s) in the "
            f"checkpoint/sidecar ({shown}); the repo is public",
            file=sys.stderr,
        )
        return 1
    print("[model-release] contents checked: no plate-shaped strings or paths")

    notes = release_notes(sidecar, model, sha, size)
    if args.dry_run:
        print(notes)
        print("[model-release] dry run: nothing uploaded")
        return 0

    state = _release_assets(tag)
    if state is None:
        return 1
    exists, is_draft, assets = state
    if exists:
        have = assets.get(model.name)
        if have and have.get("state") == "uploaded" and not _asset_matches(have, sha, size):
            print(
                f"[model-release] REFUSED: {tag} already holds a different {model.name}; "
                "pass a new --tag",
                file=sys.stderr,
            )
            return 1
        # A previous attempt that stopped part-way (gh leaves a draft when an
        # upload fails) is finished: upload what is missing, then publish.
        missing = [str(model)] if not _asset_matches(have, sha, size) else []
        side = assets.get(sidecar_path.name)
        if not side or side.get("state") != "uploaded":
            missing.append(str(sidecar_path))
        if not missing and not is_draft:
            print(f"[model-release] {tag} already holds this model; nothing to do")
            return 0
        proc = subprocess.CompletedProcess([], 0, "", "")
        if missing:
            print(f"[model-release] completing {tag}: uploading {len(missing)} file(s)")
            proc = _gh("release", "upload", tag, *missing, "--clobber")
        if proc.returncode == 0 and is_draft:
            proc = _gh("release", "edit", tag, "--draft=false")
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(notes)
            notes_file = f.name
        try:
            title = f"Production make model — {run or tag}"
            print(f"[model-release] creating {tag}, uploading {size / 1e6:.0f} MB", flush=True)
            proc = _gh(
                "release", "create", tag, str(model), str(sidecar_path),
                "--target", args.target, "--title", title, "--notes-file", notes_file,
            )  # fmt: skip
        finally:
            Path(notes_file).unlink(missing_ok=True)
    if proc.returncode != 0:
        print(f"[model-release] gh failed: {(proc.stderr or proc.stdout).strip()}", file=sys.stderr)
        return 1
    if proc.stdout.strip():
        print(f"[model-release] {proc.stdout.strip()}")  # the release URL

    # Trust the upload only once GitHub reports the same bytes.
    state = _release_assets(tag)
    if state is None or not _asset_matches(state[2].get(model.name), sha, size) or state[1]:
        print(
            f"[model-release] uploaded, but {tag} does not report a matching, published "
            f"{model.name}; check the release before relying on it",
            file=sys.stderr,
        )
        return 1
    print(f"[model-release] published {tag}: {model.name} verified (sha256 matches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
