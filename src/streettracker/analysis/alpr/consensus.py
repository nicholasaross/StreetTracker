"""Multi-frame plate-read consensus via confidence-weighted character voting.

For tracks where the detector returns multiple low-confidence reads
of the same physical plate, the per-image best-of-N rollup
(``streettracker alpr-run``'s default per-track output) is often
not the right answer -- a single 0.55-conf read for ``AB12CDE`` is
less trustworthy than four 0.55-conf reads that all agree on
``AB12CDE`` at every character position.

The consensus primitive here:

1. Pools all preferred-pipeline reads for a track at OCR confidence
   above a low floor (default 0.30 -- well below the 0.90
   "high-conf" floor used elsewhere).
2. Picks the most common plate length (mode) among them. Reads of
   other lengths are dropped from the vote -- mixing lengths would
   misalign character positions.
3. For each character position, sums per-character OCR confidence
   across all length-matching reads. The character with the highest
   summed confidence wins that position.
4. Reports an aggregate consensus confidence: the geometric mean of
   the per-position vote shares (winner's summed weight / total
   weight at that position). This is bounded in ``[0, 1]`` and reads
   close to 1.0 when reads all agree, dropping toward 1/A (A =
   alphabet size) when reads disagree uniformly.

Useful primarily for the R→L direction on this camera, where
per-image clean-read rate is 29 % but most tracks see 3-5 snaps in
the band -- four 0.6-conf agreeing reads compose into a
high-confidence consensus that no individual snap reaches.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

MIN_INPUT_CONF = 0.30


def consensus_plate(
    reads: list[dict[str, Any]],
    *,
    min_input_conf: float = MIN_INPUT_CONF,
) -> dict[str, Any] | None:
    """Compute a confidence-weighted consensus plate over a track's reads.

    ``reads`` is the per-image ``alpr.json`` record list filtered to
    one pipeline + one track. Each must carry at least
    ``ocr_text`` / ``ocr_conf`` / ``snap_index``; other keys are
    passed through into the returned dict for the winning
    contributing snap.

    Returns ``None`` if no read has text + ``ocr_conf >= min_input_conf``.
    Otherwise returns:

    ::

        {
            "ocr_text":      "AB12CDE",
            "ocr_conf":      0.92,        # consensus aggregate (geo-mean)
            "n_reads_used":  4,           # reads at the consensus length
            "n_reads_total": 5,           # reads passing min_input_conf
            "best_snap_index": 3,         # snap_index of the highest-conf
                                          # input matching the consensus
            "best_image":    "vehicle_42_main_3.jpg",
        }
    """
    eligible = [
        r for r in reads
        if r.get("ocr_text")
        and (r.get("ocr_conf") or 0.0) >= min_input_conf
    ]
    if not eligible:
        return None

    # Length filter: keep only reads at the mode length. Ties are
    # broken by total confidence at that length (favours the slate of
    # reads that, collectively, are more confident).
    length_conf: dict[int, float] = defaultdict(float)
    for r in eligible:
        length_conf[len(r["ocr_text"])] += r["ocr_conf"]
    target_len = max(length_conf, key=lambda L: (length_conf[L], L))

    same_len = [r for r in eligible if len(r["ocr_text"]) == target_len]

    # Per-position character vote, weighted by ocr_conf.
    consensus_chars: list[str] = []
    vote_shares: list[float] = []
    for i in range(target_len):
        char_weight: dict[str, float] = defaultdict(float)
        for r in same_len:
            char_weight[r["ocr_text"][i]] += r["ocr_conf"]
        total = sum(char_weight.values())
        if total <= 0:
            return None
        # max with deterministic tie-break by character ordinal so the
        # output is stable across CPython runs.
        winner_char = max(
            char_weight.items(), key=lambda kv: (kv[1], -ord(kv[0]))
        )[0]
        consensus_chars.append(winner_char)
        vote_shares.append(char_weight[winner_char] / total)

    consensus_text = "".join(consensus_chars)
    # Geometric mean of per-position vote shares. Falls to 1/A when
    # votes are uniform; rises to 1.0 when reads agree everywhere.
    log_sum = 0.0
    for share in vote_shares:
        # share > 0 guaranteed above.
        from math import log
        log_sum += log(share)
    from math import exp
    consensus_conf = exp(log_sum / target_len)

    # Find the highest-conf input read that MATCHES the consensus
    # text -- surfaces the snap whose crop you'd want to render for
    # visual confirmation.
    matching = [r for r in same_len if r["ocr_text"] == consensus_text]
    if matching:
        best = max(matching, key=lambda r: r["ocr_conf"] or 0.0)
        best_snap = best.get("snap_index")
        best_image = best.get("image")
    else:
        # Composite consensus -- no single read agrees on every char.
        # Pick the highest-conf same-length read as a visual proxy.
        best = max(same_len, key=lambda r: r["ocr_conf"] or 0.0)
        best_snap = best.get("snap_index")
        best_image = best.get("image")

    return {
        "ocr_text": consensus_text,
        "ocr_conf": round(consensus_conf, 6),
        "n_reads_used": len(same_len),
        "n_reads_total": len(eligible),
        "best_snap_index": best_snap,
        "best_image": best_image,
    }


def consensus_by_track(
    records: list[dict[str, Any]],
    pipeline: str = "preferred",
    *,
    min_input_conf: float = MIN_INPUT_CONF,
) -> dict[int, dict[str, Any]]:
    """Run ``consensus_plate`` over every track in an alpr.json record
    list. Filters to one pipeline (default ``"preferred"`` since
    Step 6 established the bespoke pipeline contributes no useful
    signal). Returns ``{track_id: consensus_dict}`` for tracks where
    consensus succeeds; tracks with no reads above the floor are
    absent from the result.
    """
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("pipeline") != pipeline:
            continue
        by_track[r["track_id"]].append(r)

    out: dict[int, dict[str, Any]] = {}
    for tid, reads in by_track.items():
        c = consensus_plate(reads, min_input_conf=min_input_conf)
        if c is not None:
            out[tid] = c
    return out
