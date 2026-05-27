"""consensus_plate(): confidence-weighted character-vote consensus.

Covers the cases that matter for the R→L direction on this camera
where individual reads are low-conf but agree across snaps:
1. unanimous agreement across reads -> high consensus conf
2. one disagreeing read overruled by majority -> consensus still right
3. length mismatch -> minority-length reads dropped
4. single read -> consensus = that read
5. no eligible reads -> None
6. tie at a position -> deterministic tie-break
"""

from __future__ import annotations

from streettracker.analysis.alpr.consensus import (
    consensus_by_track,
    consensus_plate,
)


def _read(track_id, snap_index, ocr_text, ocr_conf, pipeline="preferred"):
    return {
        "track_id": track_id,
        "snap_index": snap_index,
        "image": f"vehicle_{track_id}_main_{snap_index}.jpg",
        "ocr_text": ocr_text,
        "ocr_conf": ocr_conf,
        "pipeline": pipeline,
    }


def test_consensus_returns_none_for_empty_input() -> None:
    assert consensus_plate([]) is None


def test_consensus_returns_none_when_all_reads_below_floor() -> None:
    reads = [
        _read(1, 1, "ABC123", 0.10),
        _read(1, 2, "ABC123", 0.20),
    ]
    assert consensus_plate(reads, min_input_conf=0.30) is None


def test_consensus_single_read_passes_through() -> None:
    reads = [_read(1, 1, "AB12CDE", 0.55)]
    out = consensus_plate(reads)
    assert out is not None
    assert out["ocr_text"] == "AB12CDE"
    assert out["ocr_conf"] == 1.0  # single read = 100% vote at every position
    assert out["n_reads_used"] == 1
    assert out["n_reads_total"] == 1
    assert out["best_snap_index"] == 1


def test_consensus_unanimous_yields_high_confidence() -> None:
    """Four reads, same text -> consensus conf = 1.0 (every position
    won 100% of the weight at that position)."""
    reads = [
        _read(7, 1, "AB12CDE", 0.55),
        _read(7, 2, "AB12CDE", 0.60),
        _read(7, 3, "AB12CDE", 0.62),
        _read(7, 4, "AB12CDE", 0.50),
    ]
    out = consensus_plate(reads)
    assert out is not None
    assert out["ocr_text"] == "AB12CDE"
    assert out["ocr_conf"] == 1.0
    assert out["n_reads_used"] == 4
    # best_snap_index is the highest-conf input matching the consensus.
    assert out["best_snap_index"] == 3


def test_consensus_outvotes_single_disagreement() -> None:
    """Three reads of ``AB12CDE`` at 0.55, one read of ``AB12XDE`` at
    0.60: the 5th character votes 3*0.55 = 1.65 for 'C' vs 0.60 for 'X'.
    'C' wins despite being individually lower-conf per-read."""
    reads = [
        _read(8, 1, "AB12CDE", 0.55),
        _read(8, 2, "AB12CDE", 0.55),
        _read(8, 3, "AB12CDE", 0.55),
        _read(8, 4, "AB12XDE", 0.60),
    ]
    out = consensus_plate(reads)
    assert out is not None
    assert out["ocr_text"] == "AB12CDE"
    # 6 positions agreed (vote share 1.0), 1 position split (vote
    # share 1.65 / 2.25 = 0.733). Geo mean of those = ~ 0.956 ** (1/7).
    assert 0.95 < out["ocr_conf"] < 1.0


def test_consensus_drops_minority_length_reads() -> None:
    """Two reads of length 7 (the mode), one read of length 6 -- the
    length-6 read is excluded so it can't pollute the per-position
    vote."""
    reads = [
        _read(9, 1, "AB12CDE", 0.55),
        _read(9, 2, "AB12CDE", 0.55),
        _read(9, 3, "ABCDEF", 0.95),    # different length -- dropped
    ]
    out = consensus_plate(reads)
    assert out is not None
    assert out["ocr_text"] == "AB12CDE"
    assert out["n_reads_used"] == 2  # length-6 read not counted
    assert out["n_reads_total"] == 3


def test_consensus_tie_break_is_deterministic() -> None:
    """Two reads, both at the same confidence, disagreeing on one
    char. Sum at the disputed position is equal -- the tie-break
    must yield a stable, defined choice. The secondary key is
    ``-ord(char)``, so the character with the *lower* ordinal
    (alphabetically earlier) wins -- 'A' beats 'B'. The specific
    direction doesn't matter for correctness; determinism does."""
    reads = [
        _read(10, 1, "A", 0.50),
        _read(10, 2, "B", 0.50),
    ]
    out = consensus_plate(reads)
    assert out is not None
    assert out["ocr_text"] == "A"
    assert out["ocr_conf"] == 0.5  # tied vote -> 50% share


def test_consensus_by_track_groups_per_track_and_filters_pipeline() -> None:
    """consensus_by_track() must:
    - process only the named pipeline
    - return one consensus per track that has eligible reads
    - skip tracks with no eligible reads
    """
    records = [
        _read(1, 1, "PLATE1", 0.55, pipeline="preferred"),
        _read(1, 2, "PLATE1", 0.55, pipeline="preferred"),
        _read(2, 1, "PLATE2", 0.10, pipeline="preferred"),  # below floor
        _read(3, 1, "BESPOKE", 0.99, pipeline="bespoke"),   # wrong pipeline
        _read(4, 1, "PLATE4", 0.40, pipeline="preferred"),
    ]
    out = consensus_by_track(records, pipeline="preferred")
    assert set(out.keys()) == {1, 4}
    assert out[1]["ocr_text"] == "PLATE1"
    assert out[4]["ocr_text"] == "PLATE4"


def test_consensus_best_image_falls_back_when_no_input_matches_text() -> None:
    """Composite consensus: no individual read agrees on every char.
    best_image should still be populated -- pick the highest-conf
    same-length read as the visual proxy."""
    reads = [
        _read(11, 1, "ABXDEF", 0.50),
        _read(11, 2, "ABCXEF", 0.50),
        _read(11, 3, "ABCDXF", 0.50),
        _read(11, 4, "ABCDEX", 0.60),  # highest conf
    ]
    out = consensus_plate(reads)
    assert out is not None
    # Each position has 3 votes of 0.5 and 1 vote of 0.6 for the
    # winner; consensus assembles "ABCDEF" but no single input read
    # said that. best_image falls back to the highest-conf same-len.
    assert out["ocr_text"] == "ABCDEF"
    assert out["best_snap_index"] == 4
