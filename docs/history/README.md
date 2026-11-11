# Project history

Dated accounts of how StreetTracker reached its current state: what was tried, what was measured
and what each result replaced. [`CLAUDE.md`](../../CLAUDE.md) holds only the current position and
the standing rules.

| File | Covers |
| --- | --- |
| [`anpr_tuning.md`](anpr_tuning.md) | Plate-read tuning, May to October 2026: the status table, steps 1-16, the per-direction band verdicts, the R→L "geometry cap" and its reversal, and the night-capture findings. |
| [`makemodel.md`](makemodel.md) | The make, colour and body-type classifiers: CompCars, the UK-native classifier, the resolution ladder, crop contamination, and every retrain and promotion up to 2026-10-03. |
| [`plate_gate.md`](plate_gate.md) | The OCR-confidence fix, the gate calibration, the combined gate, and the 2026-09-28 handoff. Step 6 of that handoff is the data-integrity plan, with the Phase 0 results and the low-light capture. |
| [`project.md`](project.md) | The 2026-06-13 checkpoint, what was deployed to the Orin and when, the people side, the 2026-07-07 next-steps list, migration and cutover, and the summary-page stall. |

## Reading these files

- They were moved out of `CLAUDE.md` on 2026-11-11. The text is unchanged apart from link targets
  and heading levels.
- Later entries correct earlier ones. Passages marked SUPERSEDED, CORRECTION or UPDATE say where,
  so read to the end of a topic before quoting a number from it.
- "Above" and "below" refer to the old file's layout, not to these files.
- Anything written before the move that cites "CLAUDE.md" for a dated result (older commits,
  [`docs/data_integrity_review.md`](../data_integrity_review.md), some `.claude/` scripts) means
  these files.

## Adding to them

When a result lands, edit "Current state" in `CLAUDE.md` so it says what is true now, and append
the dated account here: the date, the numbers, the method and what it replaces. Add a new entry
instead of rewriting an old one.
