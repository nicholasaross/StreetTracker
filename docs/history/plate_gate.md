# Plate confidence, the plate gate and the 2026-09-28 handoff

> Moved from `CLAUDE.md` on 2026-11-11. The text is unchanged apart from link targets and
> heading levels. It is a dated record, written as the work happened: later entries correct
> earlier ones, and passages marked SUPERSEDED, CORRECTION or UPDATE say where. "Above" and
> "below" refer to the old file's layout. For what is true now, read "Current state" in
> [`CLAUDE.md`](../../CLAUDE.md).

## The data-integrity review and the OCR-confidence fix (2026-09-27 to 2026-10-04)

**⮕ READ FIRST (2026-09-27): [`docs/data_integrity_review.md`](../data_integrity_review.md)**
— the follow-up to the crop-contamination finding: 15 unverified-association
risks with a phased experiment plan (Phase 0 = read-only checks on existing
`output/`). Until those land, the ALPR rates below are *canonical-shape* rates,
not verified reads.

**⮕ OCR confidence bug FIXED IN CODE 2026-09-28 (R1) — existing sessions still
need re-scoring.** Under the locked fast-plate-ocr 1.1.0, `ocr_conf` was the
most confident OCR slot (0.96–0.998 for every read, garbage included), so every
`conf ≥ 0.9` gate (DVSA harvest, `vehicles`, the canonical-rate headline) passed
everything. It is now the probability of the **weakest decoded character**
(`alpr/preferred._unpack_ocr_output`; per-character probs persisted as
`ocr_char_probs`). Fix existing sessions with the panel's **Re-score plate
confidence** playbook (sessions badged **Plates v2**): `alpr-rescore` re-OCRs
the saved plate crops (no detection re-run) → `dvsa-label` → `dvsa-apply` →
`vehicles` → showcase refresh. Expect fewer DVSA-labelled tracks: reads with
one uncertain character no longer pass. **One shared plate gate** (2026-09-28):
`configs/alpr.json` `{"plate_conf_threshold": X}` (gitignored; template
`configs/alpr.example.json`) is read by dvsa-label, `vehicles` + the showcase,
the stats page's fastest-car plates and alpr-rescore's summary
(`analysis/alpr/base.resolve_plate_conf_threshold`: `--conf-threshold` /
`--conf` flag > file > default 0.9; a malformed file is an error, never a silent
fallback). **Calibrated 2026-09-28 to 0.90** (handoff step 3 below); to
re-calibrate after more sessions land, run
`uv run python .claude/ocr_conf_calibration.py` (label-free: DVSA not-found rate
on plates old enough to have an MOT, DVSA-vs-CNN colour mismatch, and snap
agreement, per confidence group and per cut-off; it marks the current gate) and
set the value where the rates level off. Then re-run `dvsa-label` →
`dvsa-apply` → `vehicles` per session and refresh the showcase; dvsa-label's
track clearing is reversible (labels are cached), so this costs no API calls,
and `_dvsa_labels.json` records the `conf_threshold` it used. Rebuild the
make/colour/body corpus afterwards so training labels drop the misreads.
**⮕ 2026-10-04: the gate is now pluggable** (`analysis/alpr/gate.py`,
`PlateGate`; `configs/alpr.json` `"plate_gate": "conf" | "combined"`). The
**combined** gate (review E1.4) counts a read at min-char conf >= 0.80 that
another snap of the track agrees with (`n_agree` on the rollup's best read) or
that >= 2 tracks anywhere share (cross-session support from every rollup);
plate colour runs upstream (`colour_suspect` reads never become a best read).
Apply/re-apply with the panel's **platecheck** playbook.

## The 2026-09-28 handoff

**⮕ NEXT SESSION — start here (handoff written 2026-09-28; steps 1-5 done
by 2026-10-04, so start at step 6, revised 2026-10-04).** PR #110 (the
review doc, the R1 OCR-confidence fix + `alpr-rescore` + panel **rescore**
playbook, `.claude/ocr_conf_calibration.py`, the shared plate gate
`configs/alpr.json`) is **merged**. Steps 1-5 were all dev-box analysis code;
step 6's E1.5(c) is the one Orin deploy. In order:

1. **Land the branch.** Done (PR #110 merged; it changed no dependencies, so
   no `uv sync` was needed).
2. **Re-score.** Done 2026-09-28 (panel *Re-score plate confidence*, 185/185
   steps, 108 min): all 46 ALPR'd sessions carry the min-character
   confidence (Plates v3).
3. **Calibrate.** Done 2026-09-28: **gate = 0.90**, written to
   `configs/alpr.json` (its `_why` key holds the evidence; full numbers in
   `.claude/ocr_calibration.json`). Not-on-DVSA-register rate by
   min-confidence group over 51,953 UK-shaped best reads: 0.95-0.98 1.5 %,
   0.90-0.95 2.5 %, 0.85-0.90 4.2 %, 0.80-0.85 7.2 %, 0.70-0.80 12.2 %. The
   rates never level off; below 0.90 a group's rate is ~3x the top group's.
   0.90 keeps 66 % of tracks at 2.0 % not-on-register, while 0.95 keeps only
   32 % at 1.5 %, halving the labelled set. The colour-mismatch check is too
   small to decide (n < 30 in most groups once training cars are excluded).
   **Follow-up (code): snap agreement beats confidence.** 0.85-0.90 reads
   that another snap agrees with are 1.0 % not-on-register; 0.90-0.95 reads
   that no snap agrees with are 7.0 %. A gate that combines agreement with
   confidence would beat any single threshold.
4. **Apply the gate.** Done 2026-09-28 with the new panel **relabel**
   playbook (*Re-apply plate gate (all sessions)*, PR #111): 32,150 car
   tracks DVSA-labelled, **12,672 stale per-track labels cleared** by the
   R13 `dvsa-apply` fix, 0 new DVSA lookups, showcase at 4,058 cars.
   Per-track labels match `_dvsa_labels.json` exactly in all 46 sessions.
   Re-run the playbook whenever `configs/alpr.json` changes (it costs no
   API calls; rescore skips sessions it has already done, so it can't).
5. **Retrain on the cleaner labels.** Done 2026-10-03. The full account moved to
   [makemodel.md](makemodel.md#retrains-on-gate-cleaned-labels-2026-09-29-to-2026-10-03).
6. **Then the review's plan, revised 2026-10-04**
   (`docs/data_integrity_review.md` §4, "Revision (2026-10-04)" has the
   reasoning). In order:
   1. **Small code fixes. Done 2026-10-04.** The `/stats` make chart counts
      only plates with current `track_ids` (E0.2: 10,071 of 29,053 label
      rows were orphans; the top-12 makes halved to 3,196 cars, Toyota
      +2.3 pp). HSV-sourced colours show as "unknown" in the mix and the
      fastest-by-colour board (E0.8: HSV coloured only 5.5 % of car tracks,
      but 78 % of them black or blue). E0.6 found no time-base anomaly (Orin
      `Europe/London`, NTP synced; `.claude/phase0_checks.py --checks e06`),
      and the BST repeat hour needs no fix: each pass counts once, and the
      repeated hour just holds two clock-hours (E3.5 note). The showcase was
      restarted on it 2026-10-04.
   2. **E1.5(c): persist per-snap fire/done timestamps and a thinned
      per-track trajectory.** A code-only Orin deploy, in the
      [schema-additive order](../../CLAUDE.md#schema-additive-config-changes). It only helps
      sessions recorded after it ships, and the Orin deletes 4K snaps after
      7 days, so land it early. **Deployed 2026-10-04 09:52** (PR #123; the
      Orin went from #89 to `5f4ceb4`, whose only runtime change is this;
      rollback `git checkout 90e10df` + restart). `session_20261004_095219`
      is the first session with `TrackRecord.main_snap_fire_unix` /
      `main_snap_done_unix` and `{session}_trajectories.jsonl` (checked on
      its first track: fire→done 703 ms, 32 trajectory rows). **Pull the
      closed `session_20261001_190826` before ~2026-10-08**, when its first
      snaps age out.
   3. **Phase 0 script. Done 2026-10-04** (`.claude/phase0_checks.py`;
      results in each experiment's row in the review, local report
      `.claude/phase0_report.{json,txt}`, git-ignored because it holds
      plates). E0.3: IR mode (no inference) took **41.7 % of September's
      session time** (new finding R16). E0.5: **R2 is real**, two cars passing
      share one plate on 1.3 % of gated read tracks. E0.9: **R3 is real**,
      ~40 % of labels from a plate read on one track describe another car,
      and cross-track support separates labels far better than confidence.
      E0.4: 11 % of the "fresh" test cars are misreads of regulars. E0.7:
      most "joggers" are near-pavement walkers.
      - **3a. New: low light (review R16 + its update).** The camera has
        been in **forced colour (`dayNight: Color`) since 2026-09-28**, so the
        IR gap is mid-Aug to 28 Sep only. The live problem is that **rush-hour
        plate reads collapse as the days shorten**: gated read rate at 17:00
        was 54.5 % in July, 15.0 % in September and 5.2 % on 1 Oct; at 08:00,
        43 % → 11 %. Forced colour doesn't help.
        **Capture analysed 2026-10-05** (review R16 "Results"; `~/ir_test`
        on the Orin, `.claude/ir_capture.py --windows`: sub-stream + 4K every
        2 s at 18:30, 21:30 and 07:00 on 4-5 Oct):
        - **Nights are quiet, not missed.** Mon-Thu car tracks/h in forced
          colour (28 Sep-1 Oct, dark) match summer daylight: 20 h 43.5 vs
          49.6, 22 h 1.8 vs 1.5, 23 h 2.2 vs 2.1 (the IR weeks: 20 h 17.0,
          blind). Dark tracks have lower confidence (0.77 vs 0.84) and are
          visible ~2 s less; 21 h (8.2 vs 12.2) may be missing a few cars.
        - **The capture itself was weak: 4 Oct was a Sunday**, the quietest
          evening. 21:30-22:00 had no traffic at all (live, offline YOLO down
          to conf 0.05, motion count and 4K snaps all empty), 18:30 no cars.
          Run future captures on weekdays. The **07:00 Monday** window matched
          the live tracker 1:1 (6 cars, 3 people, a dog; no motion
          unexplained), but its 6 cars gave 11 views in the periodic 4K snaps,
          2 plate detections and 0 reads.
        - **Low-light plate loss = blur + R→L front plates not found.** Same
          week, per car: L→R plates are still found at 07-08/18-19 h
          (81-100 %) but none read (sharpness 2-4× below day while brighter);
          R→L plate detection drops to 17-41 % (day 91 %), most likely
          headlight glare around the front plate. Gated per-car reads: day
          47.7 %, 17-18 h 7.7 %, 18-19 h 0 %, 07-08 h 0 %, 08-09 h 3.2 %.
        - **The low-light choice is the operator's** (it changes the camera):
          counting needs nothing (forced colour tracks at night); the plate
          lever is a faster shutter scheduled for 07-09 h and 17-19 h only
          (E2.8; a cap all day cost ~23 pp in daylight), via a scheduled
          `SetIsp` (admin account). IR-mode inference (option a) is now only
          a plate-read option and keeps the long exposures.
        - Scripts: `.claude/lowlight_{counts,plates}.py` (tables from
          `output/`), `.claude/ir_capture_{detect,report,plates}.py` (a pulled
          capture; outputs stay beside it, never in the repo).
   4. **E1.3 plate-colour/direction check, and the combined gate. Live
      2026-10-04.** `configs/alpr.json` is `"plate_gate": "combined"`, and the
      panel's **platecheck** playbook ran on all 49 sessions (197/197 steps,
      ~24 min; every session now badges Plates v4, every harvest records the
      combined gate). **Car journeys with a DVSA label: 33,445 (35.9 %) →
      39,499 (41.8 %)**; showcase 4,164 cars. Harvests attribute some non-car
      tracks (vans as "truck", buses, bicycles) and tracks with no `data.json`
      record; `dvsa-apply` writes car records only, so those never reach
      `data.json` (expected; review R14). Re-run platecheck (or relabel) after
      any change to `configs/alpr.json`. Simulated beforehand: tracks passing
      36,078 → 41,923 (+16.2 %), 581 new DVSA lookups. Classifier
      `analysis/alpr/plate_colour.py` (R→L shows the white front plate, L→R
      the yellow rear). **2.5 % of gated best reads carry a plate whose colour
      contradicts the direction** (L→R 3.9 %, mostly oncoming cars' front
      plates), confidence and support can't see it, and colour resolves all
      230 of E0.5's concurrent collisions. Gate comparison
      (`.claude/plate_gate_rules.py`, review E1.4): **gate H = (another snap
      agrees, or ≥ 2 tracks read the plate) and conf ≥ 0.80 and plate colour
      not wrong**. It labels 15.5 % more tracks than today's conf ≥ 0.90, with
      not-on-register 2.1 → 1.4 %, unseen colour mismatch 4.1 → 3.6 % and
      misattributed plates 2.5 % → 0. Built as: colour-inconsistent reads are
      `colour_suspect` (the best read falls back), and H is the shared gate's
      `combined` mode (dvsa-label, vehicles, showcase, stats).
   5. **E1.2 audit set** (extend `.claude/triage_rl.py`), stratified to
      over-represent what E0.5 and E1.3 flag; then calibrate E1.3 and the
      combined gate against it. It is the only check on whether a plate
      belongs to the tracked car: the DVSA register can't see that.
   6. **E2.6** (plate-blind test first: the heads were promoted on plated
      cars, but `/stats` uses them on unplated ones), E2.1 capture-band
      curves, E2.3 ghost mask on/off, E2.7 colour augmentation. E1.5(a)/(b)
      can run at any point.
   7. **E3.1 hand count, then the night A/B (E2.8).** Reframed by R16:
      after the IR switch nothing is recorded, so the A/B depends on step
      3a's choice.
   8. **Retrain (E2.5)** once the E1.4 filter and cluster-aware
      `makemodel-compare` exclusion exist (E0.4: the fresh set holds
      misreads of regulars) and fresh cars have built up.
   Also from Phase 0: show joggers as "fast-moving person tracks" until E3.2
   (E0.7).

   The production `uk_make_0929_b6` is backed up as GitHub Release
   `models-2026-10-26` (2026-10-26; see
   [Dev-box disaster recovery](../deployment.md#dev-box-disaster-recovery)).

**Cloud sessions have no `output/`**, so they can only do code work there:
- step 6.1's fixes;
- E1.5(c)'s runtime change (the Orin deploy itself runs from the dev box);
- the R16 runtime option (an IR-mode inference switch);
- the E1.3 plate-colour script and the combined gate;
- cluster-aware exclusion in `makemodel-compare`.

Anything that reads session data runs on the dev box, and the operator
pastes the output back.
