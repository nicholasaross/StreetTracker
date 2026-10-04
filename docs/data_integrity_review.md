# Data-integrity review and experiment plan (2026-09-27)

This follows up the 2026-09-24 crop-contamination finding: the make, colour and
body-type models were trained and scored on crops that mostly missed the car the
label described. This review asks where else StreetTracker **assumes a link
between two pieces of data without checking it**. It covers the whole path:
capture → ALPR → DVSA labels → training corpus → evaluation → analytics.

Every finding below cites the code that makes the assumption. Each one is marked
**confirmed** (verified from the code, and reproduced where possible) or
**hypothesis** (plausible, unmeasured; an experiment is specified). No experiment
had been run on real session data when this was written: this repo checkout has
no `output/`, so every experiment is written to run on the dev box. What has run
since, and how it changed the plan, is in [§4's 2026-10-04
revision](#revision-2026-10-04).

## Summary

1. **Confirmed defect, fixed in code 2026-09-28: the OCR confidence was
   always near 1.0.** `analysis/alpr/preferred.py:168` took
   `np.max(char_probs, axis=-1)`. Under the locked fast-plate-ocr 1.1.0,
   `char_probs` is already the per-slot max (shape `(max_plate_slots,)`), so
   this returned the single most confident slot. On the real model that is
   0.96–0.998 for every read, garbage included: a smeared misread scored 0.998,
   *higher* than the correct read's 0.983. So every `conf ≥ 0.9` gate filtered
   nothing: DVSA lookups, `vehicles`, the "canonical read rate" headline, and
   the E1 re-score. The per-track "best read" was the read with the single
   most confident slot, which says nothing about the plate. **Existing
   sessions still carry the old values until re-scored** (see E1.1).
2. **Hypothesis, high risk: the ALPR headline counts plate-shaped strings, not
   correct plates.** "R→L 69.7 % / L→R 66.8 %" means "the track produced a
   UK-shaped string". Nothing checks that the string is right or that the plate
   is on the tracked car. The fullframe path tries two candidate vehicles and
   keeps the higher *detector* confidence, so an oncoming car's plate can win.
3. **Hypothesis, high risk: DVSA labels can describe a different real car.** An
   OCR slip on a dense registration range often lands on a real car of another
   make or colour. With no working confidence gate (item 1), those labels flow
   into the make, colour and body-type corpora and the stats page. The corpus
   also keys car identity on the exact plate string, so one physical car can
   appear in both train and val.
4. **Hypothesis, high risk: the sub-stream and 4K geometry and timing are
   assumed, never measured.** The ~510 px stale-bbox offset was a symptom of
   this. The live snap bands, the door-approach bands, the road polygon on the
   sub-stream, the ghost mask and the door zone all rest on the same unmeasured
   mapping.
5. **Confirmed: live config and past conclusions were measured through
   instruments now known to be broken.** Examples: the live R→L capture band
   `[0.10, 0.20]` (its recorded rationale is the 2026-06-11 analysis that was
   later declared an artifact), the ghost mask, and the Step 13b "consensus
   loses by 43 pp" result.
6. **Confirmed gap: the CNN heads are scored on plated cars but used on
   unplated ones.** Evaluation uses plated, mostly daytime, near cars. The CNN
   is only *used* where DVSA is absent: unplated, night, far and oblique cars,
   and cars under three years old, which DVSA never labels.
7. **Hypothesis, analytics: counts are read as rates.** Stats have no
   observation-time denominator (IR periods, gaps, partial days). The tracker
   has never been checked against a hand count. Speed and the jogger split use
   one global metres-per-pixel factor. The colour chart silently falls back to
   the HSV voter that CLAUDE.md already calls rubbish.

The recurring pattern: **a proxy is reported as the target, and the proxy looks
internally consistent**. The fix pattern is the one the crop audit used: a
small, human-verified ground-truth set, plus cheap automated consistency checks
run against it.

---

## 1. Failure shapes to look for

| Shape | Crop-contamination instance | Other instances below |
| --- | --- | --- |
| A region of pixels is assumed to be the tracked object | stale fire-time bbox crop | R2 (fullframe candidate pick), R4 (sub↔4K mapping) |
| A field's meaning silently changed | — | R1 (`ocr_conf` after a library API change) |
| A proxy metric is reported as the target | "val make@1" on crops of empty road | R2 (canonical-shape rate), R11 (grouped-colour on confident passes) |
| The evaluation population ≠ the deployment population | plate-anchored val vs hint inference | R6 |
| A label is joined by key without checking the image | DVSA make ↔ stale crop | R3, R7 |
| A decision is taken on a now-suspect measurement | the "R→L geometry cap" | R5 |
| A count is treated as a rate | — | R8, R9 |
| A physical quantity is inferred through an uncalibrated transform | — | R10 |

---

## 2. Findings register

Severity reflects the blast radius if the finding holds. Confidence is how sure
the review is that it holds.

| ID | Finding | Severity | Confidence | Contaminates |
| --- | --- | --- | --- | --- |
| R1 | `ocr_conf` was always near 1.0 (fixed in code 2026-09-28; existing sessions need re-scoring) | **Critical** | **Confirmed** | every plate gate, per-track best read, DVSA harvest, ALPR headline, consensus |
| R2 | plate→track attribution never verified; headline = shape rate | **High** | Hypothesis | ALPR headline, R→L verdict, DVSA labels, showcase regulars |
| R3 | DVSA labels from misreads describe a different real car | **High** | Hypothesis (mechanism confirmed) | make/colour/body corpora + val labels, stats make/colour/body mix |
| R4 | sub-stream ↔ 4K spatial + temporal registration unmeasured | **High** | Hypothesis (symptom confirmed) | snap gating, hints, fullframe pick, ghost mask, door zone, band tuning |
| R5 | live config + recorded conclusions measured through broken instruments | **High** | **Confirmed** | R→L band, ghost mask, consensus, Step 11/15 conclusions |
| R6 | CNN heads evaluated on the easy population, deployed on the hard one | **High** | **Confirmed** (unmeasured gap) | every CNN-derived number on `/stats` and the showcase |
| R7 | corpus identity = plate string → train/val leakage via OCR variants | Medium | Hypothesis | make/colour/body val + head-to-head |
| R8 | traffic counts have no observation-time denominator | Medium | **Confirmed** (magnitude unknown) | `/stats` daily/heatmap/means, schedule miner, R12 |
| R9 | tracker never validated vs ground truth (recall, splits, class, direction) | Medium | **Confirmed** (unmeasured) | all counts; the "night ≈ 15 % of traffic" bound |
| R10 | one global m/px; jogger "valley" may be near vs far pavement | Medium | Hypothesis | jogger/walker split, mph, fastest boards |
| R11 | colour chart mixes three instruments; colour head trained with brightness jitter | Medium | **Confirmed** | `/stats` colour mix, colour head accuracy |
| R12 | resident/visitor uses "first *read* crossing", not "first crossing" | Medium | Hypothesis | showcase buckets |
| R13 | derived artefacts don't track their inputs (stale caches, stale labels, orphan plates on `/stats`) | Medium | **Confirmed** | `/stats` make chart, vehicle-box cache, `data.json` make fields |
| R14 | non-car classes flow through "vehicle" pipelines; `lane` means frame thirds | Low | **Confirmed** | CNN sidecars, hourly `by_lane` |
| R15 | durations use the wall clock; Orin clock sync and DST unverified | Low | Hypothesis | speed, durations, time-of-day stats |

---

## 3. Findings in detail

### R1 — `ocr_conf` was always near 1.0 (confirmed; fixed in code 2026-09-28)

**Code.** `analysis/alpr/preferred.py:168`:

```python
conf = float(np.mean(np.max(char_probs, axis=-1)))
```

The comment says `char_probs` is `(slots, n_chars)`. `uv.lock` pins
**fast-plate-ocr 1.1.0**, whose `core/process.py` returns
`char_probs = np.max(predictions, axis=-1)[i]`, which is already the per-slot
max, shape `(max_plate_slots,)`. Taking `np.max` over its last axis again
collapses it to **one scalar: the most confident slot**. The decoded plate has
trailing pad slots stripped, but `char_probs` keeps them. The pad slots and
the crispest character both score high, so the max sits near 1.0 whatever the
other characters say. `np.mean` of a scalar is that scalar.

**Reproduced on the real model** (fast-plate-ocr 1.1.0,
`global-plates-mobile-vit-v2-model`: 9 slots, pad `_`) on a synthetic yellow
plate `AB12 CDE` with increasing horizontal motion smear:

| Smear | Read | Old `ocr_conf` | New `ocr_conf` (weakest character) |
| --- | --- | --- | --- |
| none | `AB12CDE` (correct) | 0.983 | 0.936 |
| 25 px | `AB122OE` (misread) | **0.998** | 0.235 |
| 45 px | `AE00012` (garbage) | 0.964 | 0.171 |
| 70 px | `42470E` (garbage) | 0.967 | 0.347 |

There is no test of `_unpack_ocr_output`. The repo history is squashed at
2026-07-03 with 1.1.0 already locked, so **every `alpr-run` in the current
corpus**, including all fullframe re-enrichment, has this behaviour.

**Symptoms already on record, attributed elsewhere:**

- "OCR confidence commonly ties at 1.0" (`analysis/vehicles.py`, clustering docstring).
- "the OCR conf score isn't itself informative below 0.95" (`alpr/base.py`, the 05-29 threshold curve).
- "57 % of high-conf reads were OCR misreads" (`dvsa-label --include-non-canonical` help).
- High-confidence misreads were common enough to characterise: the 05-30
  garbage analysis found 74 % of them were clipped six-character reads
  (`alpr-run --plate-pad-frac` help).

**What it breaks** (on every session not yet re-scored).

- `dvsa-label --conf-threshold 0.9` (`cli/dvsa_label.py:181`) passes every read.
- `vehicles.CONF_THRESHOLD = 0.9` passes every read.
- The per-track best read, `argmax(ocr_conf)` (`cli/alpr_run.py:555`), ranks
  reads by their single most confident slot, which says nothing about the
  plate: in the table above the misread outranks the correct read. Where the
  values tie at 1.0 it falls back to lexicographic snap order (`_main_1`,
  `_main_10`…`_main_15`, `_main_2`…).
- Consensus weights are all near 1.0.
- `.claude/rl_rescore_e1.py` `conf >= 0.9` is a no-op.
- Every "canonical read rate @ conf ≥ 0.9" is really "canonical-shape rate".
- The only filter actually in force is the UK plate regex. A one-character slip
  that stays UK-shaped goes straight to DVSA (see R3).

**Fix (done 2026-09-28, E1.1).**

- `ocr_conf` is now the probability of the weakest decoded character (pad
  slots excluded): a plate is only as good as its worst character.
- The per-character probabilities are persisted in `_alpr.json`
  (`ocr_char_probs`), and the session stamp `_static_plates.json` records
  `"ocr_conf": "min_char"`.
- `tests/test_analysis/test_alpr/test_preferred.py` pins the 1.1.0 output
  shape. An unknown shape now raises, and `FastPlateOcrRecognizer` probes it at
  start-up, so a library change fails the run immediately instead of silently.
- `streettracker alpr-rescore <session>` re-OCRs the saved plate crops
  (`alpr_crops/`) instead of re-running full-frame YOLO, so existing sessions
  are fixed in minutes. The panel's **Re-score plate confidence** playbook runs
  it, then `dvsa-label` → `dvsa-apply` → `vehicles`, on every session still
  flagged **Plates v2**.
- The three separate 0.9 gates (dvsa-label, `vehicles`, the stats page's
  fastest-car plates) are now **one shared setting**: `configs/alpr.json`
  `{"plate_conf_threshold": X}`, default 0.9 (`analysis/alpr/base.py`).
- **Still open:** that gate is now meaningful ("every character ≥ 0.9")
  but uncalibrated. `.claude/ocr_conf_calibration.py` gives a label-free first
  answer from data already on disk. The old gate let every read through, so
  nearly every UK-shaped read was already looked up on DVSA. By confidence group
  it reports the not-on-register rate for plates old enough to have an MOT, the
  DVSA-vs-CNN colour mismatch and snap agreement, plus a threshold sweep. Confirm
  the result on the E1.2 audit set later.

### R2 — ALPR attribution is unverified; the headline is a shape rate (hypothesis)

**Code.** `TrajectoryCropDetector.detect` (`analysis/alpr/fullframe.py`):

- It ranks on-road vehicles by distance to a hint known to sit ~510 px from the
  car.
- It tries the top **two** (`DEFAULT_MAX_CANDIDATES = 2`) and keeps the
  **highest detector confidence** (`:224`), not the better-ranked candidate.
- With no plate on either candidate it falls back to the hint crop, and with no
  hint to the **whole image** (`:246`).
- `staticfilter` removes *stationary* plates only. A second *moving* car, such
  as opposing traffic crossing in frame, is not filtered.

**The metric.** The 2026-07-28 "R→L solved in software" verdict and every
per-car canonical rate since count a car as "read" when *any* snap produced a
UK-shaped string (with R1, at any confidence). It does not check that:

1. the characters are right, or
2. the plate is on the tracked car rather than an oncoming or parked one.

The verdict may well be substantially right. The ~10 % → ~70 % jump is too large
to be all artifact. But the number is unverified, and it anchors the DVSA
labels, the training corpus and the showcase.

**Automatic consistency checks available today.**

- **Plate colour vs direction.** UK front plates are white and rear plates
  yellow. On this scene R→L shows the front plate and L→R the rear (CLAUDE.md,
  visually confirmed). So a colour-mode read whose plate background disagrees
  with its track's direction belongs to another car or a parked car. This needs
  only the saved plate crops and HSV, and covers every read ever made. (Exclude
  IR-mono frames and motorbikes, which have rear plates only.)
- **Plate collisions.** The same plate (fuzzy ≥ 85) being the best read of two
  tracks that overlap in time:
  - in opposite directions: at least one read is misattributed;
  - in the same direction and adjacent: a BotSORT split (this doubles as R9's
    split-rate measurement).
- **Motion consistency.** Across a track's snaps, the plate's 4K position should
  move in the track's direction. A plate that moves the other way, or not at
  all, is suspect.

### R3 — DVSA labels from misreads (hypothesis; mechanism confirmed)

**Mechanism.** The label for a track is DVSA's record for its best read. UK
registrations are dense within an area code and age identifier. So a
one-character OCR slip on the random letters often resolves to **another real
vehicle** with its own make, colour and model. Nothing checks the returned
record against the image. The DVSA-distinct veto in `vehicles.py` only stops two
such plates *merging*; it doesn't reject the label.

R1 means no confidence gate stands in front of this. R2's arbitrary best-read
choice means the label can come from the worst read of a track.

**Blast radius.**

- `makemodel-build-uk` labels every plate-anchored crop of that track with the
  wrong make, colour and body type. Plate mode guarantees the crop shows the car
  whose plate was *read*, not the car whose record was *returned*.
- The same label is the val truth, so reported accuracy is biased. The sign is
  unknown: a right model is scored wrong on a mislabelled car.
- `/stats` make, colour and body mixes read these labels directly.

**Signals that estimate the rate without new labels.**

- **Support:** the number of independent reads (distinct snaps, tracks,
  sessions) of the exact string. Single-read plates are the risky population.
  Measure their share of the corpus.
- **Appearance agreement:** DVSA `primary_colour` group vs the colour CNN, and
  DVSA-model body type vs the body CNN, on images the CNNs didn't train on. If
  single-read labels disagree far more often than multi-read labels, the excess
  estimates the misread rate.
- **404s by plate age:** current-format plates (`LL00LLL`) encode their
  registration half-year. A plate older than ~3.5 years that 404s is almost
  certainly a misread. The 404 rate among old plates estimates the misread rate
  per lookup. The E1.2 audit then gives what fraction of misreads hit a real car.

### R4 — sub-stream ↔ 4K registration: spatial and temporal (hypothesis)

**Spatial.** `snap_assets._scale_bbox_to_image` (`:145`) maps 896×512 →
4512×2512 by independent x and y scale factors (aspect 1.750 vs 1.796),
**assuming the two streams share a field of view**. `common/door_zone.py` states
the same assumption. The road polygon was traced on a 4K frame but gates the
live planner on the sub-stream; the ghost mask, door zone and entry/exit points
use it too.

If the sub-stream is actually a crop, the error reaches ~2.6 % of width
(~115 4K px) at the frame edges, and the snap planner's `t_norm` is misregistered.

**Temporal.** There are three clocks, and none has been measured:

- **Sub-stream frame time.** Frames are stamped with `time.time()` when
  *processed* (`device/runtime.py:323`), after RTSP and decode latency.
- **Snap latency.** It is measured from fire to *disk write*
  (`device/snapshotter.py:175`), not to image exposure.
- **Done bbox.** It is the latest *processed* sub-stream position after the disk
  write (`device/runtime.py:496`).

The ~510 px done-bbox offset found on 07-28 is the symptom. Fullframe sidesteps
it for choosing the crop, but still anchors on "nearest to a stale hint". Every
landing-position analysis (band tuning, door-approach bands, "L→R departs
~0.10 t_norm during latency") uses the same unmeasured timing model.

**Missing data.** No per-snap fire or done timestamps are persisted, and no
per-track trajectory. So the offset can't be fitted from existing sessions.

### R5 — decisions measured through broken instruments (confirmed)

| Decision / recorded belief | Measured through | Status today |
| --- | --- | --- |
| **R→L pipeline band `[0.10, 0.20]`** (rationale at `device/snap_planner.py:206-211`: "R→L front plates at 0.10-0.20 (56-76 %)") | 06-11 motion-window hints, which CLAUDE.md later declares a forward-extrapolation artifact | **Live; never re-derived with fullframe crops.** A 10 %-wide band far from the camera may be starving R→L of snaps (and the corpus of front views) |
| L→R band `[0.30, 0.60]` (nudge "validated" 06-22) | completion-bbox crops (~510 px off) + R1 | Live |
| `t_usable_frac` `[0.10, 0.45]` (Step 11) | hint crops | Live |
| **Ghost mask** zero-fills rect `[750,700,960,860]` in every snap (the enrich playbook passes `--ghost-mask`) | hint crops (Step 10) | Live. The static-plate filter now does its job adaptively; the fixed rect may blank genuine plates of moving cars crossing that spot |
| Multi-frame consensus "−43 pp" (Step 13b) | hint crops (reads of different plates) + R1 (all weights 1.0) | Primitive shelved on invalid evidence |
| "78 % aliasing-free floor" / "capture-side tuning exhausted" | hint crops | Superseded in prose, still in tables |
| Anti-Smearing falsified (Step 15) | same-hint A/B + R1 | Relative result probably robust (both arms share the crop error); rates not |
| Night lever = motion blur (08-12) | fullframe, but canonical-shape rate | Laplacian mechanism evidence is independent → likely robust; night rates probably *overstated* (blurred plates misread into valid shapes) |
| Jogger threshold 2.5 m/s | global m/px (R10) | Live |
| DVSA "~25-30 % coverage", body-type 90 % mapping coverage | R1/R3 labels | Probably fine as coverage; accuracy unknown |

### R6 — CNN heads: easy evaluation, hard deployment (confirmed gap)

- **Training** uses plate-read snaps only: `plate` mode, `analysis/makemodel/uk_dataset.py`.
- **Evaluation** is on held-out plated cars' tracks (`makemodel/compare.py`;
  its docstring states this caveat).
- **Deployment:** the CNN only decides anything where DVSA doesn't:
  - unplated tracks, which skew to night (dark read rate ~17 %), far, blurred
    and oblique;
  - cars **under three years old** (DVSA 404, so never in training; newest body
    styles and EVs);
  - makes below `min_cars_per_make`, which are forced into one of 62 classes
    because there is no reject class.
- The confidence thresholds (make 0.4, colour 0.5) are uncalibrated softmax
  cut-offs.
- R→L front views are likely under-represented, given the narrow far R→L band
  (R5).

So 60.4 % make@1 is the accuracy on the easiest population. The accuracy of the
numbers `/stats` actually shows is unknown.

### R7 — corpus identity is the plate string (hypothesis)

`car_make[plate] = mk` (`uk_dataset.py:175`) and `split_val_cars` (`:326`) key
on the exact plate string. `compare.py` excludes production's training cars by
exact string. So one physical car read under two canonical spellings becomes two
"cars": it can land in both train and val, and the misread spelling carries
another car's label (R3). Every resident with hundreds of reads is a candidate.

### R8 — counts without an observation-time denominator (confirmed)

`web/stats.py` sums tracks per date, hour and weekday. `per_day_mean = total /
n_days` (`:728`) and the weekday × hour heatmap (`:631`) have no correction for:

- session gaps, unpulled sessions, Orin downtime and RTSP reconnects;
- partial first and last days;
- IR periods, during which inference is skipped entirely
  (`device/runtime.py:700`). IR periods are recorded in `_meta.json` and
  `_hourly.json`, but **no analytics code reads them**.

The busiest date and hour, the YMCA schedule miner and R12 all inherit this.

### R9 — the tracker has never been checked against ground truth (confirmed gap)

"One track ≈ one pass" is assumed. Unmeasured:

- detection recall by lighting (YOLOv8m on the 896×512 sub-stream at dusk and
  dark);
- the car split rate (only people have a measured merge rate);
- direction from track endpoints (`device/track_buffer.py:488`);
- van/truck class confusion: `/stats` counts `class_name == "car"` only;
- the effect of frame drops (queue of 2, drop-oldest) on splits.

This also makes "night ≈ 15 % of traffic", which bounds the night-capture
lever, circular: night traffic is counted by the same detector that
under-performs at night.

### R10 — one global metres-per-pixel factor (hypothesis)

`speed_m_s = speed_px_s * m_per_px` (`analysis/people.py:483`), with `m_per_px =
road_length_m / 801` averaged over the whole road axis. Perspective makes the
near pavement several times more pixels per metre than the far end.

The 2.5 m/s jogger boundary is the "valley" in a pooled bimodal person-speed
histogram. **Walkers on the near and far pavements would produce exactly that
bimodality.** 23 % joggers is high for a residential street. The fastest-car
boards and the L→R vs R→L average speeds share the bias, since the two
directions use lanes at different depths. A homography helper already exists
(`.claude/render_homography_points.py`) but is not wired in.

### R11 — the colour chart mixes three instruments (confirmed)

`web/stats.py:635` uses DVSA register colour, else a CNN colour with conf ≥ 0.5,
else the HSV `color` field. About 22 % of passes have a blank CNN colour, so at
least a fifth of unplated tracks are coloured by the HSV voter. That voter is
documented at ~40 % grouped accuracy and drifts light cars to black and blue, so
the chart is biased that way.

Separately, the colour head trains with `ColorJitter(brightness=0.2,
contrast=0.2)` (`analysis/makemodel/dataset.py:242`, shared by all heads).
White, silver and grey differ mainly in luminance, so this augmentation
perturbs exactly the cue that separates those classes.

### R12 — "first crossing of the day" is "first *read* crossing" (hypothesis)

`web/classify.py:309` takes the first crossing each day, but only read (plated)
crossings exist per car. The read rate is about 75–85 % by day and 17 % in the
dark. On winter mornings a resident's pre-dawn departure goes unread, the first
*read* crossing becomes the return, and the car drifts to visitor or YMCA staff.
Coverage gaps (R8) and misattribution (R2) push the same way.

### R13 — derived artefacts don't track their inputs (confirmed)

- **`/stats` make chart counts every plate ever DVSA-labelled.** That includes
  orphans whose `track_ids` were cleared by beacon suppression or re-enrichment,
  and every hint-era plate. `stats.py:558` has no `track_ids` filter, and
  `dvsa-label` never deletes labels.
- **The vehicle-box cache is keyed by snap filename only**
  (`analysis/vehicle_locator.py:209`). Boxes detected on a zero-filled snap
  survive after `pull --skip-existing` re-fetches the real image.
- **`dvsa-apply` never clears stale labels.** It writes make/model onto tracks
  (`analysis/dvsa_apply.py:105`) but never clears a track that lost its label.
- **`recolor` rewrites the raw `color` field in place.** That field then no
  longer means "the runtime HSV vote".

### R14 — class hygiene (confirmed, low)

`asset_prefix_for_class` maps every non-person class to `"vehicle"`
(`common/schema.py:194`): bicycle, motorbike, bus, truck and dog. So ALPR and
all three CNNs run on dog and bicycle snaps. `/stats` filters `class_name ==
"car"`, so the impact is mostly wasted work, plus sidecar rows that downstream
joins must remember to filter.

`TrackRecord.lane` is the vertical third of the frame
(`device/track_buffer.py:492`), not a road lane, yet it appears as `by_lane` in
hourly rollups.

### R15 — time base (hypothesis, low)

Durations and speeds use the wall clock (`time.time()`, `runtime.py:323`), so an
NTP step mid-track corrupts duration and therefore speed. An Orin booting
without a synced clock mislabels a session.

British Summer Time ends on **2026-10-25**, when local 01:00–02:00 repeats.
`/stats` buckets on the local ISO string, so that hour double-counts.

### Checked and sound (no action)

- Pull integrity: `pull._is_intact_jpeg` catches the SOI/EOI gaps.
- The head-to-head design: same held-out cars, bootstrap by car, fresh-compare
  promotion gate. It inherits R3, R6 and R7 only through its inputs.
- The time-shift chance controls for round trips and the classifier.
- The `class_suspect` aspect-ratio guardrail, which uses sub-stream geometry and
  is unaffected by R4.
- The strict config loader and schema-additive deploy order.
- The static-plate filter's affine-consistency design. It uses fire and done
  bboxes as the motion reference, but relative motion, not absolute position.

---

## 4. Experiment plan

The ordering principle: **fix the instruments before re-measuring anything, and
build one small human-verified ground-truth set that several experiments
share.** Each experiment states the question, the method, the cost and a
decision rule.

### Revision (2026-10-04)

Between 2026-09-28 and 2026-10-04 (CLAUDE.md handoff steps 1-5), every
session was re-scored with the min-character confidence (E1.1). The plate gate
was calibrated label-free to 0.90 and applied. The corpus was rebuilt on the
post-gate labels (`uk_crops_0929_576`), and all three heads were retrained and
promoted on fresh-session head-to-heads. What that taught:

1. **Fixing a measurement beat every tuning change.** The stale-hint fix, the
   full-frame crops, the clean crops (+22.5 pp make@1) and the clean labels
   (+4 to +7 pp on the fresh week) were all instrument fixes. The ordering
   principle above stands.
2. **The DVSA register checks OCR, not attribution.** It calibrated the gate
   without hand labels. But it can't see an oncoming car's plate pinned on the
   tracked car, because that plate is real. Nor can it see a misread that lands
   on another real car (R3). So the 2.0 % not-on-register rate at 0.90 is a
   lower bound on label error. E1.2 is still the only way to measure
   attribution and to turn that bound into a rate.
3. **Snap agreement beats confidence.** Reads at 0.85-0.90 that another snap
   agrees with are 1.0 % not-on-register; reads at 0.90-0.95 that no snap
   agrees with are 7.0 % (`.claude/ocr_calibration.json`, `agreement_rescue`).
   This is E1.4's support signal, and it makes E2.4's consensus question
   largely the same question.
4. **About 88 % of cars are regulars.** A week yields only ~100 cars that are
   in no training corpus, so heads are now judged with `makemodel-compare
   --eval-session` (fresh cars) plus `--include-trained-cars`. R7 now affects
   that fresh set: a misspelt read of a regular can pass as a "fresh" car.
5. **The per-car crop cap doesn't help** (tested 2026-10-01).
6. **The IR half of the night fix conflicts with the runtime.** The runtime
   skips inference whenever frames turn monochrome (`device/runtime.py:699`,
   `device/ir_detector.py`). A supplementary IR illuminator only helps once
   the camera drops its IR-cut filter, which makes the frames mono, so the
   tracker would record nothing.

**What changes.**

- *Ship without measuring:* E0.2, because the `track_ids` filter is right by
  construction and step 4 already cleared 12,672 stale labels. E0.8, because
  R11 already puts the HSV share near 22 %, past the 10 % rule.
- *Done or retired:* E0.1 (superseded); E2.4 (folded into E1.4); E2.5's
  corpus rebuild and `--max-per-car` run (done; the cap stays as an option).
- *Pulled forward:*
  - the DST repeat-hour fix from E3.5, with E0.6 (BST ends 2026-10-25);
  - E1.5(c), because it only produces data for sessions recorded after it
    ships, and the Orin deletes 4K snaps after 7 days;
  - a label-free first cut of E1.4: the combined agreement + confidence gate;
  - E2.6(a), because all three heads were promoted on plated cars but `/stats`
    shows them for unplated ones.
- *Reordered:* E0.5 and E1.3 run before E1.2, so E1.2's sample can
  over-represent the reads they flag.
- *Retargeted:* E0.4 now reads `uk_crops_0929_576` and also checks the
  fresh-session evaluation set.
- *Added:* E0.9 (R3 rate from colour agreement) and E2.8 (night capture,
  shutter-only unless the runtime changes).

**Revised order.**

1. Small code fixes: E0.2's `/stats` filter, E0.8's HSV-as-unknown, the DST
   repeat hour (E3.5), and the E0.6 check.
2. E1.5(c) runtime persistence: a code-only Orin deploy, in the
   schema-additive order.
3. Phase 0 script: E0.3, E0.4, E0.5, E0.7, E0.9.
4. E1.3 plate colour, and the combined gate (E1.4's first cut).
5. E1.2 audit set, stratified to over-represent flagged reads; then calibrate
   E1.3 and the combined gate against it.
6. E2.6 (plate-blind first), E2.1, E2.3, E2.7. E1.5(a) and (b) can run at any
   point.
7. E3.1 hand count, then E2.8 at dusk.
8. Retrain (E2.5) once the E1.4 filter and cluster-aware exclusion exist and
   fresh cars have built up.

The rest of Phase 3 (E3.2-E3.4; E3.4 waits on E0.3's coverage map) and Phase 4
follow unchanged.

### Phase 0 — read-only checks on existing outputs (dev box, ~1 day, no GPU)

These confirm or kill several findings in minutes. They only read `output/` and
`runs/`.

| ID | Question | Method | Decision rule |
| --- | --- | --- | --- |
| **E0.1** (R1) | Is `ocr_conf` saturated in practice? | **Superseded:** `alpr-rescore` prints the share of reads at ≥ 0.9 before and after, plus the tracks that pass the DVSA gate before and after. | Use that output to size the impact on DVSA labels; see E1.1. |
| **E0.2** (R13) | How much of the `/stats` make chart is orphan or stale plates? | Per session: labels with empty `track_ids`, and labels whose plate is no read in the current `_alpr_by_track.json`. Recompute the make chart with and without them. | **2026-10-04: ship the `track_ids` filter without measuring** (it is right by construction). Was: any top-12 make shifting > 2 pp → ship it. |
| **E0.3** (R8) | How much time is actually observed? | Per session: start/end, IR periods (`_meta.json`), gaps > 120 s between consecutive track starts during 07:00–19:00 (outage proxy), `frames_processed / pipe_fps` vs wall duration. Build an hour-by-hour coverage map across all dates. | Any date or weekday-hour cell < 95 % covered → build E3.4 before quoting daily means or heatmaps. IR periods non-empty → count them as unobserved. |
| **E0.4** (R3, R7) | How much of the corpus is single-read or split across OCR variants? | In the production corpus manifest (`uk_crops_0929_576` since 2026-10-01; was `uk_crops_0924_576`): cluster plates with the `vehicles` fuzzy rule (ratio ≥ 85, same length). Count clusters spanning train and val, and their share of val crops. Count per-plate read support from `_alpr.json`. Also count the "fresh" cars in a `makemodel-compare --eval-session` run that fuzzy-match a corpus plate (added 2026-10-04). | Leakage > 2 % of val tracks → re-run `makemodel-compare` with cluster-aware exclusion (cheap). Single-read plates > 10 % of corpus cars → prioritise E1.4. |
| **E0.5** (R2, R9) | Do plates collide across simultaneous tracks? | Same canonical plate (fuzzy ≥ 85) as best read on ≥ 2 tracks whose [start, end] windows are within 10 s. Split into opposite direction (misattribution) and same direction, adjacent (BotSORT split). | Opposite-direction collisions > 1 % of read tracks → R2 is real, prioritise E1.2/E1.3. The same-direction rate is the first car split-rate estimate. |
| **E0.6** (R15) | Is the time base sane? | `timedatectl` on the Orin (NTP synced?). Scan `data.json` for `duration_visible < 0`, `time_start` non-monotone in `events.jsonl`, and session label vs first-track time. | Any anomaly → switch durations to the monotonic clock and log sync state in meta. Run it with E3.5's DST fix, before BST ends on 2026-10-25. |
| **E0.7** (R10) | Is the jogger mode just the near pavement? | Split person speeds (≥ 6 detections) by pavement, using the median y of entry and exit points (post-07-19 sessions) or a y threshold on the hq tile. Plot per-pavement histograms. | Each pavement unimodal, modes differing by ~the perspective ratio → the jogger class is an artifact; suspend the jogger/dog-jog stats until E3.2. |
| **E0.8** (R11) | What is the colour chart made of? | Re-run the stats colour loop, tagging each car track's colour source (DVSA / CNN / HSV / unknown). Cross-tab source × colour. | **2026-10-04: render HSV-sourced tracks as "unknown" without measuring** (R11 already puts the HSV share near 22 %). Was: HSV share > 10 %, or black/blue over-represented in HSV rows → do so. |
| **E0.9** (R3; added 2026-10-04) | How often does a DVSA label describe a different real car? | On sessions recorded after `uk_crops_0929_576` was built (no head trained on them), compare each labelled track's DVSA `primary_colour` group with the colour CNN's per-track read, split by read support (agreeing snaps) and confidence group. The CNN crops the car whose plate was read, so a misread that lands on another real car shows up as a mismatch. Add the not-on-register rate among plates old enough to have an MOT (`.claude/ocr_conf_calibration.py`). | Single-read labels mismatching well above multi-read labels (and above the CNN's own fresh-car error, ~17 % exact) → the excess estimates the misread-to-real-car rate; adopt the E1.4 filter before the next retrain. |

### Phase 1 — build the instruments (week 1)

**E1.1 — Fix the OCR confidence (R1). Done 2026-09-28.** The threshold was
calibrated label-free the same day (0.90, from the DVSA not-on-register rate by
confidence group); E1.2 confirms or moves it. What shipped differs from the
plan below in one respect: the per-track best read is still the
highest-confidence read (now meaningful), not "most-supported string".

- Code: compute the per-read confidence over decoded characters (min, and
  product), and persist `ocr_char_probs`.
- Add a unit test with a 1.1.0-shaped `PlatePrediction` (pad slots at 1.0, one
  weak character) asserting the weak character dominates.
- Add an `alpr-rescore` path that re-OCRs saved `alpr_crops/` and rewrites
  `_alpr.json` and `_alpr_by_track.json` without re-running full-frame YOLO.
- Per-track "best" becomes most-supported string, then highest min-char
  confidence. Record `ocr_conf_version: 2` in the stamp so consumers can tell
  old from new.
- *Decision:* thresholds are chosen in E1.2 by a precision target (e.g. ≥ 98 %
  char-exact at the plate level), never by assumption.

**E1.2 — Attribution + OCR audit set (R1, R2, R3 ground truth).**

- About 300 reads, stratified: direction × {day, dusk, dark} × {plate-anchored
  rank-0, rank-1 won, hint/whole-image fallback}. Plus 100 DVSA-labelled cars
  stratified by read support.
- Extend `.claude/triage_rl.py` (labelling site, `:8091`). Per item show the 4K
  snap with the tracked car's sub-stream trajectory hint, the chosen vehicle box
  and the plate box. The labeller answers:
  1. Is the boxed plate on the tracked car?
  2. The true plate text.
  3. For DVSA items: do make, colour and body match the photo?
- About 3 operator-hours.
- *Outputs:* verified per-car read rate by direction and lighting, with a
  bootstrap CI (replaces the shape rate); OCR precision/recall vs the new
  confidence (calibrates E1.1); misattribution rate by fallback path;
  DVSA-label error rate by support (feeds E2.5).
- Keep it as a **frozen regression set**: every future ALPR or DVSA change
  reports against it.

**E1.3 — Plate-colour/direction consistency, automatic, every read (R2).**

- Classify each saved plate crop as yellow or white (HSV on the plate
  background, excluding glyph pixels). Skip IR-mono frames.
- Calibrate on the E1.2 labels, then run corpus-wide.
- *Decision:*
  - A mismatch rate materially above the E1.2 error floor → change candidate
    selection:
    - prefer rank 0 unless rank 1's detector confidence wins by a margin;
    - require the plate position to agree with the track's motion;
    - reject reads whose plate colour contradicts the direction.
  - Use the check as a standing per-session health metric in the panel.

**E1.4 — DVSA label-quality signals (R3).**

- **First cut now, label-free (2026-10-04):** a plate gate that combines snap
  agreement with min-character confidence, since agreement separates the
  not-on-register rate better than confidence alone (see the revision above).
  Build it into the shared gate, then validate it on E1.2.

- Compute, per labelled plate:
  - read support (E0.4);
  - DVSA colour group vs held-out colour CNN;
  - DVSA body type vs body CNN;
  - age-identifier plausibility.
- Estimate the misread rate among old-plate 404s.
- Validate the combined score against the E1.2 DVSA items.
- *Decision:* adopt the cheapest filter that achieves ≤ 2 % label error on
  E1.2. For example "support ≥ 2 OR (min-char conf ≥ τ AND colour-consistent)".

**E1.5 — Measure sub-stream ↔ 4K registration and timing (R4).**

- (a) **Spatial.** Grab a sub-stream frame and a 4K snap of a static scene
  (quiet night road). Feature-match (ORB/SIFT, RANSAC homography) and report
  the residual of the current anisotropic-scale model. Pass: max residual
  < 0.5 % of width across the road polygon. Otherwise ship a fitted transform
  in `snap_assets` and re-map the polygon, ghost mask and door zone.
- (b) **Temporal.** If the Reolink OSD clock is on (or can be enabled for an
  hour), read it in sub-stream frames vs processing wall time to get the
  sub-stream latency. Read it in 4K snaps vs fire time to get the exposure
  delay.
- (c) **Runtime, schema-additive.** Persist per-snap `fire_unix` / `done_unix`
  and a decimated per-track trajectory (`[t, x1, y1, x2, y2]` every ~3rd frame,
  ~2 KB/track). Then fit the exposure offset per session by aligning
  fullframe-detected 4K car boxes to the interpolated trajectory.
  **Code written 2026-10-04:** `TrackRecord.main_snap_fire_unix` /
  `main_snap_done_unix` (parallel to `main_snaps`; fire decision and
  JPEG-on-disk, wall clock), and a `{session}_trajectories.jsonl` sidecar
  rather than a `TrackRecord` field, so `data.json` doesn't grow. Each line
  holds every 3rd bbox plus the last, capped at ~200 rows, as `[dt, x1, y1,
  x2, y2]` from `t0_unix`. Pull already copies `*.jsonl`, and prune only
  deletes 4K JPEGs. The fit itself is still to write, once sessions carry
  the data.
- Deploy in the order in [Schema-additive config
  changes](../CLAUDE.md#schema-additive-config-changes). No `camera.json` change
  is needed, so this is a code-only deploy.
- *Decision:* with a measured offset, replace "nearest to stale hint" in both
  `fullframe` and `vehicle_locator` with "nearest to the *predicted* position at
  fire + δ". Re-run the E1.2 strata to confirm the pick error drops.

### Phase 2 — re-measure and repair ALPR + ML (weeks 2–3)

**E2.1 — Re-derive landing curves per direction (R5).** No new capture needed.

- Using existing fullframe snaps, key each snap by the **detected** car
  position's `t_norm` (as `rl_rescore_e1.py` does).
- Score it with *verified* reads: E1.1 confidence plus E1.3 consistency, with
  E1.2 as the check.
- Produce read-rate vs `t_norm` curves for R→L and L→R, by lighting.
- *Decision:* propose new `pipeline_t_usable_by_direction` bands where the
  curves say reads happen.

**E2.2 — Live R→L band A/B (R5).** If E2.1 shows R→L reads well outside
`[0.10, 0.20]`:

- Deploy the proposed band on alternating days. It is a value change, so safe
  to deploy directly.
- Measure verified per-car R→L reads and R→L corpus crops per day.
- Watch `snap_stats.dropped`, since the HTTP semaphore is shared.

**E2.3 — Ghost mask on/off (R5).** Re-score three fullframe sessions without
`--ghost-mask`, keeping the static filter.

- *Decision:* if verified reads rise and the static filter still suppresses
  FD61PVX-style beacons, drop the mask from the enrich playbook.

**E2.4 — Consensus re-test (R5). Folded into E1.4's combined gate
(2026-10-04):** snap agreement is consensus at the string level. Re-run `measure_consensus.py` on fullframe
sessions with E1.1 confidences against E1.2 truth. The Step 13b negative was
measured on crops of different physical plates, with all weights 1.0.

**E2.5 — Rebuild and retrain on clean labels and identities (R3, R7).**

*Partly done 2026-10-01/03:* the corpus was rebuilt on post-gate labels
(`uk_crops_0929_576`) and all three heads retrained and promoted; the
`--max-per-car` run didn't help. Still to do: the E1.4 filter and
cluster-aware splits and exclusion. Judge with `makemodel-compare
--eval-session` (fresh cars) plus `--include-trained-cars`, since val-split
head-to-heads shrink to ~67 cars once corpora share most of their cars.

- Rebuild the corpus with:
  - the E1.4 label filter;
  - identity-clustered train/val splits (clusters, not strings);
  - label support recorded in the manifest.
- Retrain make, colour and body. `makemodel-compare` must exclude by cluster.
- Also run `--max-per-car` (resident dominance is already flagged as an untested
  lever).
- *Decision:* promote only on the head-to-head *and* on E2.6's deployment-like
  slices.

**E2.6 — Evaluate where the heads are deployed (R6).**

- (a) **Plate-blind:** for held-out plated tracks, drop every snap whose plate
  was read. Classify from the rest, which the trajectory rule locates; this is
  the proxy for unplated tracks.
- (b) **Stratify** accuracy by direction, lighting band, and crop height
  quartile.
- (c) **Open-set:** cars of makes below `min_cars_per_make`. Report how often
  they are assigned above the confidence threshold.
- (d) **Calibration:** reliability diagram per head. Set make/colour/body
  thresholds for a stated precision.
- (e) **Age skew:** plate-age distribution of the training cars vs all read
  plates.
- *Decision:* `/stats` shows CNN-derived mixes only for slices whose measured
  accuracy clears a bar, labelled as estimates. Everything else shows as
  "unknown".

**E2.7 — Colour-head augmentation A/B (R11).** Retrain colour with brightness
jitter off (keep contrast or none). Compare white/silver/grey confusion with
`makemodel-compare --target colour`.

**E2.8 — Night capture A/B (R5, R9; added 2026-10-04).** The 2026-08-12
analysis found night plates detected but smeared by motion (CLAUDE.md
Next-steps item 5).

- Test a night-scheduled faster shutter alone, on alternating dusks (19-21 h).
  Score with verified reads (E1.3 plus the combined gate): canonical-shape
  night rates are probably overstated (R5).
- Don't add an IR illuminator without a runtime change. It only helps once the
  camera switches to IR mode, and the runtime skips inference on monochrome
  frames (`device/runtime.py:699`), so the tracker would go blind. Either make
  the IR skip configurable (after checking YOLO recall on IR frames) or leave
  IR out.
- Run E3.1's dusk and dark hand counts first: "night ≈ 15 % of traffic" comes
  from the detector that struggles at night (R9), and it bounds what this can
  win.
- *Decision:* keep the night schedule if verified dusk reads rise and day reads
  don't fall.

### Phase 3 — analytics semantics (weeks 3–4)

**E3.1 — Tracker ground truth (R9).**

- Record three 20-minute sub-stream clips (day, dusk, dark) with ffmpeg on the
  dev box, and run `streettracker batch` on them.
- Hand-count passes by direction and class (car / van / truck / bike / person).
- *Outputs:* recall by lighting, split rate, direction accuracy, and the
  car-vs-truck share for vans.
- *Decision:* apply per-lighting correction factors to counts or state the
  measured error. Decide whether vans (COCO "truck") belong in "journeys".

**E3.2 — Ground-plane speed (R10).** Measure the four points from
`.claude/render_homography_points.py` and fit the homography. Recompute speeds
from trajectories (E1.5c) or entry/exit points, then re-derive the
walker/jogger boundary and backfill `_people.json`.

**E3.3 — Classifier robustness (R12).**

- For operator-tagged cars (showcase metadata), check bucket stability when
  reads are dropped at the measured hour-of-day read rate.
- Check first-crossing polarity vs sunrise time.
- *Decision:* gate "first crossing" on daylight or read-rate, or add
  unplated-pass evidence at those times.

**E3.4 — Coverage-normalised stats (R8).**

- Persist an explicit uptime log: session start/stop, RTSP reconnects and IR
  periods are already known to the runtime.
- Normalise per-day, heatmap and schedule-miner rates by observed hours, and
  grey out uncovered cells.

**E3.5 — Hygiene (R13–R15).** The `/stats` make-chart filter and the DST
repeat hour are pulled forward to the first step of the revised order
(2026-10-04); `dvsa-apply` has cleared before writing since PR #111.

- The `/stats` make chart counts only plates with current `track_ids`.
- Key the vehicle-box cache by (name, size, mtime).
- `dvsa-apply` clears before it writes.
- Derived files carry an input fingerprint (inputs' size/mtime, `crop_mode`,
  model hash, `ocr_conf_version`). The panel badges stale ones.
- Durations use the monotonic clock.
- Handle the DST repeat hour.
- Restrict CNN inference to car-like classes.
- Rename or drop `lane`.

### Phase 4 — guardrails against recurrence

1. **Every headline number names its ground truth.** Report the verified rate
   next to the proxy rate, from the frozen E1.2 set, until the two agree.
2. **An instrument change log.** When an instrument is found wrong, list every
   decision measured through it and its status. The R5 table is the template;
   keep it in CLAUDE.md.
3. **Visual spot-check sheets by default.** Every audit script writes a contact
   sheet of 50 random items with the relevant boxes drawn. The crop problem was
   visible in minutes once someone looked.
4. **Pin third-party output semantics with tests** (R1's root cause: a library
   API changed shape under a lenient unpacker). Fail loudly on an unexpected
   shape rather than coercing it.
5. **Provenance stamps on derived artefacts** (E3.5), so "which instrument
   produced this?" is answerable from the file.

### Until the fixes land — how to quote the numbers

| Quote | Replace with |
| --- | --- |
| "R→L 69.7 % / L→R 66.8 % read rate" | "canonical-shape read rate; correctness unverified (see R2)" |
| "`ocr_conf ≥ 0.9`" on a session flagged **Plates v2** | "UK-shaped" (the old confidence gate was inert, R1). After re-scoring: "every character ≥ 0.9; threshold calibrated on the DVSA register, which misses misreads that land on a real car" |
| "make@1 60.4 %" | "60.4 % per track on plated held-out cars; unplated accuracy unmeasured (R6); labels unaudited (R3)" |
| daily means / heatmap | "raw counts; observation time not normalised (R8)" |
| joggers | "fast-moving person tracks; may be a near-pavement artifact (R10)" |
