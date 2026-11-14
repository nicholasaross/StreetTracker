# Make, colour and body-type classifier history

> Moved from `CLAUDE.md` on 2026-11-11. The text is unchanged apart from link targets and
> heading levels. It is a dated record, written as the work happened: later entries correct
> earlier ones, and passages marked SUPERSEDED, CORRECTION or UPDATE say where. "Above" and
> "below" refer to the old file's layout. For what is true now, read "Current state" in
> [`CLAUDE.md`](../../CLAUDE.md).

## Make/model classification

**⮕ CROP CONTAMINATION (2026-09-24 — read before trusting any make/colour/
body-type number below):** the corpus builder AND all three CNN inference
commands cropped each 4K snap around the stale FIRE-TIME bbox
(`resolve_bbox_hint`); the snap lands ~0.7 s later, so the car has moved
on. Audit of 799 plate-anchored snaps of `uk_crops_0730_576`
(`.claude/makemodel_crop_audit.py`): only **7.6 %** of training crops
held ≥80 % of the labelled car, **66.5 %** held <30 % (empty road / a
parked car); the hint sat a median 701 px (0.85 car-widths) off the car.
Same root cause as the R→L "geometry cap" (fixed for ALPR only, 07-28).
Every make@1 / colour / body-type figure to date was trained AND
validated on these crops. **Fix:** `analysis/vehicle_locator.py` —
`makemodel-build-uk --crop-mode plate` (default) crops the vehicle box
holding the car's own plate read on that snap (65 % of snaps, ~all
cars); inference `--crop-mode auto` follows the checkpoint (the trainer
now records `crop_mode` + `crop_pad_frac`): plate-trained models infer
`fullframe` (plate anchor, else nearest on-road vehicle to the hint —
~7 % estimated wrong-pick on snaps without a plate read,
`.claude/fullframe_crop_spotcheck.py`); legacy checkpoints keep `hint`
unchanged. **Compare old vs new models only head-to-head:**
`makemodel-compare <corpus> --candidate runs/<run>/best.pt` scores
production (as deployed + on clean crops) and the candidate on the SAME
held-out cars (the candidate corpus's val split minus every car in
production's training corpus), per track, with a by-car bootstrap CI →
`runs/<run>/compare.json`. The build-train playbook runs it as step 3, and
the panel's promote recommendation uses it; a run on a different crop
family with no fresh report is "can't compare" (never judged on raw val
make@1), and one-click "Promote best model" refuses it. **First
head-to-head (2026-09-24, `uk_crops_0730_576` val minus 0707 cars = 430
cars / 572 tracks):** production B6 as deployed (hint@0.25) **35.1 %**
per-track → the SAME model on clean crops (fullframe@0.1) **42.1 %
(+7.0 pp, 95 % CI +3.9..+10.2)**; the 0730 B6 retrain on ~25 % more
hint-crop data tied production (+0.0, CI −3.3..+3.2) — the crops, not
data volume, were the bottleneck. Held-out cars are plated (plate-anchor
crops), so the gain on unplated tracks (trajectory rule) will be smaller.

**Clean-crop retrain — PROMOTED 2026-09-25.** `runs/uk_make_0924_b6`:
B6@528 b8, 20-epoch cosine / patience 5, on `uk_crops_0924_576` (plate
mode, pad 0.1: **62 makes / 7,319 cars / 81,933 crops**; 1,671 further
DVSA-labelled cars had no plate-anchored snap). Val make@1 (plate-anchored
crops — the easiest views, not comparable with anything earlier) peaked
**0.868 at epoch 2**, then drifted 0.83-0.85 while train loss kept falling
(memorising) → early-stopped at epoch 7. Head-to-head (`compare.json`,
883 held-out cars / 1,319 tracks / 4,223 snaps, minus all 0707 cars):

| model                            | crops         | per-track | per-car | shared-makes per-track |
| -------------------------------- | ------------- | --------- | ------- | ---------------------- |
| old production (0707 B6, 45 mk)  | hint@0.25     | 37.9 %    | 35.7 %  | 38.9 %                 |
| old production, clean crops      | fullframe@0.1 | 43.9 %    | 38.8 %  | 45.0 %                 |
| **uk_make_0924_b6 (62 makes)**   | fullframe@0.1 | **60.4 %**| **50.2 %** | **61.2 %**          |

**+22.5 pp per-track (95 % CI +19.6..+25.4)** — the panel recommended,
the operator promoted (prior model backed up as
`makemodel_b0.20260925T174701.pt`); `makemodel` now infers
`crop=fullframe pad=0.1` automatically, and the reinfer playbook
(started 2026-09-25) re-runs every session with it. Caveats: plated held-out cars (unplated tracks use the
trajectory rule, so their gain is smaller) and the val cars also picked
the best epoch (mild optimism). **Ops:** at `--batch-size 8`, B6@528
spilled ~1.06 GB into WDDM system memory (desktop compositor held
1.2-1.4 GB) and every epoch ran ~200 min instead of ~85. **Measured
2026-09-29** (real model, dummy data, desktop apps closed): batch 8
reserves 9.49 GiB and still spills 834 MiB (3.4 img/s); batch 6 reserves
7.46 GiB with no spill (18.2 img/s, 5.4× faster). Batch 8 can't fit a
10 GB card beside the Windows desktop, so **build-train now defaults to
batch 6**. Per-car crop dominance
(top 500 of 7,319 cars hold 47.6 % of crops; residents LA68CWY 668 /
FD61PVX 589) is an untested lever (`--max-per-car`).

**Colour + body type — clean-crop retrains PROMOTED 2026-09-27.** Both
EfficientNet-B0 @384 on the same plate-anchored `uk_crops_0924_576`
(`runs/uk_colour_0926_b0`, `runs/uk_body_0926_b0`; batch 32, ~40 min/epoch
— single-process photo loading, not the GPU, is the bottleneck for B0).
Both peaked early (colour epoch 1, body epoch 3) then memorised, like
make. Head-to-head (`makemodel-compare --target colour|body_type`, every
pass counts, production's training cars excluded):

| head      | held-out            | as deployed      | old model, clean crops | new model            | new − deployed (95 % CI)      |
| --------- | ------------------- | ---------------- | ---------------------- | -------------------- | ----------------------------- |
| colour    | 907 cars / 1,359 tr | 44.8 % (53.3 % grouped) | 55.2 % (63.5 %)  | **57.5 % (64.6 %)**  | +12.7 pp (+10.1..+15.2)       |
| body type | 767 cars / 1,180 tr | 52.0 %           | 65.3 %                 | **68.5 %**           | +16.5 pp (+13.5..+19.6)       |

Most of each gain is the crop fix itself (old model on clean crops: +10.4
/ +13.4 pp); retraining adds ~2-3 pp. Old models backed up as
`models/{colour,bodytype}_b0.20260927T094243.pt` (untracked); inference
now runs `crop=fullframe pad=0.1` automatically, and a re-run of every
session with both new heads was queued on 2026-09-27.

The dataset-level enrichment pivot. Two prongs:

- **DVSA-first (shipped, PRs #41/#42/#43).** The DVSA MOT API returns
  ground-truth `make`/`model`/`year` from a plate. Pipeline:
  `dvsa-label <session>` harvests `_dvsa_labels.json` (needs
  `configs/dvsa.json`; canonical-plate-filtered so OCR garbage doesn't
  bill the API) → `dvsa-apply <session>` writes make/model onto each
  car's `data.json` + `events.jsonl` record → `vehicles` joins it
  per-vehicle, and `vehicles <dir> --across <dirs>` pools repeat
  vehicles across sessions (same-day / different-day). `TrackRecord` +
  `Vehicle` carry `make`/`model`/`year`/`make_model_source` (= `"dvsa"`).
  Coverage is the readable, ≥3yr-old subset (~25-30 % of cars on this
  scene): newer cars 404 (no MOT record yet), unread plates aren't
  covered at all.
- **Universal CNN (the unreadable majority) — pivoted to UK-native
  training (2026-06-02).** `docs/makemodel_design.md` has the original
  CompCars plan, now partly superseded by the trajectory below:
  1. **CompCars CNN (PR #46).** EfficientNet-B0 fine-tuned on CompCars'
     surveillance subset → **98.8 % CompCars-val** make@1. **But ~1 % on
     real UK cars** — a domain gap (CompCars = zoomed _frontal_ Chinese
     toll-camera shots; this scene = _rear/oblique_ UK street views),
     plus UK makes CompCars never had (Vauxhall/Tesla/MG/Mini…). A dead
     end as a UK predictor. The inference CLI (`streettracker makemodel`)
     - the lifted `analysis/snap_assets.py` helpers are reusable infra.
  2. **UK-native classifier (PR #47).** Train on THIS scene instead: a
     DVSA plate→make lookup auto-labels the car's own crops. Pipeline:
     `pull --only-main` (Orin→local) → `alpr-run … --pipeline preferred
--pre-crop --ghost-mask .claude/ghost_mask.json` → `dvsa-label` →
     `makemodel-build-uk` (crops; leakage-safe **by-car** split) →
     `makemodel-train-uk`. Make-only for now (model-level too sparse).
  - **Status (2026-06-03): RESOLUTION-bound, not task-bound — verdict
    resolved, CNN is viable.** The earlier "~21 % task-bound ceiling"
    was an artifact: `VehicleCropper` hardcoded `output_size=224`, so
    _every_ prior lever (tuning, 2× data, best-view) was silently capped
    at 224 px input while the source cars were 400-800 px (p50 574). Re-
    extracting higher-res (`makemodel-build-uk --output-size`) + training
    there (`makemodel-train-uk --input-size`) breaks it cleanly.
    Resolution ladder (same 800-car corpus, by-car split, seed 0):

    | Config                  | make@1     | note              |
    | ----------------------- | ---------- | ----------------- |
    | B0 @224 (old "ceiling") | 21.7 %     | —                 |
    | B0 @384                 | 32.2 %     | +10.5 pp          |
    | **B0 @512**             | **37.6 %** | best + cheapest   |
    | B4 @384                 | 29.5 %     | overfit, < B0@384 |
    | B5 @456                 | 37.7 %     | overfit, ≈ B0@512 |

    Resolution is the lever (+16 pp 224→512, decelerating). **Backbone
    capacity is saturated** — B4/B5 (`--backbone`) overfit the small
    corpus (train loss → ~0.04 while val stalls) and don't beat B0, so
    **B0 is right-sized**. The three "negatives" at 224 were all
    resolution-capped, not task limits — best-view collapsed to 9-10 %
    precisely because fewer-crops-at-224 starves a resolution-starved
    model further. **EfficientNet-B5 @456 is the production UK model**
    (make@1 0.303 on 1,229 cars — see the 2026-06-08 UPDATE below: B5 0.303

    > B4 0.284 > B0 0.271; the 06/03 B0@512 "37.6 %" was small-val-inflated,
    > honest ≈28 %); predictions carry `make_model_source = "cnn"`, distinct
    > from the `"dvsa"` ground truth.

  - **Only untapped lever is data.** B5's overfitting means capacity is
    data-starved: growing the 800-car corpus is what would let a bigger
    backbone push past 37.6 % (a collection effort — mine more Orin
    sessions — not a tuning knob). Higher-res on B0 (640+) keeps
    decelerating; a coarser target (body-type) is still untried. Banked
    - now productive: the 800-car corpus, the full mining pipeline, and
      the `--output-size` / `--input-size` / `--backbone` / `--top-by-area`
      flags.
  - **Mining the Orin grows the corpus** (`pull` snap-bearing
    sessions → `alpr-run` → `dvsa-label` → `makemodel-build-uk` +
    retrain). It grew 523→800 cars without lifting make@1 — but only
    because that was measured _at 224 px_, where resolution (not data)
    was the bottleneck. Now that resolution is lifted and bigger
    backbones overfit 800 cars, **data is the next lever** to push past
    37.6 %. Ops knowledge: **the Orin prunes 4K snaps after ~1 week**
    (keeps JSON) — pull + process within the week or the training images
    are gone. Don't run `alpr-run` on the Orin (compute-bound running the
    live tracker); pull to the dev-box 3080.

  - **UPDATE 2026-06-08 — the 37.6 % was small-val optimism; data IS a
    lever (it lifted _honest_ make@1 ~21 → ~28 %).** Pulled + enriched 2
    big new sessions (incl. the 3-day `session_20260601_173815`); corpus
    grew **539 → 1,229 cropped cars / 24 → 29 makes / 3,952 → 9,841
    crops**. Retrained B0@512 b16: **make@1 0.271 (29 makes) / 0.284 (24
    makes, apples-to-apples)** — _looked_ like a drop from 0.376, but a
    **leakage-free audit** (`runs/eval_gen.py`) shows the 06/03 baseline
    reproduces 0.3755 on its own ~108-car val yet scores only **0.207 on
    659 unseen cars** — so 0.376 was a small-val artifact and the true
    cross-session make@1 rose ~0.21 → ~0.28 on the bigger, more honest
    val. **Lesson: size the val (>~200 cars) before trusting make@1.** The
    `--backbone b4/b5` re-test on 1,229 cars **reverses the old "overfits
    800 cars" verdict** — at 2.3× data capacity pays off: **B0 0.271 < B4
    0.284 < B5 0.303** (monotonic, 29-make val). **B5@456 is now the
    default model** (make@1 0.303). Inference is **UK-aware**:
    `streettracker makemodel` auto-detects the make-only checkpoint + reads
    its `arch` + `input_size` from the checkpoint, so any backbone is a
    drop-in default (`make_model_source="cnn"`).

## Corpus rebuild and B5 retrain (closed 2026-06-14)

From the 2026-06-13 checkpoint.

**Make-model: CLOSED 2026-06-14.** Corpus rebuilt (`runs/uk_crops_0614_512` =
**32,777 crops / 36 makes**, ~4× the 0611 corpus after folding in the new
session), B5@456 retrained **make@1 0.349 → 0.402** (+5.3 pp on a trustworthy
6,570-crop val; plateaued 0.39-0.40 over epochs 16-30, not a fluke), **promoted**
(`runs/uk_make_0614_b5/best.pt` → `…/models/makemodel_b0.pt`; prior 0.349/0.303
backed up beside it), and **all 17 sessions re-inferred** with it + showcase
refreshed. Data-is-the-lever, confirmed again (4× data → +5.3 pp). The
`*_0613_*` runs are dead partials from a sleep-killed overnight job — ignore.

## B6, the VLM bake-off and body type (2026-07-08 to 2026-09-25)

Item 4 of the 2026-07-07 next-steps list (the rest of the list is in
[project.md](project.md)).

4. **⮕ SUPERSEDED 2026-09-25 — clean-crop B6 promoted.** Every make@1
   below was trained and validated on stale-bbox crops (~2/3 missed the
   car; see [Make/model classification](#makemodel-classification)).
   Production is now **`uk_make_0924_b6`** (B6@528, plate-anchored
   `uk_crops_0924_576`, 62 makes / 7,319 cars / 81,933 crops), promoted on
   a head-to-head: **60.4 % vs 37.9 % per-track make@1 (+22.5 pp, 95 % CI
   +19.6..+25.4)** on 883 held-out cars. Colour + body-type heads still
   run the legacy hint crops — retrain them on the same corpus next.
   History as it stood before the crop fix:
   **Make classifier — B6 SHIPPED, VLM FALSIFIED (2026-07-08):**
   B6@528 trained on `uk_crops_0707_576` (45 makes / 3,942 cars /
   47,881 crops) → **make@1 0.451**, promoted over B5's 0.441. VLM
   bake-off DONE — **Qwen3-VL 8B loses decisively** (0.15-0.18 vs B6's
   ~0.29-0.30 per-car-largest-crop on 400 val cars; ~15 s/image vs ms)
   and shows its own OOD attractor (MINI 23 %) — no pipeline role at
   this size. **Positive surprise: B6 largely fixed the June OOD
   VW-collapse** (VW max-share on unlabelled tracks: B5 19 % → B6 11 %,
   plausible Ford-topped mix) — corpus growth + capacity, not
   structural after all. Bake-off methodology gotchas recorded in
   `.claude/vlm_bakeoff.py` (cross-corpus val leakage: 276/400 of the
   0707 val cars were in B5's train set; per-car-largest is a harder
   metric than the trainer's per-crop val). **Body-type coarse target —
   infra SHIPPED (2026-07-08):** `analysis/makemodel/bodytype.py` maps
   the DVSA _model_ string (prefix-matched, trim-suffix-tolerant) to a
   silhouette class {hatchback, saloon, suv, mpv, van, pickup, coupe} at
   **90 % of labelled cars** (45 % hatchback / 32 % suv / ~8 % van+saloon
   / ...). `makemodel-build-uk` now tags `body_type` per crop; train it
   with `makemodel-train-uk <corpus> --target body_type` (the make path
   is byte-identical when `--target make`). The DVSA "UNKNOWN" placeholder
   is now filtered in `normalize_make` (PR #76, #77). **VERDICT
   (2026-07-09, `runs/uk_body_0708_b0`, b0@384, 20ep early-stop @12):
   body_type@1 = 0.694 per-crop / 0.667 per-car — beats make (0.45) by
   ~+24 pp but short of the 80-90 % hope.** The win is CONCENTRATED: the
   two dominant, visually-distinct classes — **hatchback + suv, 77 % of
   cars — read ~0.75 per-car**; the minority classes are weak (saloon
   0.37 / mpv 0.42 / van 0.59 / coupe 0.43 / pickup 0.21 recall),
   capped by rear-view ambiguity + tiny support + label noise (one model
   name → predominant body style mislabels estates/variants), and mostly
   confuse INTO hatchback. So body type is a usable coarse "hatchback vs
   suv vs other" tag for the majority, NOT a clean win — surface it only
   if that split is wanted; inference/showcase wiring is an unbuilt,
   optional follow-up. Reproduce: `.claude/bodytype_eval.py`. Ops: the
   8B VLM spills
   ~17 GB commit to system RAM on the 10 GB 3080 — never run it beside
   panel jobs (it starved an `alpr-run` to death on 2026-07-08).

## Retrains on gate-cleaned labels (2026-09-29 to 2026-10-03)

Step 5 of the 2026-09-28 handoff (the other steps are in [plate_gate.md](plate_gate.md)).

5. **Retrain on the cleaner labels.** Run the build-train playbook (make),
   plus `--target colour|body_type`, head-to-head, and promote only on a
   clear win.
   **Rebuild the corpus; don't reuse `uk_crops_0924_576`.** Checked
   2026-09-29 against the post-gate labels: 28 % of its crops no longer have
   a valid label (23.1 % come from tracks the gate now rejects, 3.9 % from
   tracks now labelled a *different make*), and 45 % of its 7,319 "cars"
   were phantom plates from misreads. Training on wrong labels fits the
   early-peak-then-memorise curve every head showed. Estimated rebuild:
   ~3,900 cars / ~59k crops / ~36 makes (was 62), so judge only
   head-to-head. The top 500 cars would still hold ~55 % of crops, so test
   the per-car cap as a separate run: build-train uncapped vs **max
   crops/car = 30** (the dashboard field; capped runs get `_capN` dirs), each
   compared against production.
   **Progress (2026-10-01):** the uncapped run (`runs/uk_make_0929_b6`,
   corpus `uk_crops_0929_576`: 4,347 cars / 76,033 crops / 62 makes) ran all
   20 epochs at batch 6 (~100 min/epoch); val make@1 peaked **0.932** at epoch
   15 and never fell away (the old corpus peaked at epoch 2, then dropped).
   Its build-train head-to-head is **inconclusive**: +1.3 pp per track (95 % CI
   −2.6..+6.3) on only **67 held-out cars**, because 93 % of the rebuilt
   corpus's cars are also in production's training corpus and are excluded.
   The cap-30 run (`runs/uk_make_1001_b6_cap30`, 39,219 train crops,
   ~69 min/epoch) followed automatically. **To decide:** enrich the unseen
   week `session_20260924_104939` (pulled 2026-09-30) once training frees the
   GPU, then score both candidates on its fresh cars with `makemodel-compare
   <corpus> --candidate <best.pt> --eval-session session_20260924_104939
   --exclude-corpus runs/uk_crops_0924_576 <the other candidate's corpus>`.
   Pass the same exclusions for both runs so they score identical cars
   (`--exclude-corpus` replaces the default production-corpus exclusion, so
   name 0924 explicitly; each run always excludes its own corpus).
   **Fresh cars are scarce on this street:** the cap-30 run (best val 0.934,
   early-stopped at epoch 13) also tied production on the 67 cars (+0.0 pp),
   and `session_20260930_211033` (one day) had 124 DVSA-labelled cars of which
   only **15** were in no training corpus (~88 % regulars). A week gives
   maybe ~100 fresh cars, so also run each candidate with
   `--include-trained-cars`: every labelled car in the eval sessions, scored
   on its new passes (all sessions after every model's training data). It
   has several hundred cars and matches what the classifier mostly sees, but
   it rewards recognising known cars, so read it beside the fresh-car result
   and use `--out` to keep the two reports apart.
   **Verdict (2026-10-01) — uncapped `uk_make_0929_b6` PROMOTED.** Scored on
   the unseen sessions `session_20260924_104939` + `session_20260930_211033`
   (reports `runs/uk_make_*/compare_{fresh,known}_week0924.json`), per-track
   make@1:

   | model | 117 fresh cars | 710 cars, new passes |
   | --- | --- | --- |
   | production (`uk_make_0924_b6`) | 80.8 % | 93.6 % |
   | **uncapped `uk_make_0929_b6`** | **85.0 %** (+4.2, CI −1.2..+10.0) | **95.1 %** (+1.5, CI +0.0..+2.9) |
   | cap-30 `uk_make_1001_b6_cap30` | 84.4 % (+3.6, CI −0.7..+8.2) | 94.0 % (+0.4, CI −1.2..+1.9) |

   Better than production in both tests and never worse, from cleaner labels;
   the 710-car CI touches zero, so it was a judgement call, not the panel's
   strict "clear win". **The per-car cap doesn't help** (no better fresh,
   −1.1 pp known) — keep the option, don't use it by default. Prior model
   backed up as `makemodel_b0.20261001T224515.pt`; the reinfer playbook was
   queued to follow the week's enrich (done 2026-10-02: all 48 sessions,
   showcase at 4,173 cars).
   **Colour + body type — retrained on `uk_crops_0929_576` and PROMOTED
   (2026-10-03).** EfficientNet-B0 @384, batch 32, 20 epochs / patience 5
   (`runs/uk_colour_1002_b0`: early-stopped at 15, best val 0.906 at epoch
   10; `runs/uk_body_1002_b0`: all 20, best 0.940 at 16 — neither showed the
   old epoch-1-to-3 peak). Same fresh-week head-to-heads, per track:

   | head | fresh cars | all cars, new passes |
   | --- | --- | --- |
   | colour | 76.2 % → **83.3 %** (+7.1, CI +0.6..+13.8; grouped +5.9, CI +0.6..+12.1), 118 cars | 86.9 % → **93.5 %** (+6.6, CI +4.7..+8.6; grouped +5.3), 711 cars |
   | body type | 87.8 % → **89.1 %** (+1.3, CI −2.6..+4.7), 109 cars | 92.8 % → **95.9 %** (+3.1, CI +1.7..+4.5), 674 cars |

   Colour is a clear win on both sets; body type is clear on the large set
   and level on fresh cars (both near ceiling there). Prior models backed
   up as `models/{colour,bodytype}_b0.20261003T*.pt` (untracked); every
   session re-run with both heads via the panel (2026-10-03, 96 jobs, then
   a showcase refresh). **Step 5 is complete** — all three heads now train
   on the label-clean corpus.

## A stopped vehicle captures the crops of passing cars (found 2026-11-14)

Found by chance, checking the first enrich after the inference commands were
merged into one: `session_20261113_080245` came back with red as its most
common colour, 28.1 % of 663 confident tracks, where the eight full sessions
since 9 October sit between 9.2 % and 11.5 %.

The red is confined to one stretch of 13 November. By the hour, the red
share is 7-10 % at 08:00 and 09:00, 92 % at 10:00 (100 of 109 tracks), 50 %
at 11:00 and back to 9 % at 12:00. Inside that stretch all three heads agree
with each other at confidence 1.0: the make is Citroën for 44 of 48 tracks
in the ten minutes from 10:10, the body type hatchback, the colour red, and
the crop source `trajectory`. For the 100 tracks called red in the 10:00
hour, the Orin's own sub-stream colour vote says black (84), blue (8) or
nothing (8).

The 4K snaps show why. A red Citroën was stopped in the road at the far end
of the street. Most tracks get their first snap while the car is at the far
end (the first snap's box centre is above y = 110 of 512 for 65-70 % of
tracks all day), and with a stopped vehicle sitting there on the road
polygon, the trajectory rule's "on-road vehicle nearest the track's box" is
that vehicle. The heads classified the crop they were given correctly. It
was the wrong car.

It is not a tracking fault and not extra traffic: the window holds 186 car
tracks, inside the 44-215 range of the same 90 minutes on 17 other days,
with normal net displacement (median 584 px against 579 px for the rest of
the day).

The plate side coped. Of 488 reads of the stopped car's plate on those
tracks, the static-plate filter marked 465 and the colour check 117; the
plate ended up as the best read of 2 tracks.

It predates the 2026-11 code changes. A scan of every session since
2026-10-01 for ten-minute windows of 8 or more classified cars in which one
make + colour + body type takes at least 70 % found seven episodes, all in
files written before those changes except the last:

| Date | Window | Cars | Returned as |
| --- | --- | --- | --- |
| 2026-10-02 | 11:00-11:10 | 9 of 11 | grey Ford SUV |
| 2026-10-14 | 14:10-14:20 | 8 of 11 | white Peugeot van |
| 2026-10-27 | 09:20-09:30 | 11 of 14 | white DAF van |
| 2026-10-27 | 10:00-10:20, 10:30-10:50 | 48 of 60 | red Kia hatchback |
| 2026-11-03 | 10:10-11:10 | 106 of 113 | silver Renault hatchback |
| 2026-11-13 | 10:00-10:40 | 84 of 92 | red Citroën hatchback |

That is 266 of 26,688 classified cars (1.0 %), and a lower bound: the scan
only sees episodes long and busy enough to fill a ten-minute window, and on
13 November it caught 84 of roughly 130 affected cars.

What it touches: the `/stats` colour and body-type mixes and the
fastest-by-colour board, which use the CNN values for unplated cars. The
make chart uses the register and is not affected. Nothing has been changed;
the open question (review E2.6 (f)) is a guard for the heads like the one
the plates have, for example refusing a vehicle box that recurs at one place
across many tracks, or choosing the candidate that moved between a track's
fire-time and completion-time boxes.

## The stationary-vehicle guard (2026-11-14)

The locator the three heads share (`analysis/vehicle_locator.py`) now leaves
stationary vehicles out of a passing track's candidates. The rule is in
`analysis/stationary.py`; every figure here comes from
`.claude/stationary_vehicle_check.py` over the eleven sessions recorded from
2026-10-01 to 2026-11-14, which reads only saved files (the cached vehicle
boxes, the records, the plate reads and the per-image predictions).

**The rule.**

- A vehicle is standing still around a snap when at least three of the
  tracks nearest in time show the same box (IoU ≥ 0.9), one or more before
  the snap and one or more after. The snap's own track counts if it shows
  the box. Only the three nearest other tracks on each side are looked at,
  and only their snaps at least 10 s away.
- A track is on that vehicle, and keeps it, when the track's own sub-stream
  box (at fire time or at completion) overlaps it by IoU ≥ 0.5.
- Otherwise a candidate box is refused when it is that vehicle (IoU ≥ 0.7)
  or, for a crop chosen without a plate read, when the vehicle stands in
  front of it and covers a quarter or more of it.
- The snap then takes the next candidate, or has no crop.

**What it changes.**

| | |
| --- | --- |
| Trajectory crops changed | 2,566 of 59,747 (4.3 %); 2,026 of them left with no crop |
| ...because the box is the vehicle | 2,099 |
| ...because the box is hidden behind it | 467 |
| Plate-anchored crops changed | 120 of 18,929 (0.6 %) |
| Tracks with a changed crop | 1,147 of 28,466 (4.0 %); 668 with every crop changed |

By session the share of trajectory crops changed runs from 1.4 % to 8.2 %,
and is 30.2 % on `session_20261113_080245`. That is four times the 1.0 %
lower bound of the ten-minute-window scan above. Both directions are hit
alike (7.4 % of L→R crops and 6.7 % of R→L, in the four worst sessions).

**The heads on those crops.** On DVSA-labelled tracks, the saved per-image
predictions on trajectory crops, against the register:

| Trajectory crops | Colour right | Make right |
| --- | --- | --- |
| the guard changes | 24 of 145 (16.6 %) | 19 of 145 (13.1 %) |
| ...the box is the vehicle | 20 of 127 | 16 of 127 |
| ...the box is hidden behind it | 4 of 18 | 3 of 18 |
| all others | 2,828 of 3,506 (80.7 %) | 2,787 of 3,506 (79.5 %) |

**The choice itself.** On the 10,971 snaps where a track's own gated plate
was read, the vehicle holding the plate is known. Ignoring the plate, the
trajectory rule picks it in 86.5 % of them without the guard and 87.3 % with
it. In the 176 snaps where the guard acted, the pick was right in 12 before
and 98 after, with no crop in 76. Of the 12, 7 are boxes partly behind a
stopped vehicle and keep their crop in a real run, where the plate anchors
it. The other 5 are lost: the rule takes the plated vehicle itself for a
stationary one that the track is not on (5 of 10,971, 0.05 %).

**A colour run on 13 November** (a copy of the session, so the files in
`output/` are as they were): 439 crops refused, 355 snaps left with no crop.
Red falls from 28.1 % of 663 confident tracks to 12.0 % of 584. In the 10:00
hour it was 100 of 109 tracks and is 12 of 54; at 11:00, 32 of 64 and now
4 of 43. The session's `_stationary_vehicles.json` names the Citroën's box
from 10:03:36 to 11:23:42 UTC, 420 crops over 132 tracks. Sixteen red
tracks are left in those two hours, among them the Citroën's own track (897)
and the last car to pass it (1826). The 10:00 hour is still above the usual
tenth, so a few wrong crops remain there.

**What the measurements ruled out**, in the order tried:

1. *Counting matches* (the box in 2 of the 6 nearest snaps). It also flags
   a car that drives in and stops, or one pulling away, whose box in its own
   track's snap is the tracked car. Asking for a sighting on each side leaves
   those alone and lost no right crop.
2. *Refusing only the vehicle's box.* The first colour re-run still had 40
   red tracks of 91 in the 10:00 hour. With the Citroën refused, the next
   candidate was often the passing car's own box with the Citroën in front
   of it, and where a pedestrian or a car behind stretched the Citroën's box
   it was no longer recognised (IoU 0.87).
3. *Two sightings, one before and one after, without the snap's own.* Snaps
   fire at fixed places along the road, so two moving cars often leave
   matching boxes. Of 16 labelled crops refused for matching such a
   "vehicle" at IoU 0.7-0.9, 8 were right. With three tracks required, 4
   labelled crops are left in that band and 1 is right.
4. *Hidden by a half.* Accuracy by the share of the box a stationary
   vehicle in front covers: under a quarter, 47 of 59 colour and 49 of 59
   make, as on a clear crop (76 %); a quarter to a half, 2 and 3 of 12; a
   half to three quarters, 2 and 0 of 6. A box *in front of* a stationary
   vehicle is unaffected (89 % and 84 % of 1,159).
5. *Hidden boxes with a plate read in them.* Right in 12 (colour) and 15
   (make) of 19, so they are kept.

**Not covered.** The first and the last car to pass a stopped vehicle
(nothing shows the vehicle on one side of them). A stop that fewer than
three tracks see. A box less than a quarter hidden. And the corpus builder,
which anchors its crops with `plate_anchored_box` directly: on labelled
tracks 39 plate-anchored crops (0.4 %) moved off a stationary vehicle, most
of them a parked car whose box was the tightest round a passing car's plate.

**Cost.** Working out the stationary vehicles takes 2.2 s on the largest
session (15,000 snaps). The first command on a session classifies the
changed snaps a second time, about 3 % of them.

**Applying it.** Nothing in `output/` has been re-inferred. The panel's
reinfer playbook over the sessions since 2026-10-01 (earlier ones if the
`/stats` history matters) rewrites the per-track files, and a showcase
refresh picks them up. Tracks that lose every crop show as "unknown".
