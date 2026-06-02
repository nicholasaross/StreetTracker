# Make/Model Classification — design

Drafted 2026-05-28 after the post-Step-13a pivot to dataset-level
enrichment. **STATUS 2026-06-01:** the §7 open questions are resolved
(see that section) and the DVSA-first prong has shipped (PRs
#41/#42/#43 -- see CLAUDE.md "Make/model classification"). This doc is
now the live spec for the remaining **CompCars CNN** work; start at §9
step 1 (request CompCars access -- the multi-day blocker).

## 1. Goal

Add a make/model attribute to each finalised vehicle track and to the
per-vehicle aggregated record. UK-market accuracy is the binding
constraint: predictions need to be recognisable to an operator who
visually IDs cars on the same street every day. A "Ford Focus 2018"
prediction on what is obviously a Vauxhall Astra is more damaging than
no prediction at all — false reads erode trust in the dashboard.

Scope of this doc: off-device, batch-mode classification on the 4K
snaps the runtime already captures. Live-runtime classification on the
Orin is **out of scope** — see §3.

## 2. Why CompCars

User-picked starter dataset (vs Stanford Cars / both / UK-specific
fine-tune). Rationale:

- **Coverage** — 1716 models across 161 makes, vs Stanford Cars' 196
  US-centric models from 2013. CompCars includes BMW / VW / Audi /
  Mercedes lineups that dominate UK residential roads.
- **Two complementary subsets**:
  - **Web-nature**: 136k web-collected images. Diverse poses + lighting.
    Good for backbone pretraining.
  - **Surveillance-nature**: 50k frontal images from real surveillance
    cameras, 281 models. Much closer match to the StreetTracker scene
    (oblique-front view, motion blur, real-world artefacts).
- **Tradeoff** acknowledged: no widely-published "use this off-the-shelf"
  pretrained checkpoint. We will fine-tune ourselves, vs the
  download-and-go path Stanford Cars would have offered.

## 3. Integration

### Where in the pipeline

Off-device, batch, post-session. Same model as `streettracker alpr-run`:

- Reads `<session>_data.json` for track list + classes
- Reads `<session>/<prefix>_<id>_main_<N>.jpg` for the 4K snaps
- Reads `TrackRecord.main_snap_bboxes` for the BotSORT-tracked vehicle
  crop. Reuses [analysis/alpr/precrop.py](src/streettracker/analysis/alpr/precrop.py)'s
  `_detect_with_hint` path so the make/model classifier sees the same
  cropped vehicle that ALPR's plate detector does — avoids re-doing
  the vehicle-pre-crop work and guarantees we agree on which car is
  the subject.

**Not** wired into the live runtime. The Orin's `pipe_fps=9.93` (Step
13a soak) is already at compute headroom with YOLO + BotSORT +
sub-stream decode; adding another inference pass per snap would either
need a separate GPU stream (complexity) or slow the tracker (regression
on coverage).

### CLI shape (mirrors `alpr-run`)

```
streettracker makemodel <session_dir>
    [--model <path-or-name>]      # default: bundled fine-tuned EfficientNet-B0
    [--top-k 5]                   # how many candidates to keep per image
    [--conf-threshold 0.4]        # below this -> emit null make/model
    [--limit N]                   # process first N images only (smoke tests)
    [--ghost-mask <path>]         # reuse ALPR's ghost-mask for parked cars
    [--pre-crop]                  # default on -- match the alpr-run default
```

Writes:

- `<session>_makemodel.json` — per-image rows, mirrors `_alpr.json`
  shape (`track_id`, `snap_index`, `top_k: [(make, model, year, conf)]`)
- `<session>_makemodel_by_track.json` — per-track confidence-weighted
  best (`track_id`, `make`, `model`, `year`, `conf`, `n_high_conf_reads`)

Optional crops directory `<session>/makemodel_crops/` for debugging
the per-snap input.

### Where the result surfaces

- **`TrackRecord`** (schema additive) gains `make`, `model`, `year`,
  `make_model_conf`. All four `str | float | None`. Set by a
  `streettracker makemodel` post-process pass that rewrites
  `data.json` — same pattern as `streettracker recolor`.
- **`vehicles.json`** (per-vehicle aggregator) gains a `make_models`
  histogram and a `top_make_model` field. Voted across visits using
  confidence-weighted aggregation, same shape as the existing
  `colors` and `directions` histograms.
- **Dashboard** — out of scope for v1; once the data is reliable, a
  "today's vehicle population by make" panel is a natural addition.

## 4. Model architecture

Recommendation: **EfficientNet-B0** backbone, fine-tuned on CompCars'
surveillance-nature subset starting from ImageNet weights, with a
multi-head classifier:

- Head 1: **make** (~161 classes). Coarse; high accuracy expected.
- Head 2: **model** (~1716 classes total, ~281 in surveillance subset).
  Fine; lower accuracy expected.
- Head 3 (optional): **year** (binned to 5-year ranges).

Reasoning:

- B0 has ~5M params, fits in the existing CPU/GPU budget for batch
  alpr-run-style passes. Bigger backbones (B3, ConvNeXt-Tiny) trade
  ~3-5% accuracy for ~3x compute — not worth it for v1 since the
  bottleneck is training data quality + dataset domain match, not
  backbone capacity.
- Multi-head separately optimises a coarse vs fine objective. Make is
  a smaller decision and recoverable from logo / grille shape; model
  is harder and benefits from sharing features with make. A single
  flat head over (make × model) wastes capacity.
- Avoid ViT-family for v1: needs more training data than CompCars'
  per-class median to fit well without strong augmentation.

If the surveillance-nature fine-tune underperforms (val top-1 <60%),
fall back to fine-tuning from a web-nature-trained checkpoint
(domain-adapt to surveillance).

## 5. Training data path

1. **Acquire CompCars** — original dataset distributed via [CUHK
   site](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html);
   request access form. ~12 GB total. HuggingFace and Kaggle mirrors
   exist for the surveillance subset specifically.

2. **Preprocess** — surveillance-nature is already car-cropped frontal
   shots; minimal work. Resize to 224x224, normalise to ImageNet stats.
   For web-nature pretrain, run the vehicle pre-crop (same YOLO COCO
   pass we use in `analysis/alpr/precrop.py`) to align with
   surveillance subset's cropped style.

3. **Splits** — by `car_id` (CompCars provides this), not by image,
   so the same physical car can't leak across train / val. 80/20
   split. Stratify by make to keep rare makes representative in val.

4. **Augmentation**:
   - Random crop ±10% + resize back to 224
   - Brightness / contrast ±20%
   - Slight rotation ±5° (UK cars sit at oblique angles in the
     scene; over-rotating breaks the make/model intuition since
     people identify cars from canonical angles)
   - Horizontal flip — **off**. CompCars labels are pose-specific;
     flipping a right-driver-side car into a left-driver-side car
     is a domain mismatch for the surveillance subset's frontal angle.

5. **Training**:
   - Loss: weighted cross-entropy per head, summed. Weight rare makes
     by inverse frequency.
   - Optimizer: AdamW, LR 1e-4 with cosine decay, weight decay 1e-2.
   - Batch size 64 on the RTX 3080. ~1 hour per epoch. Target 20
     epochs with early stopping on val top-1.
   - Mixed precision (`torch.autocast`) — the 3080 supports it.

6. **Target metrics** (val on CompCars surveillance):
   - Make top-1 ≥ 85%
   - Model top-1 ≥ 60%, top-5 ≥ 85%

## 6. Validation on StreetTracker data

CompCars val accuracy is necessary but not sufficient. Final check is
a **hand-labelled subset of the soak data**:

- Sample ~100 cars from the most-recent multi-hour soak
- Operator labels make + model from visual + plate-rules-lookup (DVLA
  data via plate is the gold standard for UK)
- Report top-1, top-5 on this set
- Confusion matrix to spot systematic failures (e.g. all SUVs labelled
  as Audi Q5 → backbone is grille-fixated)

This validation set becomes the **regression test** for any future
fine-tune.

## 7. Open questions for the user

**RESOLVED 2026-06-01** (operator decisions during the kickoff):
- (1) DVSA integration is **built + shipped** — the DVSA MOT API (not
  DVLA) gives make/model/year; `dvsa-label`/`dvsa-apply` auto-label
  readable plates *and* seed the validation set. This is the DVSA-first
  prong (PRs #41/#42/#43).
- (3) Year granularity = **exact** (not binned).
- (4) v1 = **batch** (post-session); live runtime classification stays
  out of scope.
- (5) Train on the **local RTX 3080** (~10h), not cloud.
- (2) Ship bar: surface **make-only when model-conf is low**; targets
  make top-1 ≥85 %, model top-1 ≥60 %. Revisit after the hand-labelled
  validation pass.

No blockers remain except the CompCars dataset-access request (§9 #1).
The original questions, for reference:

1. **DVLA integration?** UK plate → make + model + year is freely
   queryable for owners and via paid APIs for others. For *recurring*
   plates, this gives a cheap ground truth — we could use it to
   auto-label the validation set, and longer-term to bootstrap a
   StreetTracker-specific fine-tune set. Privacy / legality of
   automated DVLA lookups for non-owners is the question.

2. **Acceptable miss-rate.** What's the user-facing threshold for
   "ship it"? Make top-1 ≥ 85% is achievable; model top-1 is harder.
   If "model unknown but make confident" is useful, the v1 cut can
   surface make only when model conf < threshold.

3. **Year granularity.** CompCars labels exact year. Most UK
   operators care about *generation* (pre-2018 Focus vs post-2018
   Focus) — finer than that is noise. Bin to 5-year ranges?

4. **Live vs batch trade-off.** v1 is batch (post-session). If the
   dashboard needs near-live make/model, would need a separate
   pipeline (likely a second model running off the snapshotter
   stream, decoupled from BotSORT). Out of scope for v1; flag if
   it's the eventual target.

5. **Compute for training.** RTX 3080 ~10h end-to-end including data
   exploration / hyperparameter search. Local is fine; if the user
   would rather rent a cloud GPU for speed, a 4090 cuts that to ~3h.

## 8. Effort estimate

| Phase | Hours |
|---|---|
| CompCars download + access form + preprocess | 4-8 |
| Training pipeline (loader, train script, val loop, logging) | 6-10 |
| Fine-tune + HP search | 6-10 |
| Hand-labelled soak validation set (~100 cars) | 4-6 |
| CLI scaffold (`streettracker makemodel`) | 4-6 |
| Schema additions + vehicles.json integration | 3-5 |
| Tests | 3-5 |
| **Total** | **30-50h** |

(For context, the entire ANPR Step 7-14 trajectory was ~40h. Make/model
is comparable in scope to "alpr-run wholesale port" by itself.)

## 9. Suggested decision sequence

Once the open questions at §7 are resolved, the natural sequence is:

1. **Get CompCars access form approved.** Multi-day turnaround,
   blocking everything else.
2. **Hand-label a 100-car validation set** while waiting. Doesn't
   need the model — it's the regression test we'll use throughout.
3. **Stand up the training pipeline + smoke-train on a small subset.**
   Verifies the loader, augmentation, and val loop before paying
   the full training cost.
4. **Full fine-tune** on the surveillance subset.
5. **Validate on the hand-labelled set.** If acceptable, scaffold
   the CLI + ship. If not, iterate on training data / loss / backbone
   before doing the CLI work.
