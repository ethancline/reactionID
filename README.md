# reactionID

Identifies muon decay-in-flight events from MUSE cooker output.

**New here? Read [GUIDE.md](GUIDE.md) first.** It explains how the project works end
to end, what every file does and in which order things run, and where to edit to
change the model or its inputs. This README is the short reference.

## Pipeline

```
g4PSI MC .root  x N
      |
      |  g4PSI_merge   (NOT hadd - see below)
      v
merged .root, RunInfo preserved
      |
      |  muse/script/cook_mc_chain.sh          (12 cooker stages)
      v
<tag>_<i>_features.csv                          one row per event, 131 columns
      |
      |  merged with a schema check + manifest.json
      v
data/train.csv, data/valid.csv
      |
      |  reactionID.py / benchmark_gbdt.py
      v
model/output.pth, bce_*.pdf
```

### Merging MC files

**Merge with `g4PSI_merge`, never `hadd`.** `hadd` cannot merge the custom
`MRTRunInfo` class, so it silently replaces it with a bare `TObject`; reading
`runNumber` off that returns garbage. Timing alignments are selected by run number
at several stages of the analysis, so a merged file with a broken `RunInfo` gets
the wrong calibration period applied and every TOF-derived quantity is wrong -
with no error anywhere.

`g4PSI_merge` clones the `RunInfo` from the first input, sums `nrOfEvents`, and
writes it back:

```bash
g4PSI_merge -o mc17606_210MeV_LH2_1.root mc17606_..._{1,2,3,4,5,6,7}.root   # train
g4PSI_merge -o mc17606_210MeV_LH2_2.root mc17606_..._{8,9,10}.root          # valid
```

Check it survived before cooking:

```bash
root -l -b -q -e 'gSystem->Load("libmusetree"); TFile f("merged.root"); f.Get("RunInfo")->Dump();'
# expect class=MRTRunInfo and the real runNumber, not class=TObject
```

Merging first also gives the `ReactionID_Plots` recipe ten times the statistics in
a single cooker run, instead of needing the outputs merged afterwards.

### Regenerating the dataset

From the **muse repository root** (recipe paths are relative to the CWD):

```bash
# the 21Sep25 production as currently used: file 1 -> train, file 2 -> validation
./script/cook_mc_chain.sh --tag mc17606_210MeV_LH2 --files 1-2 --momentum 210 \
    --indir ../reactionID/mc_merged --outdir ../reactionID/cooked \
    --csvdir ../reactionID/data --valid-from 2
```

(GUIDE.md section 5.1 also gives the command for the 15Apr25 production.)

Stages are skipped when their output is newer than all of their inputs, so after
changing one plugin you can re-run the whole command and only the affected stages
cook. `--only <stage>` / `--from <stage>` narrow it further; `--list-stages` prints
the chain in order; `--dry-run` shows the cooker commands without running them.

The merge writes `manifest.json` next to the CSVs recording the tag, file range,
train/validation split, momentum, recipe list and the muse git SHA, so a dataset can
be traced back to the code that produced it.

### Training and evaluating

```bash
TRAIN=data/train.csv,data_15Apr25/train.csv       # both productions, pooled
python reactionID.py train --train $TRAIN --valid data/valid.csv --holdout-frac 0.15
python export_onnx.py --all --verify --train $TRAIN  # fits the trees, sets both thresholds
python make_report.py --plots                        # scores both shipped models on valid.csv
python benchmark_gbdt.py   --train data/train.csv --valid data/valid.csv
```

**Nothing is chosen on `valid.csv`.** 15% of the training data is held out
(stratified, seeded, recorded in the checkpoint). The MLP early-stops on it, and
`export_onnx.py` rebuilds the same split to set both models' thresholds at 90%
signal efficiency on it. `make_report.py` then loads the shipped models - the
trees from `model/decay_gbdt.joblib`, not a refit - and scores `valid.csv` once at
those fixed thresholds. Efficiency there comes out near 90%, not exactly on it,
which is the sign it was not tuned to that file.

## Deploying the model into the cooker

The trained model runs inside the cooker's `ReactionID` plugin, which writes a
per-event decay verdict to a `ReactionID` branch.

```bash
python export_onnx.py --all --verify        # -> model/decay_{gbdt,mlp}.onnx
cp model/*.onnx ../muse/src/plugins/analysis/ReactionID/models/
cd ../muse/build && make install             # installs to ~/.muse/shared/ReactionID/

cd .. && ./script/cook_mc_chain.sh --tag $TAG --files 1-10 --momentum 210 \
    --outdir ../reactionID/root_files --csvdir ../reactionID/data --only reactionID
```

Both models export to one contract - `float32[N,135]` raw columns in, one
probability out - so the plugin has a single code path and the model is chosen by
configuration:

```
-c ReactionID:setModel:'"decay_mlp.onnx"'
```

The MLP's one-hot expansion and standardisation are baked into its ONNX graph, so
the C++ never reimplements them. Each file carries its own `input_columns`,
`threshold` and `model_tag` in ONNX metadata; the plugin maps those names onto the
feature builder at startup and refuses to run if any column is unknown.
`-c ReactionID:setDecayThreshold:<t>` overrides the model's threshold (it used to be
silently overwritten by it).

**The feature vector is built once,** by `decayfeatures::Builder`
(`muse/include/decayfeatures.h`). `muonDecay_out` writes it to CSV for training and
`ReactionID` feeds it to the model, so training and inference cannot drift apart.
`verify_inference.py` checks that end to end:

```bash
python verify_inference.py --rid <run>_RID.root --csv <run>_features.csv [--json parity.json]
```

### Reading the ReactionID branch

| field | meaning |
|---|---|
| `score` | decay probability; NaN when nothing scored the event |
| `is_decay` | `score > threshold` |
| `threshold` | operating point, from the model's own metadata |
| `inputs_valid` | a model ran on this event |
| `model_id` | 1 gbdt, 2 mlp, 0 none |
| `run`, `entry` | provenance for joining |

**Check `entry` before trusting a row.** Chef fills the output tree for every
event, but a plugin whose `process()` was skipped upstream (cryptor's blinding)
never updates the object, so the previous event's verdict is written again. A row
is its own result only when `entry` equals the tree entry index.

## Files

| file | role |
|---|---|
| `reactionData.py` | CSV contract: which columns are inputs, one-hot levels, NaN handling, standardisation |
| `reactionModel.py` | the MLP, training loop, checkpointing |
| `reactionPlots.py` | metrics and figures |
| `reactionID.py` | CLI entry point |
| `benchmark_gbdt.py` | boosted-trees baseline and permutation feature importance |
| `export_onnx.py` | exports both models to ONNX and verifies them against the originals |
| `verify_inference.py` | compares the cooker's scores against Python's, event by event |
| `make_report.py` | scores the shipped models on `valid.csv` into `report.json` |
| `audit_data.py` | train/validation overlap, single-input leakage screen, stale-row count -> `audit.json` |
| `review_pipeline.sh` | everything after training, in order: export, deploy, cook, parity checks, cooker plots, report, deck |
| `make_presentation.py` | the review deck, `muse/doc/slides/reaction_id_review.html`, built only from result files |
| `make_slides.py`, `make_slides_v2.py` | earlier decks; v2 hardcodes some numbers in a `MEASURED` dict |
| `attic/` | superseded code, see `attic/README.md` |

## Things to know about the data

**The columns are defined in C++.** `muse/src/plugins/analysis/muonDecay_out/src/muonDecay_out.cpp`
emits the header and the rows from the same `col()` calls, so they cannot disagree.
`reactionData.py` is the other half of the contract and refuses to run if the CSV
does not carry the columns a checkpoint was trained on.

**NaN means "not measured", not zero.** Every column that can be absent gets an
`_isnan` companion feature and the NaN itself is zeroed, so the network is told the
value is missing rather than being handed a sentinel like `-10000` to read as a
number.

**Cuts are columns, not filters.** The exporter writes a row for every event and
exports `chv_veto` and `has_truth` as columns; `reactionData.load_csv` applies them
where you can see it happening.

**The class prior is not physical.** 57% of the MC muons decay somewhere; the
label (`decay_relevant`, a decay upstream of or inside the target) is positive for
13% of events, 16% after the chv cut. A 210 MeV/c muon has βγ ≈ 2.0 and a decay
length of ~1.3 km, so the real probability over the labelled window
(Z ≈ −1500 to +200 mm) is about 0.13% - `reactionData.physical_prior` computes it
and `make_report.py` quotes precision there, with the false-positive rate measured
on true non-decays only. Accuracy and average precision on this sample are not
physics numbers.

**The chv cut is MC-only and not class-blind.** `chv_veto` comes from the g4PSI
`CHVL/CHVR_Time` leaves. It removes 19% of no-decay events, 1% of upstream decays,
5% of target decays and 26% of downstream decays, so it shapes the training
population - and on data it is not applied unless the trigger vetoes the same thing.

**The label mixes two problems.** Decay vertices span Z ∈ [−1500, +5000] mm; only
about 10% are in the target region. A decay metres downstream of SPS is a different
problem from one that fakes target scattering. `decay_region` (0 none, 1 upstream,
2 target, 3 downstream) splits them, and the metrics are broken out by it.

**The SPS PID columns are constant.** In the current CSVs `sps_corr_pid` is always 4,
`sps_corr_real_pid` always −11 and `sps_corr_second_pid` always −13 wherever they are
measured: the SPS recipe never runs the RF-peak calibration the PID assignment needs.
They are excluded in `reactionData.EXCLUDED_COLUMNS`. On this pure-μ⁺ MC the BH PID
is constant too (`bh_pid` = 1, `bhd_corr_real_pid` = −13), so the dead-column check
drops those as well.

**MC needs a TOF alignment override for the beta_out check.** Run 17606 falls into
the `<run nr="10000">` block of `ReactionID.xml`, so the MC is handed that period's
real-beam per-bar offsets (~21-30 ns). The MC `tof_raw` is already right - median
7.51 ns against 7.46 ns expected for an elastic 210 MeV/c muon over ~2 m - so
subtracting those drives `out_tof` negative and beta collapses to ~0.22 with most
entries in underflow. Overriding the alignment puts the elastic peak back at 0.905
(expected 0.894):

```bash
Z0=$(python3 -c "print(', '.join(['1.7']*18))")
Z1=$(python3 -c "print(', '.join(['31.7']*18))")   # beta_shift[1] = -30 + p
cooker recipes/ReactionID/ReactionID_Plots.xml <inputs> out.root \
    -c ReactionID:setMomentum:210 \
    -c "ReactionID:load_beta_alignment:0,\"$Z0\"" \
    -c "ReactionID:load_beta_alignment:1,\"$Z1\""
```

The 1.7 ns is only good enough to make the comparison plot readable. The per-bar
alignment in `mc17606_beta_alignment.txt` is fitted from MC truth by
`ReactionID_Plots_MC.xml` on **training** file 1, and applied when plotting
validation file 2, so the cut-based benchmark is not calibrated on the events it is
scored on. It is deliberately not written into the shared init XML.

**The model is not calibrated for real data.** An earlier model (threshold 0.807,
trained on a different label) flagged 96% of run-24444 events; that has not been
re-measured for the current models. The training sample is pure μ⁺ at a single
momentum; a real run has a different beam composition (BH PIDs come back as a mix of
-11, -13 and 211 rather than all 13), and the top inputs are absolute scintillator
times, which move with data calibration. Treat the real-data score as uncalibrated
until the model is retrained or reweighted on data.

**Use `recipes/Vertex/Vertex.xml` for training data, not `recipes/tracking/Vertex_sim.xml`.**
They run different plugins. `VertexReconstruction` (the former) forms the geometric
vertex for every GEM×STT pair with no selection; `VertexRecon` (the latter) keeps only
vertices passing `is_target_vertex` and a BH PID cut, which is ~0.2% of events on this
MC and leaves the `vtx_*` columns empty. The chain uses the former, and the 13
`vtx_*` columns (position, θ, DOCA, their uncertainties and pulls, arm) are filled
for ~67% of events. The reaction flags VertexReconstruction never sets are not
exported from it; ReactionID's versions arrive as `rid_*` and are excluded as inputs.
