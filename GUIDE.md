# ReactionID: how the project works, and where to change things

This guide is for someone who knows the MUSE experiment and a little Python and C++,
but not this code. It explains what every piece does, the order things run in, and,
in [section 7](#7-how-to-change-things), exactly where to edit to change the model or
its inputs. `README.md` is the short reference; this is the long version.

Contents

1. [The big picture](#1-the-big-picture)
2. [Words used here](#2-words-used-here)
3. [The C++ side (muse/)](#3-the-c-side-muse)
4. [The Python side (reactionID/)](#4-the-python-side-reactionid)
5. [Running everything, step by step](#5-running-everything-step-by-step)
6. [What happens to a column on its way into a model](#6-what-happens-to-a-column-on-its-way-into-a-model)
7. [How to change things](#7-how-to-change-things)
8. [The safety checks, and what a failure means](#8-the-safety-checks-and-what-a-failure-means)
9. [Troubleshooting](#9-troubleshooting)
10. [Known limitations](#10-known-limitations)

---

## 1. The big picture

The goal is to flag events where the beam muon **decayed in flight before or inside
the target**. The positron from such a decay can enter a scattering arm and look like
a scattered muon. The classifier looks at one event's detector response and returns a
**score** between 0 and 1. Above a **threshold**, the event is called a decay.

```
  g4PSI simulation (.root, with MC truth)
        |
        |  muse/script/cook_mc_chain.sh  --  13 cooker stages:
        |  mc2root, SPS, BH, STT, GEM, BM, VETO, GEMtrack, STTtrack,
        |  Vertex, PathLength, reactionID, features
        v
  cooked .root files, one per stage  (reactionID/cooked/)
        |
        |  last stage: the muonDecay_out plugin
        |  calls decayfeatures::Builder once per event
        v
  one CSV row per event, 236 columns  (reactionID/data/train.csv, valid.csv)
        |
        |  Python: train two models, choose thresholds, export to ONNX
        v
  model/decay_gbdt.onnx, model/decay_mlp.onnx
        |
        |  copied into muse, installed with `make install`
        v
  cooker ReactionID plugin: calls the SAME decayfeatures::Builder,
  feeds the vector to ONNX Runtime, writes the "ReactionID" branch
        |
        |  ReactionID_Plots_MC.xml
        v
  classifier vs MC truth: confusion matrix, per-region plots, beta plots
```

The most important design point is that **one piece of C++ builds the feature
vector**, `decayfeatures::Builder`. The CSV used for training and the vector fed to
the model inside the cooker come from the same code, so they cannot drift apart.
`verify_inference.py` checks this for every event.

There are **two models**. They are trained on the same data and exported to the same
kind of file, and the cooker can run either:

- **Boosted trees (GBDT)**, `decay_gbdt.onnx`. This is the default, and it performs better.
- **A neural network (MLP)**, `decay_mlp.onnx`. It is the alternative.

---

## 2. Words used here

| term | meaning in this project |
|---|---|
| **input / feature / column** | One number per event that the model sees, such as `tof_raw` or `stt_z`. There are 135 of them. |
| **NaN** | "Not measured", for example when there is no STT track. It is kept as NaN on purpose, never replaced by -1 or -10000. |
| **label** | The answer the model is taught: `decay_relevant` = 1 if the muon decayed upstream of or inside the target region, else 0. |
| **decay region** | Where the MC muon actually decayed: 0 no decay, 1 upstream (Z < -200 mm), 2 target (\|Z\| and R < 200 mm), 3 downstream. Regions 1 and 2 have label 1. |
| **fit / holdout / validation** | Three separate sets of events. **Fit** (85% of the training CSVs) is what the model learns from. **Holdout** (15% of the training CSVs) is used to stop training and choose the threshold. **Validation** (`data/valid.csv`, a separate MC file) is used once, for the reported numbers. |
| **score, threshold** | Model output in [0,1], and the cut on it. The threshold is chosen so that 90% of labelled decays in the holdout pass it. |
| **efficiency** | The fraction of true (label 1) decays that are flagged. |
| **false-positive rate (FPR)** | The fraction of events with no decay that are wrongly flagged. |
| **purity / precision** | Of the flagged events, the fraction that really are decays. It depends on how common decays are, so it is quoted at the **physical prior**. |
| **physical prior** | How often a real beam muon decays in the labelled window: 1 − exp(−L/βγcτ) ≈ 0.13% at 210 MeV/c. The MC is enriched far above this, to 16%. |
| **AUC** | One number summarising how well the scores separate the two classes: 1.0 is perfect and 0.5 is a coin flip. It does not depend on the threshold. |
| **GBDT** | Gradient-boosted decision trees (scikit-learn `HistGradientBoostingClassifier`): hundreds of small decision trees added together. It handles NaN natively and needs no scaling. |
| **MLP** | Multi-layer perceptron, a plain neural network (PyTorch). It needs its inputs scaled and NaN turned into 0 plus an "is missing" flag. |
| **ONNX** | A standard file format for a trained model. Python writes it, and C++ (ONNX Runtime) reads it. |
| **checkpoint** | The saved MLP (`model/output.pth`): weights, input scaling, and the list of columns. |
| **permutation importance** | How much the AUC drops when one column is shuffled between events. A big drop means the model relies on that column. |
| **ablation** | Retraining without a group of columns, to see what that group is worth. |

---

## 3. The C++ side (muse/)

### Where the columns come from

**`muse/src/decayfeatures/src/decayfeatures.cpp`** (header: `muse/include/decayfeatures.h`)
is a small library, not a plugin, that whichever plugin owns it calls.

- `Builder::attach(plugin)` is called in the host plugin's `startup()`. It finds the
  input branches: `BH_Hits`, `SPS_Hits`, `VETO_Hits`, `BM_Hits`, `TrackHits` (STT),
  `Tracks` (GEM), `Vertices`, and the g4PSI truth leaves. **A missing branch is not
  an error.** Its columns just come out NaN.
- `Builder::build()` is called once per event. **Every column is one call to
  `col("name", value)`, so the list of `col()` calls in `build()` is the list of CSV
  columns, in order.** The header and the rows are produced by the same calls, so
  they cannot disagree.
- There are three helpers, called from `build()`:
  - `writeScintColumns` handles per-wall scintillator columns (`bhc_*`, `spslf_*`, `bm0_*`, and so on).
  - `writeTrackColumns` handles STT/GEM track columns (`stt_*`, `gem_*`).
  - `writePathColumns` handles track-to-scintillator correlation (`*_corr_*`).
- The truth columns (`MuonDecay*`, `decay_region`, `decay_relevant`) are computed in
  `build()` from the g4PSI leaves. `kLabelColumn` in the header names the label.

### The two plugins that use it

- **`muse/src/plugins/analysis/muonDecay_out/`**: `muonDecay_out::process()` calls
  `features.build()` and writes the row to CSV. It exists only to produce training data.
  - Recipe: `recipes/ReactionID/muonDecay_out.xml`.
  - `-c` options: `setOutputFile`, `setMomentum`, `setFileIndex`, `setTargetRegion`.
- **`muse/src/plugins/analysis/ReactionID/`**:
  - `ReactionID::startup()` sets up the cut-based analysis and calls `setupML()`.
  - `setupML()` finds the ONNX file and reads its `input_columns` list. It maps each
    name onto the Builder's column number. **If any name is missing, it names the
    column in the log and switches the classifier off** (the branch stays empty)
    rather than guessing.
  - `runML()` (called first in `process()`) builds the vector, passes it to the model,
    and fills the **`ReactionID` branch**. Its fields are `score`, `threshold`,
    `is_decay`, `inputs_valid`, `model_id` (1 gbdt, 2 mlp), `n_inputs`, `run`, and
    `entry`. They are defined in `muse/include/ReactionIDtree.h`.
  - The rest of `process()` is the older cut-based analysis. It fills the
    `allScattering` branch.
  - `-c` options:

    | option | effect |
    |---|---|
    | `setModel:'"decay_mlp.onnx"'` | which model to run (default `decay_gbdt.onnx`) |
    | `setDecayThreshold:<t>` | override the model's own threshold |
    | `setMomentum:210` | beam momentum (needed on MC; data takes it from slow control) |
    | `setTargetPosition:0` | 0 = LH2 (needed on MC) |

  - `muse/include/onnxmodel.h` is the small ONNX Runtime wrapper. It never throws: if
    the model is missing, the branch stays empty and the cook carries on.
  - Model files are looked up in `~/.muse/shared/ReactionID/`, which `make install`
    fills from `src/plugins/analysis/ReactionID/models/`, then in that source directory.

### The cooker's own judgement of the model

**`muse/src/plugins/analysis/ReactionID/src/Plotting.cpp`** is run by
`recipes/ReactionID/ReactionID_Plots_MC.xml`. It reads the `ReactionID` branch that
the reactionID stage wrote, and MC truth through the same Builder. At the end, the
functions below write PNG/PDF files next to the output file:

- `fill_truth_plots()` (per event) and `draw_truth_plots()` (at the end) produce:
  - the confusion matrix and its log line, `ReactionID truth comparison: TP=… FN=… FP=… TN=…`;
  - agreement by decay region;
  - where the missed decays were (Z, R);
  - the score distribution;
  - the outgoing β plots, including the classifier-vs-cuts comparison on the same
    vertices (log line `ReactionID cut-based, full reconstruction only: …`);
  - the **β timing calibration from truth**. It prints the `load_beta_alignment`
    values to use; the current values, fitted on training file 1, are in
    `reactionID/mc17606_beta_alignment.txt`.

### The driver script

**`muse/script/cook_mc_chain.sh`** runs the 13 stages in order for each file. Its
stage table is the `STAGES=(...)` array near the top. It skips a stage whose output is
newer than its inputs, so after changing one plugin, re-running the same command only
re-cooks what changed. At the end it concatenates the per-file CSVs into `train.csv`
and `valid.csv` (files numbered `--valid-from` or higher go to validation). It also
writes `manifest.json`, which records the tag, files, momentum, recipes and git SHA.
Useful flags: `--list-stages`, `--dry-run`, `--only <stage>`, `--from <stage>`,
`--force`, `-n <events>` (for a quick test).

---

## 4. The Python side (reactionID/)

There are many files, but only a few matter day to day. They fall into four groups.

### 4a. The core library (imported by everything, never run directly)

| file | what it holds |
|---|---|
| `reactionData.py` | **The CSV contract.** `LABEL_COLUMN`; `TRUTH_COLUMNS` and `META_COLUMNS` (never inputs); `EXCLUDED_COLUMNS` (inputs dropped on purpose, each with a reason); `CATEGORICAL_COLUMNS` (IDs one-hot encoded for the MLP). `load_csv()` applies the `has_truth` and `chv_veto` cuts. `feature_columns()` picks the inputs and drops dead columns. `FeatureSpec` and `build_matrix()` do the MLP's one-hot/NaN expansion, and `Standardizer` does its scaling. `holdout_split()` makes the fit/holdout split. `physical_prior()` computes the prior. `GBDT_DEFAULTS` and `gbdt_params()` give the tree settings. |
| `reactionModel.py` | **The MLP.** `reactionLearner` (the network: `DEFAULT_WIDTHS`, `DEFAULT_DROPOUT`), `train_model()` (the training loop, with early stopping on the holdout), `save_model()` and `load_model()` (the checkpoint, including widths, scaling and column list). |
| `reactionPlots.py` | Metrics and matplotlib figures for `reactionID.py`: `predict()`, `working_points()`, `report_metrics()`, `plot_and_test_model_BCE()`. |

### 4b. The main pipeline, in the order you run it

| step | file | reads | writes |
|---|---|---|---|
| 1 | `reactionID.py train` | training CSV(s), `valid.csv` | `model/output.pth` (MLP checkpoint), `bce_*.pdf` |
| 2 | `export_onnx.py --all --verify` | the checkpoint, the training CSV(s) | `model/decay_mlp.onnx`, `model/decay_gbdt.onnx`, `model/decay_gbdt.joblib`. It fits the trees here, and sets both thresholds on the holdout. |
| 3 | *(cooker)* reactionID stage | the ONNX files | `cooked/<tag>_<i>_RID.root` with the `ReactionID` branch |
| 4 | `verify_inference.py` | the RID file, that file's `_features.csv` | pass/fail; `results/parity_*.json` with `--json` |
| 5 | *(cooker)* `ReactionID_Plots_MC.xml` | truth and the RID file | `cooker_plots/*.png`, and the log (`logs/plots_valid_file2.log`) |
| 6 | `make_report.py --plots` | both shipped models, `valid.csv` | `report.json`, `feature_importance.{pdf,md,csv}` |
| 7 | `audit_data.py` | the CSVs, the RID file | `audit.json` (overlap, leakage screen, stale rows) |
| 8 | `make_presentation.py` | everything above | `muse/doc/slides/reaction_id_review.html` |

**`review_pipeline.sh` runs steps 2 to 8 in order.** After retraining (step 1) you
normally just run `./review_pipeline.sh`.

### 4c. Optional studies (run when you want to answer a question)

| file | question it answers | writes |
|---|---|---|
| `tune.py` | Which boosted-tree settings work best? It searches on a split carved from training. | `model/gbdt_params.json`, which `export_onnx.py` then uses |
| `ablate.py` | What is each group of inputs worth? How much comes from which columns are NaN? | `ablation.json` (with `--json`) |
| `compare_batches.py` | Are two MC productions similar enough to pool for training? | printout |
| `benchmark_gbdt.py` | A quick boosted-trees baseline with importances. It uses its **own** quick settings, not the shipped ones, so treat it as a sanity check only. | printout |

### 4d. Legacy (kept for reference)

- `make_slides.py`: an earlier deck. It is still imported by `make_presentation.py`
  for the page template and the two model colours, so do not delete it.
- `make_slides_v2.py`: an earlier deck with some numbers typed in by hand.
  Superseded by `make_presentation.py`.
- `attic/`: superseded code, explained in `attic/README.md`.

### 4e. Where the result files live

| path | contents |
|---|---|
| `data/` | the 21Sep25 production: `train.csv` (file 1), `valid.csv` (file 2), `manifest.json` |
| `data_15Apr25/` | the 15Apr25 production (28 files), training only |
| `mc_merged/`, `root_files/` | raw g4PSI input files |
| `cooked/` | every cooker stage's output, per file |
| `model/` | `output.pth`, the two `.onnx` files, `decay_gbdt.joblib`, `gbdt_params.json` |
| `report.json`, `ablation.json`, `audit.json`, `results/` | numbers the deck reads |
| `cooker_plots/` | the cooker's figures (`calib_file1_*` = file 1, `valid_file2_*` = file 2) |
| `logs/` | the log of every step, including `export.log` and `plots_valid_file2.log` |
| `before_fix/` | a snapshot of the models and reports before the September 2026 review |

---

## 5. Running everything, step by step

### 5.0 Environment

The cooker needs its libraries on the path:

```bash
source /Users/ethancline/MUSE/muse/switch_to_MUSE_path.sh
```

Always call the MUSE cooker as `~/.muse/arm64/bin/cooker`, not just `cooker`. Homebrew
installs an unrelated program with the same name. `cook_mc_chain.sh` and
`review_pipeline.sh` already do this. Run cooker commands **from the `muse/`
directory**, because recipe paths are relative to it.

Python lives in the project's virtual environment: `reactionID/rID/bin/python`, or
`source rID/bin/activate`.

### 5.1 Make the training data (cooker; hours)

From `muse/`, for the two productions currently used:

```bash
# 21Sep25 production: file 1 -> training, file 2 -> validation
./script/cook_mc_chain.sh --tag mc17606_210MeV_LH2 --files 1-2 --momentum 210 \
    --indir ../reactionID/mc_merged --outdir ../reactionID/cooked \
    --csvdir ../reactionID/data --valid-from 2

# 15Apr25 production: all 28 files -> training only
./script/cook_mc_chain.sh --tag mc17606_mu_positive_210MeV_LH2_15Apr25 --files 1-28 --momentum 210 \
    --indir ../reactionID/root_files --outdir ../reactionID/cooked \
    --csvdir ../reactionID/data_15Apr25 --valid-from 99 --jobs 3
```

If you merge raw g4PSI files, use `g4PSI_merge`, never `hadd`. The README explains why.

### 5.2 Train the MLP (Python; ~25 minutes on the laptop GPU)

From `reactionID/`:

```bash
rID/bin/python reactionID.py train --train data/train.csv,data_15Apr25/train.csv \
    --valid data/valid.csv --holdout-frac 0.15
```

The comma joins the two productions into one training set. The run prints the
holdout loss every 5 epochs and stops 25 epochs after the best one. The best
checkpoint is `model/output.pth`. It then prints the validation numbers and writes
`bce_*.pdf`.

### 5.3 Export, deploy, check, report, deck (about 15 minutes)

```bash
./review_pipeline.sh > logs/review_pipeline.log 2>&1
```

This runs `export_onnx.py` (fitting the trees too), copies the ONNX files into
`muse/src/plugins/analysis/ReactionID/models/`, runs `make install`, and cooks the
validation file with each model. It then runs both parity checks, the cooker truth
plots, `make_report.py`, `audit_data.py` and `make_presentation.py`. If any check fails
it stops (`set -e`), and the failing step is at the end of the log.

### 5.4 Optional

```bash
rID/bin/python ablate.py --train data/train.csv,data_15Apr25/train.csv --valid data/valid.csv --json ablation.json  # ~25 min
rID/bin/python tune.py --train data/train.csv data_15Apr25/train.csv --valid data/valid.csv                       # ~30 min
rID/bin/python make_presentation.py      # rebuild the deck after either
```

---

## 6. What happens to a column on its way into a model

Follow a single column, for example `stt_z`, from the cooker to the models:

1. **C++**: `Builder::build()` calls `col("stt_z", t.position.Z())` in
   `writeTrackColumns`. If there is no STT track it writes NaN instead.
2. **CSV**: `muonDecay_out` writes it as one column of the row.
3. **`reactionData.load_csv()`** reads the CSV(s) and keeps events with
   `has_truth == 1` and `chv_veto == 0`.
4. **`reactionData.feature_columns()`** decides whether it is an input. It is not if:
   - it is in `TRUTH_COLUMNS`, `META_COLUMNS` or `EXCLUDED_COLUMNS`; or
   - `zero_information_columns()` finds it can carry no information in the training
     data. That means it is constant, or always NaN, or constant where measured with a
     missing-pattern identical to another column. It prints what it dropped and why.

   The result is the ordered list of **135 input columns**.
5. **Boosted trees**: take those 135 columns as raw numbers, NaN included (see
   `export_onnx.export_gbdt`).
6. **MLP**: `FeatureSpec.fit()` (on the fit set) records two things:
   - which columns are categorical, from `CATEGORICAL_COLUMNS`, and their values;
   - which columns can be NaN.

   `build_matrix()` then expands the columns. A categorical column becomes one 0/1
   column per value plus an `_isnan` column. A numeric column that can be NaN becomes
   the value (NaN→0) plus an `_isnan` 0/1 column. That turns 135 columns into 438.
   `Standardizer` then subtracts the mean and divides by the standard deviation.
7. **Export**: both ONNX files take the **same 135 raw columns**. The MLP's expansion
   and scaling are built into its ONNX graph (`PreprocessedMLP` in `export_onnx.py`),
   so C++ never re-implements them. The file's metadata stores `input_columns`, the
   135 names in order, together with `threshold`, `label` and provenance.
8. **Cooker**: `ReactionID::setupML()` looks each name up in the Builder
   (`features.indexOf(name)`). Every event, `runML()` copies those 135 values, in that
   order, into the model.

Because the connection is **by name**, adding a column to the Builder does not change
what an existing model sees. A new column only enters a model when you retrain.

---

## 7. How to change things

Each recipe says where to edit and what to re-run. "Retrain and redeploy" always means:

```bash
rID/bin/python reactionID.py train --train data/train.csv,data_15Apr25/train.csv --valid data/valid.csv
./review_pipeline.sh > logs/review_pipeline.log 2>&1
```

### 7a. Add a new input column

Example: the angle between the GEM (incoming) and STT (outgoing) tracks.

1. **C++** in `muse/src/decayfeatures/src/decayfeatures.cpp`, inside
   `Builder::build()`, after the two `writeTrackColumns(...)` calls, where `haveSTT`,
   `haveGEM`, `sttIdx` and `gemIdx` are already defined:

   ```cpp
   // Opening angle between the incoming (GEM) and outgoing (STT) tracks, rad.
   col("gem_stt_angle", (haveGEM && haveSTT)
       ? GEM_Tracks_->tracks[gemIdx].direction.Angle(STT_Tracks_->tracks[sttIdx].direction)
       : kNaN);
   ```

   Rules:
   - Use `kNaN` when the quantity can't be computed. Never use a sentinel like -1.
   - Every event must call `col()` the same number of times, in the same order. So
     write the NaN branch rather than skipping the call. `muonDecay_out` refuses to
     write a row whose length differs from the header.
   - If the column needs a branch the Builder doesn't read yet, add it in
     `Builder::attach()` with `resolve(...)`, and a member in `decayfeatures.h`. Also
     make sure that branch's tree is in the `<source>` of **both**
     `recipes/ReactionID/muonDecay_out.xml` and `recipes/ReactionID/ReactionID.xml`.
     If only the first has it, training sees real values while the cooker sees NaN.
2. **Build**: `cd muse/build && make install`.
3. **Re-cook the CSVs**: re-run the two `cook_mc_chain.sh` commands from section 5.1,
   adding `--only features --force`. You need `--force` here. The Builder is its own
   library (`libdecayFeatures`), and the script's up-to-date check only looks at
   `libmuonDecay_out`, so without it the old CSVs would be kept. Check that the new name
   is in the header: `head -1 data/valid.csv | tr , '\n' | grep gem_stt_angle`.
4. **Nothing to change in Python.** New columns are picked up automatically. If
   `feature_columns()` prints that it dropped yours, it was constant or always NaN:
   check the C++.
5. **Retrain and redeploy.** `verify_inference.py` (inside `review_pipeline.sh`)
   confirms the cooker fills the new column exactly as the CSV does.

### 7b. Stop the model using a column

Add it to `EXCLUDED_COLUMNS` in `reactionData.py`, with the reason as the value:

```python
EXCLUDED_COLUMNS = {
    ...
    "spsrf_time": "absolute time - not calibrated on data",
}
```

The column stays in the CSV, so you can still look at it. Retrain and redeploy.

### 7c. Treat an ID column (bar number, PID, status) as a category

Add it to `CATEGORICAL_COLUMNS` in `reactionData.py`. The MLP then gets one 0/1 input
per value rather than treating bar 7 as "between" bars 6 and 8. The trees are
unaffected. Retrain and redeploy.

### 7d. Change what counts as a "relevant" decay (the label)

- To change the **region boundaries**:
  - The label is computed in `Builder::build()` (the `--- truth ---` block), using
    `targetRadius_` and `targetHalfZ_` (default 200 mm).
  - For the CSV, set them with `-c muonDecay_out:setTargetRegion:<radius_mm>,<half_z_mm>`.
    To make that permanent, add it to the `features` line in the `STAGES` table of
    `cook_mc_chain.sh`, or change the defaults in `decayfeatures.h`.
  - The cooker's truth plots use the Builder's defaults, so **change the defaults in
    `decayfeatures.h` if you want the plots to agree**.
- To define a **new label column**:
  1. Add a `col("my_label", ...)` in the truth block.
  2. Add it to `TRUTH_COLUMNS` in `reactionData.py`.
  3. Set both `kLabelColumn` (in `decayfeatures.h`) and `LABEL_COLUMN` (in
     `reactionData.py`) to it. They must match: the ONNX file records its label, and
     `ReactionID` prints a warning at startup if they differ.

Then re-cook the features, retrain and redeploy.

### 7e. Change the neural network

- **Size and shape**: `DEFAULT_WIDTHS` and `DEFAULT_DROPOUT` at the top of
  `reactionModel.py`. For example, `DEFAULT_WIDTHS = (128, 128)` gives two hidden
  layers of 128. The sizes are saved in the checkpoint, so old checkpoints still load
  and the ONNX export follows automatically.
- **Layer types** (for example a different activation): edit the loop in
  `reactionLearner.__init__`. If you add a layer type ONNX can't export,
  `export_onnx.py` will fail loudly.
- **Training settings** are flags to `reactionID.py train`:

  | flag | default | meaning |
  |---|---|---|
  | `--epochs` | 150 | maximum passes over the data |
  | `--patience` | 25 | stop after this many epochs without the holdout loss improving |
  | `--lr` | 5e-3 | learning rate (a cosine schedule with restarts is in `train_model`) |
  | `--batch-size` | 128 | events per step |
  | `--holdout-frac` | 0.15 | share of training kept for early stopping and thresholds |
  | `--no-categorical` | off | feed IDs as plain numbers instead of one-hot |

- The optimiser and the loss (`Adam`, `BCEWithLogitsLoss`) are set in `train_model()`.

Retrain and redeploy.

### 7f. Change the boosted-tree settings

The shipped trees use `model/gbdt_params.json` (written by `tune.py`). If that file is
absent they use `GBDT_DEFAULTS` in `reactionData.py`. Either edit the JSON by hand
(the `"params"` block; keys are `HistGradientBoostingClassifier` arguments such as
`learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`,
`l2_regularization`), or let `tune.py` search for better ones. Then run
`./review_pipeline.sh`. No MLP retraining is needed, because the trees are fitted
inside `export_onnx.py`.

### 7g. Change the operating point (threshold)

- **Permanently**: `export_onnx.py --target-eff 0.95` sets both thresholds for 95%
  efficiency on the holdout. To use it, edit that line in `review_pipeline.sh` and
  re-run the script.
- **For one cook**: add `-c ReactionID:setDecayThreshold:0.95` to the cooker command.
  The log confirms it with `using the configured threshold …`.
- To see the cost first, the working-point table in the deck (and in `report.json`
  under `working_points`) lists the false-positive rate and precision at 50/80/90/95/99%
  efficiency.

### 7h. Run the other model in the cooker

Add `-c 'ReactionID:setModel:"decay_mlp.onnx"'` to the reactionID-stage cooker
command. The quotes are needed because it is a string argument. The `model_id` field
in the branch records which model ran.

### 7i. Add a new MC production to the training

1. Cook it into its own CSV directory with `cook_mc_chain.sh`, using a high
   `--valid-from` so it is all training (as for 15Apr25 in section 5.1).
2. Check it looks like the others:
   `rID/bin/python compare_batches.py --a data/train.csv --b <newdir>/train.csv --valid data/valid.csv`.
3. Add it to the comma-joined `--train` list in `reactionID.py train`, in
   `review_pipeline.sh` (`TRAIN=`), and in any `ablate.py`/`tune.py` runs. Retrain and
   redeploy.

Keep `data/valid.csv` untouched. Always report on a file no model has trained on.

### 7j. Change or add a cooker truth plot

- The histograms are booked in `ReactionID::book_truth_plots()` (`Plotting.cpp`),
  filled in `fill_truth_plots()` (per event, all events) or in the vertex loop of
  `process_plots()` (per vertex), and drawn in `draw_truth_plots()`.
- Each drawn canvas is a numbered block: `--- 1. confusion matrix`, `--- 2. agreement
  by decay region`, and so on.
- `readable(h)` sets legible axis text, and `gStyle->SetImageScaling(2.0)` gives
  high-resolution PNGs.
- Rebuild with `make install`, then re-run the plots cook (the `ReactionID_Plots_MC.xml`
  step in `review_pipeline.sh`).

---

## 8. The safety checks, and what a failure means

| check | where | what failure means | what to do |
|---|---|---|---|
| ONNX vs native model | `export_onnx.py --verify` (in `logs/export.log`) | the ONNX file does not reproduce the Python model | usually a new layer type or preprocessing step that exported differently; do not ship |
| cooker vs Python, event by event | `verify_inference.py` (`results/parity_*.json`) | the cooker feeds the model a different vector than the CSV | a column filled differently in the two recipes (a branch missing from one `<source>`), or a stale model file in `~/.muse/shared` |
| cooker vs Python confusion matrix | deck slide "judged in the cooker"; compare the log line with `report.json["truth"]` | the two scoring paths disagree | same causes as above, or different thresholds |
| schema errors in `reactionData` | "CSV is missing N expected columns", "does not have the same columns" | the CSV came from a different Builder than the model or the other CSV | re-cook the features, or retrain |
| `ReactionID: model wants column 'x', which this build does not produce` | cooker log | the installed Builder lacks a column the model was trained with | rebuild and install the muse version that made the CSV |

---

## 9. Troubleshooting

- **`cooker: Unknown options: c`** means you ran Homebrew's `cooker`. Use
  `~/.muse/arm64/bin/cooker`.
- **`Library not loaded: @rpath/libmusetree…`**: the library path isn't set. Run
  `source muse/switch_to_MUSE_path.sh`. macOS removes `DYLD_LIBRARY_PATH` when it
  starts system programs such as `nohup`, so don't wrap the cooker in `nohup`.
- **`install_name_tool: no LC_RPATH load command …` during `make install`** is
  harmless. It appears when a library is re-installed; `make install` still succeeds.
- **About 7% of rows in a `ReactionID` branch repeat the previous event.** Cryptor
  blinding skips those events upstream. A row belongs to its event only when `entry`
  equals the tree entry number (`verify_inference.py` and the plots already check this).
- **Recipes not found**: run the cooker from the `muse/` directory.
- **The deck looks stale**: `make_presentation.py` only reads result files, so re-run
  whichever step produces the number that is out of date (see the table in 4b).

---

## 10. Known limitations

These are covered in more detail in the deck (slides "before real data" and "review
findings"):

- **Single setting**: trained on one momentum (210 MeV/c) of pure μ⁺ MC. It has not
  been re-measured on real data.
- **Absolute times**: the most important inputs are raw scintillator times, which
  shift with data calibration.
- **Missing-value signal**: about AUC 0.91 of the performance comes from *which*
  columns are missing (how reconstruction fails), which may differ on data.
- **The chv veto** is applied only to MC and removes the classes at different rates.
- **Label region**: the "target" part of the label is a 200 mm box, much larger than
  the LH2 cell, and many of the misses sit at its edges.
- **`rid_is_decay` is meaningless on MC.** The chain's reactionID stage uses real-data
  timing offsets, so the column is 1 for every vertex. It is not a model input.
