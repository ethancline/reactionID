"""Loading and validating the feature CSVs written by the cooker's muonDecay_out plugin.

The plugin builds its header from the same calls that write the rows, so the header
is authoritative. This module is the other half of that contract: it checks the
columns it was handed against what the model was trained on, and refuses to guess.
"""

import json
import os

import numpy as np
import pandas as pd

# Columns that are labels or bookkeeping, never model inputs.
#
# The label is decay_relevant, not the raw MuonDecay flag: a decay downstream of
# the target cannot contaminate a scattering measurement, and 72% of the decays in
# this sample are downstream. Training on MuonDecay spends most of the model's
# capacity separating events that do not matter. decay_region stays in the meta
# block so the per-region breakdown still works.
LABEL_COLUMN = "decay_relevant"
TRUTH_COLUMNS = [
    "MuonDecay",
    "decay_relevant",
    "decay_region",
    "MuonDecay_X",
    "MuonDecay_Y",
    "MuonDecay_Z",
    "MuonDecay_Mom",
    "MuonDecay_DirX",
    "MuonDecay_DirY",
    "MuonDecay_DirZ",
    "MuonDecay_W",
]
META_COLUMNS = ["run", "event", "entry", "file_index", "has_truth"]

# Excluded on purpose, with the reason, rather than silently dropped:
EXCLUDED_COLUMNS = {
    # sc_detector::find_pid() picks the species by comparing the hit's RF phase
    # against fit_rf_mean[3], which only sc_plugin::calib_rf_peak() ever sets.
    # The BH and BM recipes run it; SPS_monitor_with_tree.xml and
    # VETO_monitor_with_tree.xml do not, so fit_rf_mean stays all-zero there and
    # the strict < in the comparison always selects the first hypothesis. Every
    # SPS PID column is therefore a constant (observed: pid == 4, real_pid == -11,
    # second_pid == -13) and carries no information at all.
    "sps_corr_pid": "SPS recipe never runs calib_rf_peak - constant",
    "sps_corr_real_pid": "SPS recipe never runs calib_rf_peak - constant",
    "sps_corr_second_pid": "SPS recipe never runs calib_rf_peak - constant",
    # ReactionID's own cut-based decay verdict. Kept in the file as the benchmark
    # to compare the classifier against, but training on it would be circular.
    "rid_is_decay": "physics-cut benchmark, not an input",
    # RF phase. Real physics - it is a bunch-referenced arrival time, so a decay
    # positron at beta = 1 arrives at a different phase than the muon would - and
    # it is not MC truth: it is built from digitised TDC times only. But in this
    # MC its power is flattering. The bunch is one clean Gaussian per species
    # (0.48 ns for mu/pi, 0.32 ns for e) with no tails and no run-to-run drift,
    # the RF clock is smeared by only 50 ps, the sample is pure mu+ so no species
    # overlap, and SPS/VETO never run calib_rf_peak so their phase is not even
    # centred per bar on real data. It is also redundant: RF alone reaches AUC
    # 0.975, but dropping all fourteen columns leaves the model at 0.9976 against
    # 0.9975 with them. Free to remove, so remove it - the same arrival-time
    # information stays in the per-wall times and tof_raw.
    **{c: "RF phase - optimistic in MC, and redundant with the wall times"
       for c in ("bhc_rf", "bhd_rf", "spslf_rf", "spslr_rf", "spsrf_rf", "spsrr_rf", "veto_rf",
                 "bm0_rf", "bm1_rf", "bm2_rf", "bm2_oot_rf",
                 "bhc_corr_rf", "bhd_corr_rf", "sps_corr_rf")},
    # Everything else read off ReactionID's allScattering branch. These are
    # ReactionID's *outputs*, so they cannot be inputs to ReactionID's model: in
    # the plugin the classifier runs before the vertex loop that fills them, and
    # the builder read the previous event's values. That skew made 14% of cooked
    # scores disagree with Python; shifting these columns by one event reproduced
    # the cooker to 0.001%. The ablation put their combined worth at 0.99x, so
    # nothing is lost. Kept in the CSV for diagnostics.
    "n_scatters": "ReactionID output - circular, and stale inside the plugin",
    "n_good_scatters": "ReactionID output - circular, and stale inside the plugin",
    "rid_theta": "ReactionID output - circular, and stale inside the plugin",
    "rid_doca": "ReactionID output - circular, and stale inside the plugin",
    "rid_side": "ReactionID output - circular, and stale inside the plugin",
    "rid_id": "ReactionID output - circular, and stale inside the plugin",
    "rid_t_target": "ReactionID output - circular, and stale inside the plugin",
    "rid_is_target": "ReactionID output - circular, and stale inside the plugin",
    "rid_is_good_doca": "ReactionID output - circular, and stale inside the plugin",
    "rid_is_intime_tof": "ReactionID output - circular, and stale inside the plugin",
}


def zero_information_columns(df, columns):
    """Columns that cannot possibly inform the model, with a reason for each.

    A column is uninformative when its measured values are all identical *and* it
    is either never or always missing - so neither the value nor the companion
    _isnan indicator varies across events.

    This is a mechanical check on purpose. Every dead column found so far was dead
    silently, and several of them (the out-of-time hit columns, the PID columns, the
    single-setting kinematics) would start varying on real data or on a mixed-species
    sample. A model trained while they were frozen has no defined behaviour when
    they move, so they must be dropped loudly rather than carried as constants.
    """
    dead = {}
    # Columns constant where measured are only informative through their
    # missingness. That is genuine information, but in this schema it is normally
    # already carried by an explicit companion flag (bhc_corr_ok, stt_valid,
    # n_vertices, ...). Only drop such a column once some other column is found to
    # have exactly the same missingness pattern, so nothing is lost silently.
    masks = {}
    constant_where_measured = {}

    for c in columns:
        v = df[c].to_numpy(dtype=np.float64)
        nan = np.isnan(v)
        if nan.all():
            dead[c] = "never measured"
            continue
        if len(np.unique(v[~nan])) > 1:
            masks.setdefault(nan.tobytes(), []).append(c)
            continue
        if not nan.any():
            dead[c] = f"constant at {v[~nan][0]:g}"
        else:
            constant_where_measured[c] = (nan.tobytes(), v[~nan][0])

    for c, (key, value) in constant_where_measured.items():
        twin = next((o for o in masks.get(key, []) if o != c), None)
        if twin is not None:
            dead[c] = f"constant at {value:g} where measured; missingness duplicates {twin}"
    return dead

# Bar and plane identifiers are categorical: bar 7 is not 'between' bars 6 and 8 in
# any sense the network should exploit. One-hot these rather than feeding the index.
CATEGORICAL_COLUMNS = [
    "bhc_bar",
    "bhd_bar",
    "spslf_bar",
    "spslr_bar",
    "spsrf_bar",
    "spsrr_bar",
    "veto_bar",
    "bhc_corr_bar",
    "bhd_corr_bar",
    "sps_corr_bar",
    "bhc_corr_pid",
    "bhd_corr_pid",
    "sps_corr_pid",
    "bhc_corr_real_pid",
    "bhd_corr_real_pid",
    "bhc_corr_second_pid",
    "bhd_corr_second_pid",
    "bh_pid",
    "rid_id",
    "sps_side",
    "vtx_side",
    "rid_side",
    "stt_status",
    "gem_status",
    "bm0_bar",
    "bm1_bar",
    "bm2_bar",
]


def feature_columns(df, drop_uninformative=True, verbose=True):
    """Model input columns, in a stable order, given a loaded dataframe.

    Set drop_uninformative=False to see the full exported schema, e.g. when
    auditing which columns the cooker is failing to fill.
    """
    drop = set(TRUTH_COLUMNS) | set(META_COLUMNS) | set(EXCLUDED_COLUMNS)
    cols = [c for c in df.columns if c not in drop]
    if not drop_uninformative:
        return cols

    dead = zero_information_columns(df, cols)
    if dead and verbose:
        print(f"  dropping {len(dead)} uninformative columns:")
        by_reason = {}
        for c, why in dead.items():
            by_reason.setdefault(why, []).append(c)
        for why in sorted(by_reason, key=lambda w: -len(by_reason[w])):
            names = by_reason[why]
            shown = ", ".join(names[:6]) + (f" ... (+{len(names) - 6})" if len(names) > 6 else "")
            print(f"    {why}: {shown}")
    return [c for c in cols if c not in dead]


def load_csv(path, drop_chv_veto=True, require_truth=True):
    """Read one feature CSV, or several joined with commas ("a.csv,b.csv").

    Several files are pooled row-wise - e.g. two MC productions of the same beam
    setting - and must share one schema; compare_batches.py checks they are
    interchangeable before they are pooled for training.

    The plugin writes a row for every event and exports its cuts as columns, so the
    selection is applied here where it is visible, not silently upstream.
    """
    paths = [p for p in str(path).split(",") if p]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"no feature CSV at {p} - run script/cook_mc_chain.sh")

    frames = [pd.read_csv(p) for p in paths]
    for p, f in zip(paths[1:], frames[1:]):
        if list(f.columns) != list(frames[0].columns):
            raise ValueError(f"{p} does not have the same columns as {paths[0]} - cooked with a different builder?")
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"{path} has no '{LABEL_COLUMN}' column; is this a feature CSV?")

    n_all = len(df)
    if require_truth and "has_truth" in df.columns:
        df = df[df["has_truth"] == 1]
    if drop_chv_veto and "chv_veto" in df.columns:
        df = df[df["chv_veto"] == 0]
    df = df.reset_index(drop=True)
    print(f"{' + '.join(os.path.basename(p) for p in paths)}: {len(df)} of {n_all} events after cuts")
    return df


class FeatureSpec:
    """The exact recipe for turning a feature CSV into a model input matrix.

    Both the categorical levels and which numeric columns get an _isnan companion
    have to be pinned by the training set. Deriving them per file makes train and
    validation expand to different widths, which is a silent disaster if the shapes
    happen to agree and a crash if they do not.
    """

    def __init__(self, columns, levels, nan_columns, categorical=True):
        self.columns = list(columns)
        self.levels = {k: list(v) for k, v in levels.items()}
        self.nan_columns = list(nan_columns)
        self.categorical = categorical

    @classmethod
    def fit(cls, df, columns=None, categorical=True):
        cols = list(columns) if columns is not None else feature_columns(df)
        levels, nan_cols = {}, []
        for c in cols:
            v = df[c].to_numpy(dtype=np.float64)
            if categorical and c in CATEGORICAL_COLUMNS:
                levels[c] = [float(x) for x in np.unique(v[~np.isnan(v)])]
            elif np.isnan(v).any():
                nan_cols.append(c)
        return cls(cols, levels, nan_cols, categorical=categorical)

    def names(self):
        out = []
        for c in self.columns:
            if c in self.levels:
                out += [f"{c}=={lv:g}" for lv in self.levels[c]]
                out.append(f"{c}_isnan")
            else:
                out.append(c)
                if c in self.nan_columns:
                    out.append(f"{c}_isnan")
        return out

    def state_dict(self):
        return {"columns": self.columns, "levels": self.levels, "nan_columns": self.nan_columns, "categorical": self.categorical}

    @classmethod
    def from_state(cls, state):
        return cls(state["columns"], state["levels"], state["nan_columns"], categorical=state.get("categorical", True))


def build_matrix(df, spec):
    """Return (X, y, names, meta_df) for a CSV, following a fixed FeatureSpec.

    NaN is the plugin's 'not measured' marker. Columns that can be missing get a
    companion _isnan indicator and the NaN itself is zeroed, so the network is told
    the value is absent instead of being handed a sentinel like -10000 to read as a
    number.
    """
    missing = [c for c in spec.columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing {len(missing)} expected columns: {missing[:8]}"
            f"{' ...' if len(missing) > 8 else ''}\n"
            "The exporter's schema changed - retrain, or point at a matching CSV."
        )

    parts, names = [], []
    unseen = {}
    for c in spec.columns:
        v = df[c].to_numpy(dtype=np.float64)
        if c in spec.levels:
            known = np.zeros(len(v), dtype=bool)
            for lv in spec.levels[c]:
                hit = v == lv
                known |= hit
                parts.append(hit.astype(np.float32))
                names.append(f"{c}=={lv:g}")
            nan = np.isnan(v)
            parts.append(nan.astype(np.float32))
            names.append(f"{c}_isnan")
            # A value the training set never saw one-hots to all zeros, which is
            # indistinguishable from 'absent'. Say so rather than losing it quietly.
            n_unseen = int((~known & ~nan).sum())
            if n_unseen:
                unseen[c] = n_unseen
        else:
            parts.append(np.nan_to_num(v, nan=0.0).astype(np.float32))
            names.append(c)
            if c in spec.nan_columns:
                parts.append(np.isnan(v).astype(np.float32))
                names.append(f"{c}_isnan")

    if unseen:
        top = sorted(unseen.items(), key=lambda kv: -kv[1])[:5]
        print("  warning: values absent from the training set, one-hot to all zeros: " + ", ".join(f"{c} ({n})" for c, n in top))

    X = np.stack(parts, axis=1).astype(np.float32)
    y = df[LABEL_COLUMN].to_numpy(dtype=np.float32)
    meta = df[[c for c in (META_COLUMNS + TRUTH_COLUMNS + ["rid_is_decay", "n_vertices"]) if c in df.columns]].copy()
    return X, y, names, meta


class Standardizer:
    """Per-feature mean/std, fitted on train and carried in the checkpoint.

    Without this the first nn.Linear sees columns spanning -10000 to ~2000 with no
    normalisation in front of it, and a saved model cannot be applied to new data
    because the scaling it implicitly learned is not recorded anywhere.
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, X):
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0  # constant columns pass through untouched
        self.std = std
        return self

    def transform(self, X):
        if self.mean is None:
            raise RuntimeError("Standardizer used before fit()")
        return ((X - self.mean) / self.std).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state(cls, state):
        return cls(mean=np.asarray(state["mean"]), std=np.asarray(state["std"]))


def holdout_split(df, frac=0.15, seed=1234):
    """Split a training frame into (fit, holdout), stratified on the label.

    The holdout is where every choice made after fitting is made: the MLP's early
    stopping epoch and both models' operating thresholds. The validation file is
    then scored once, at those fixed choices, so the numbers reported on it are
    not tuned to it.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError(f"holdout fraction must be in (0, 1), got {frac}")
    rng = np.random.default_rng(seed)
    y = df[LABEL_COLUMN].to_numpy()
    hold = np.zeros(len(df), dtype=bool)
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        hold[rng.choice(idx, int(round(frac * len(idx))), replace=False)] = True
    return df[~hold].reset_index(drop=True), df[hold].reset_index(drop=True)


# Muon lifetime (PDG), mass in MeV/c^2.
MUON_CTAU_M = 658.638
MUON_MASS = 105.658


def physical_prior(momentum, z_lo_mm, z_hi_mm):
    """Probability that a beam muon decays over [z_lo, z_hi], 1 - exp(-L / (beta gamma c tau)).

    This is the rate the positive class has in a real beam, as opposed to the
    enriched rate the MC was generated with. The window should be the one the
    label covers: from where the simulation starts the muon to the downstream
    edge of the target region.
    """
    L = abs(z_hi_mm - z_lo_mm) * 1e-3
    decay_length = (abs(momentum) / MUON_MASS) * MUON_CTAU_M
    return float(-np.expm1(-L / decay_length))


def label_window(df):
    """Z extent (mm) of the positive class in a truth-carrying frame."""
    pos = df[df[LABEL_COLUMN] == 1]
    return float(pos["MuonDecay_Z"].min()), float(pos["MuonDecay_Z"].max())


def load_dataset(path, spec=None, standardizer=None, categorical=True, df=None):
    """CSV path (or an already-loaded frame) -> (X, y, names, meta, spec, standardizer).

    Pass the spec and standardizer from the training set when loading validation or
    new data; leave them None to fit them.
    """
    if df is None:
        df = load_csv(path)
    if spec is None:
        spec = FeatureSpec.fit(df, categorical=categorical)
    X, y, names, meta = build_matrix(df, spec)
    if standardizer is None:
        standardizer = Standardizer().fit(X)
    return standardizer.transform(X), y, names, meta, spec, standardizer


# The boosted-tree settings the shipped model is fitted with. export_onnx.py
# (which ships it) and make_report.py (which derives its operating point) both
# fit it, so both must read the same settings - otherwise the threshold in the
# model file would come from a different model than the one it is attached to.
# tune.py overwrites these by writing model/gbdt_params.json.
GBDT_DEFAULTS = {"max_iter": 400, "learning_rate": 0.1, "early_stopping": True, "validation_fraction": 0.1}
GBDT_PARAMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "gbdt_params.json")


def gbdt_params(path=GBDT_PARAMS_FILE):
    """Boosted-tree settings: the tuned ones if tune.py has written them, else the defaults."""
    params = dict(GBDT_DEFAULTS)
    if os.path.exists(path):
        with open(path) as f:
            params.update(json.load(f)["params"])
    return params


def read_manifest(path):
    """The manifest cook_mc_chain.sh writes next to the CSVs, or None.

    For pooled training data ("a.csv,b.csv") this is the first file's manifest;
    read_manifests() returns all of them."""
    found = read_manifests(str(path).split(",")[0])
    return found[0] if found else None


def read_manifests(path):
    """Every manifest behind a (possibly pooled, comma-joined) CSV path, in order."""
    out = []
    for one in [p for p in str(path).split(",") if p]:
        m = os.path.join(os.path.dirname(one) or ".", "manifest.json")
        if os.path.exists(m):
            with open(m) as f:
                out.append(json.load(f))
    return out
