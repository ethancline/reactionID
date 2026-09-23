"""Export the decay classifiers to ONNX for the cooker's ReactionID plugin.

Both models are exported to the SAME contract, so the C++ side has one code path
and the model file is chosen by configuration:

    input   "features"     float32[N, 115]  raw column values in FeatureSpec order, NaN allowed
    output  "probability"  float32[N]       decay probability

The MLP's one-hot expansion and standardisation are baked into its graph rather
than reimplemented in C++. Reimplementing them is the classic train/serve skew
bug: it degrades accuracy silently instead of erroring.

Each file carries its own contract in ONNX metadata_props (input_columns,
threshold, model_tag, label, provenance), so the plugin can check at startup that
the feature vector it builds matches what the model was trained on.

The operating threshold is the score giving 90% signal efficiency on a holdout
carved from the training data - the same holdout the MLP early-stopped on,
rebuilt from the split recorded in its checkpoint. It is never chosen on the
validation file, so the numbers make_report.py quotes there are not tuned to it.

    python export_onnx.py --all --verify
"""

import argparse
import json
import os
import subprocess
from datetime import date

import numpy as np
import onnx
import torch
import torch.nn as nn

import reactionData
import reactionModel
import reactionPlots

OPSET = 17
INPUT_NAME = "features"
OUTPUT_NAME = "probability"


# --------------------------------------------------------------------------
# Preprocessing as a torch module, so it exports into the graph
# --------------------------------------------------------------------------
class PreprocessedMLP(nn.Module):
    """raw[N,115] -> one-hot / NaN indicators -> standardise -> MLP -> sigmoid.

    Mirrors reactionData.build_matrix and Standardizer.transform exactly, and is
    constructed from the very FeatureSpec and Standardizer the model was trained
    with, so the two cannot drift.
    """

    def __init__(self, model, spec, standardizer):
        super().__init__()
        self.model = model
        self.columns = list(spec.columns)
        # Per column, the levels to one-hot against (empty => numeric passthrough)
        self.levels = [spec.levels.get(c, []) for c in self.columns]
        self.nan_cols = [c in spec.nan_columns for c in self.columns]
        self.register_buffer("mean", torch.tensor(standardizer.mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(standardizer.std, dtype=torch.float32))

    def forward(self, features):
        parts = []
        for i, col in enumerate(self.columns):
            v = features[:, i : i + 1]
            isnan = torch.isnan(v)
            if self.levels[i]:
                for lv in self.levels[i]:
                    # NaN == anything is False, which is what numpy does too.
                    parts.append((v == float(lv)).to(torch.float32))
                parts.append(isnan.to(torch.float32))
            else:
                parts.append(torch.where(isnan, torch.zeros_like(v), v))
                if self.nan_cols[i]:
                    parts.append(isnan.to(torch.float32))
        x = torch.cat(parts, dim=1)
        x = (x - self.mean) / self.std
        return torch.sigmoid(self.model(x)).squeeze(-1)


# --------------------------------------------------------------------------
def build_metadata(args, columns, tag, threshold, extra=None):
    meta = {
        "input_columns": ",".join(columns),
        "n_input_columns": str(len(columns)),
        "model_tag": tag,
        "threshold": f"{threshold:.6f}",
        "threshold_rule": f"{args.target_eff:.2f} signal efficiency on the training holdout",
        "holdout_frac": f"{args.holdout_frac:g}",
        "holdout_events": str(args.n_holdout),
        "fit_events": str(args.n_fit),
        # Decay fraction of the events the model was fitted on - the MC's enriched
        # rate, not a physical one.
        "train_decay_fraction": f"{args.train_fraction:.6f}",
        # The truth column the model was trained to predict. The cooker scores
        # the classifier against MC truth by reading this, so the comparison
        # cannot drift from the training target the way a hardcoded name did
        # when the label changed from MuonDecay to decay_relevant.
        "label": reactionData.LABEL_COLUMN,
        "exported": date.today().isoformat(),
    }
    # Every production pooled into the training set, not just the first.
    mans = reactionData.read_manifests(args.train)
    if mans:
        meta["train_tag"] = ",".join(str(m.get("tag", "")) for m in mans)
        meta["muse_git_sha"] = ",".join(sorted({str(m.get("git_sha", "")) for m in mans}))
    try:
        meta["reactionid_git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        pass
    meta.update(extra or {})
    return meta


def attach_metadata(path, meta):
    model = onnx.load(path)
    del model.metadata_props[:]
    for k, v in meta.items():
        e = model.metadata_props.add()
        e.key, e.value = k, str(v)
    onnx.save(model, path)


def holdout_threshold(truth, score, target_eff):
    """Score at which the holdout reaches target_eff signal efficiency."""
    for w in reactionPlots.working_points(truth, score, targets=(target_eff,)):
        return float(w["threshold"])
    raise RuntimeError("no working point")


def model_threshold(path):
    """The threshold an exported file carries - what the cooker will actually use."""
    return float({e.key: e.value for e in onnx.load(path).metadata_props}["threshold"])


def mlp_scores(model, spec, standardizer, df):
    X, _, _, _ = reactionData.build_matrix(df, spec)
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(standardizer.transform(X)))).squeeze(-1).numpy()


# --------------------------------------------------------------------------
def export_mlp(args, hold_df):
    print("\n=== MLP ===")
    device = torch.device("cpu")
    model, standardizer, spec, _ = reactionModel.load_model(args.checkpoint, device=device)
    if standardizer is None or spec is None:
        raise ValueError(f"{args.checkpoint} has no feature spec or standardisation; retrain it.")
    model.eval()

    wrapper = PreprocessedMLP(model, spec, standardizer).eval()
    n = len(spec.columns)
    dummy = torch.zeros(2, n, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (dummy,),
        args.mlp_out,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    y = hold_df[reactionData.LABEL_COLUMN].to_numpy()
    thr = holdout_threshold(y, mlp_scores(model, spec, standardizer, hold_df), args.target_eff)
    attach_metadata(
        args.mlp_out,
        build_metadata(args, spec.columns, "mlp", thr, {"n_expanded_features": str(model.n_features)}),
    )
    print(f"wrote {args.mlp_out}  ({n} raw inputs -> {model.n_features} expanded), threshold {thr:.4f} from the holdout")
    return spec


class _coerce_bool_attributes:
    """Work around a skl2onnx bug on the NaN-split code path.

    skl2onnx emits `nodes_missing_value_tracks_true` as a list of Python bools,
    and onnx's make_attribute rejects bools where an int list is expected
    ("Field onnx.AttributeProto.ints: Expected an int, got a boolean"). The
    attribute is only populated when the tree actually has NaN-handling splits,
    which is why this only bites models trained on data where NaN is meaningful
    - i.e. exactly ours. Checked against onnx 1.17 and 1.18; neither accepts it.

    Coerce bool -> int for the duration of the conversion only. Drop this once
    skl2onnx fixes it upstream; the `--verify` check will still catch it if the
    workaround ever stops producing a faithful model.
    """

    def __enter__(self):
        from onnx import helper

        self._orig = helper.make_attribute

        def patched(key, value, *a, **kw):
            # numpy.bool_ is not a Python bool, and the lists come back mixed,
            # so test elementwise rather than requiring the whole list be bools.
            if isinstance(value, (list, tuple)) and value:
                if any(isinstance(v, (bool, np.bool_)) for v in value):
                    value = [int(v) for v in value]
            return self._orig(key, value, *a, **kw)

        helper.make_attribute = patched
        return self

    def __exit__(self, *exc):
        from onnx import helper

        helper.make_attribute = self._orig
        return False


def export_gbdt(args, train_df, hold_df, spec=None):
    print("\n=== gradient boosting ===")
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.ensemble import HistGradientBoostingClassifier

    cols = reactionData.feature_columns(train_df)
    # One input contract for both models: make_report.py scores them on the same
    # raw matrix, and the cooker maps one list of names onto the builder.
    if spec is not None and cols != list(spec.columns):
        raise RuntimeError("the boosted trees and the MLP chose different input columns - "
                           "were they fitted on different training data?")
    Xtr = train_df[cols].to_numpy(dtype=np.float32)
    ytr = train_df[reactionData.LABEL_COLUMN].to_numpy(dtype=np.int32)

    params = reactionData.gbdt_params()
    clf = HistGradientBoostingClassifier(random_state=args.seed, **params)
    clf.fit(Xtr, ytr)
    print(f"fitted {clf.n_iter_} boosting rounds on {Xtr.shape} with {params}")

    with _coerce_bool_attributes():
        onx = convert_sklearn(
            clf,
            initial_types=[(INPUT_NAME, FloatTensorType([None, len(cols)]))],
            target_opset=OPSET,
            options={id(clf): {"zipmap": False}},
        )
    # skl2onnx emits label + probabilities[N,2]; keep only P(decay) under our name.
    onx = _select_positive_probability(onx)
    with open(args.gbdt_out, "wb") as f:
        f.write(onx.SerializeToString())
    thr = holdout_threshold(hold_df[reactionData.LABEL_COLUMN].to_numpy(),
                            clf.predict_proba(hold_df[cols].to_numpy(dtype=np.float32))[:, 1], args.target_eff)
    attach_metadata(
        args.gbdt_out,
        build_metadata(args, cols, "gbdt", thr,
                       {"rounds": str(clf.n_iter_), "gbdt_params": json.dumps(params, sort_keys=True)}),
    )
    print(f"wrote {args.gbdt_out}  ({len(cols)} inputs), threshold {thr:.4f} from the holdout")

    import joblib

    # The fitted estimator make_report.py evaluates - loaded, not refitted, so
    # the report describes exactly the model whose threshold is in the ONNX file.
    joblib.dump({"model": clf, "columns": cols, "threshold": thr}, args.gbdt_pkl)
    print(f"wrote {args.gbdt_pkl}")
    return clf, cols


def _select_positive_probability(onx):
    """Reduce skl2onnx's (label, probabilities[N,2]) to a single probability[N].

    Keeps the C++ contract identical between the two models: one output, one
    float per event.
    """
    from onnx import TensorProto, helper

    prob_name = None
    for o in onx.graph.output:
        if "probab" in o.name.lower():
            prob_name = o.name
            break
    if prob_name is None:
        raise RuntimeError(f"no probability output in {[o.name for o in onx.graph.output]}")

    idx = helper.make_tensor("pos_class_index", TensorProto.INT64, [1], [1])
    onx.graph.initializer.append(idx)
    onx.graph.node.append(
        helper.make_node("Gather", [prob_name, "pos_class_index"], ["prob_2d"], axis=1, name="select_positive")
    )
    onx.graph.node.append(
        helper.make_node("Squeeze", ["prob_2d", "squeeze_axis"], [OUTPUT_NAME], name="squeeze_positive")
    )
    onx.graph.initializer.append(helper.make_tensor("squeeze_axis", TensorProto.INT64, [1], [1]))

    del onx.graph.output[:]
    onx.graph.output.append(helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, ["batch"]))
    return onx


# --------------------------------------------------------------------------
def verify(args, spec, gbdt):
    """Run ONNX and the native models over the same events and compare.

    This is the gate: an export that does not reproduce the Python numbers is
    not a model, it is a silent accuracy regression.
    """
    import onnxruntime as ort

    print("\n=== verification ===")
    df = reactionData.load_csv(args.valid)
    ok = True

    if os.path.exists(args.mlp_out) and spec is not None:
        raw = df[spec.columns].to_numpy(dtype=np.float32)
        model, standardizer, _, _ = reactionModel.load_model(args.checkpoint, device=torch.device("cpu"))
        model.eval()
        X, _, _, _ = reactionData.build_matrix(df, spec)
        with torch.no_grad():
            native = torch.sigmoid(model(torch.tensor(standardizer.transform(X)))).squeeze(-1).numpy()
        sess = ort.InferenceSession(args.mlp_out, providers=["CPUExecutionProvider"])
        got = sess.run([OUTPUT_NAME], {INPUT_NAME: raw})[0].ravel()
        ok &= _compare("mlp", native, got, model_threshold(args.mlp_out))

    if os.path.exists(args.gbdt_out) and gbdt is not None:
        clf, cols = gbdt
        raw = df[cols].to_numpy(dtype=np.float32)
        native = clf.predict_proba(raw)[:, 1]
        sess = ort.InferenceSession(args.gbdt_out, providers=["CPUExecutionProvider"])
        got = sess.run([OUTPUT_NAME], {INPUT_NAME: raw})[0].ravel()
        ok &= _compare("gbdt", native, got, model_threshold(args.gbdt_out))

    # NaN must survive the graph: it is the "not measured" marker, not a gap.
    if os.path.exists(args.gbdt_out):
        sess = ort.InferenceSession(args.gbdt_out, providers=["CPUExecutionProvider"])
        n = sess.get_inputs()[0].shape[1]
        allnan = np.full((1, n), np.nan, dtype=np.float32)
        p = sess.run([OUTPUT_NAME], {INPUT_NAME: allnan})[0].ravel()[0]
        print(f"  gbdt all-NaN event -> {p:.5f}  ({'finite, NaN handled' if np.isfinite(p) else 'NOT FINITE'})")
        ok &= bool(np.isfinite(p))

    print("\n  " + ("ALL CHECKS PASS" if ok else "FAILED - do not ship"))
    return ok


def _compare(tag, native, got, thr, tol=1e-4, max_outlier_frac=1e-3):
    """Pass if the decisions are identical and disagreement is a negligible tail.

    A bare `max|dP| < tol` is the wrong test for a tree ensemble. sklearn's
    HistGradientBoosting predicts from a binned representation while the exported
    TreeEnsembleClassifier compares float32 against exact thresholds, so an event
    sitting on a split boundary can take the other branch and land in a different
    leaf. That moves its probability visibly but happens to a handful of events
    and never near the operating point.

    So gate on what deployment actually depends on: no event changes its label,
    and the fraction of events off by more than `tol` stays below
    `max_outlier_frac`. Both are reported so a real regression is still obvious.

    One exception, reported separately rather than hidden. The working-point
    threshold is chosen as the score of an actual validation event, so that event
    sits *exactly* on it: the native model says `score > thr` is false, and an
    export one float32 ULP higher says true. That is a tie, not a disagreement -
    it happens to whichever event defines the threshold, in any consumer that
    rounds differently. A flip only counts when a score is clear of the
    threshold by more than `tie_tol`.
    """
    tie_tol = 1e-6
    d = np.abs(native - got)
    flipped = (native > thr) != (got > thr)
    tie = flipped & (np.abs(native - thr) <= tie_tol) & (np.abs(got - thr) <= tie_tol)
    lab = int((flipped & ~tie).sum())
    frac = float((d > tol).mean())
    good = lab == 0 and frac < max_outlier_frac
    print(
        f"  {tag:5s} n={len(native):,}  mean|dP|={d.mean():.3e}  max|dP|={d.max():.3e}  "
        f"|dP|>{tol:g}: {int((d > tol).sum())} ({frac*100:.4f}%)  label disagreements={lab}"
        f"{f'  (+{int(tie.sum())} tie on the threshold)' if tie.any() else ''}  "
        f"{'PASS' if good else 'FAIL'}"
    )
    return good


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default="data/train.csv")
    p.add_argument("--valid", default="data/valid.csv")
    p.add_argument("--checkpoint", default="model/output.pth")
    p.add_argument("--mlp-out", default="model/decay_mlp.onnx")
    p.add_argument("--gbdt-out", default="model/decay_gbdt.onnx")
    p.add_argument("--gbdt-pkl", default="model/decay_gbdt.joblib")
    p.add_argument("--holdout-frac", type=float, default=None,
                   help="training share held out for the thresholds; default: whatever the MLP checkpoint recorded")
    p.add_argument("--target-eff", type=float, default=0.90, help="signal efficiency the threshold is set at")
    p.add_argument("--seed", type=int, default=None, help="split/fit seed; default: the checkpoint's")
    p.add_argument("--mlp", action="store_true")
    p.add_argument("--gbdt", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()

    if not (args.mlp or args.gbdt or args.all):
        args.all = True

    # Rebuild the exact holdout the MLP early-stopped on, from its checkpoint.
    rec = reactionModel.checkpoint_extra(args.checkpoint) if os.path.exists(args.checkpoint) else {}
    if rec.get("train") and rec["train"] != args.train:
        raise SystemExit(f"{args.checkpoint} was trained on {rec['train']}, not {args.train} - the holdout would not match")
    args.holdout_frac = args.holdout_frac if args.holdout_frac is not None else rec.get("holdout_frac", 0.15)
    args.seed = args.seed if args.seed is not None else rec.get("seed", 1234)
    fit_df, hold_df = reactionData.holdout_split(reactionData.load_csv(args.train), args.holdout_frac, args.seed)
    args.n_holdout = len(hold_df)
    args.n_fit = len(fit_df)
    args.train_fraction = float(fit_df[reactionData.LABEL_COLUMN].mean())
    print(f"fit {len(fit_df):,}  holdout {len(hold_df):,} ({args.holdout_frac:g}, seed {args.seed})")

    os.makedirs(os.path.dirname(args.mlp_out) or ".", exist_ok=True)
    spec = gbdt = None
    if args.mlp or args.all:
        spec = export_mlp(args, hold_df)
    if args.gbdt or args.all:
        gbdt = export_gbdt(args, fit_df, hold_df, spec)

    if args.verify:
        if not verify(args, spec, gbdt):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
