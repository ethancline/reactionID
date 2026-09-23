"""Collect the numbers behind the results deck into one JSON.

Evaluates both shipped models - the MLP checkpoint and the boosted trees that
export_onnx.py fitted and saved - on the validation CSV, at the thresholds their
ONNX files carry, broken down by where the muon actually decayed, and dumps
permutation feature importances. Nothing here is fitted or tuned on this file.

    python make_report.py --out report.json
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import auc, roc_auc_score, roc_curve

import reactionData
import reactionModel
import reactionPlots

REGIONS = {0: "no decay", 1: "upstream", 2: "target", 3: "downstream"}



def permutation_importance_onnx(session, X, y, columns, n_repeats=3, seed=1234):
    """Drop in ROC AUC when each raw input column is shuffled.

    Run against the exported ONNX rather than the in-memory estimator, so the
    ranking describes the artifact that is actually deployed in the cooker - and
    so the same protocol applies to both models, making them comparable. The MLP's
    graph carries its own preprocessing, so both take the identical 115 raw columns.
    """
    rng = np.random.default_rng(seed)

    def score(Xa):
        return session.run(["probability"], {"features": Xa})[0].ravel()

    fpr, tpr, _ = roc_curve(y, score(X))
    base = auc(fpr, tpr)

    means, stds = np.zeros(len(columns)), np.zeros(len(columns))
    for j in range(len(columns)):
        drops = []
        col = X[:, j].copy()
        for _ in range(n_repeats):
            X[:, j] = rng.permutation(col)
            f, t, _ = roc_curve(y, score(X))
            drops.append(base - auc(f, t))
        X[:, j] = col
        means[j], stds[j] = np.mean(drops), np.std(drops)
    return base, means, stds


def importance_block(session, X, y, cols, n, seed, tag):
    base, means, stds = permutation_importance_onnx(session, X, y, cols, seed=seed)
    order = np.argsort(means)[::-1]
    print(f"  {tag}: baseline AUC {base:.4f}, top feature {cols[order[0]]} ({means[order[0]]:+.5f})")
    return {
        "n_events": int(n),
        "baseline_auc": float(base),
        "metric": "drop in ROC AUC when the column is shuffled",
        "features": [{"name": cols[i], "mean": float(means[i]), "std": float(stds[i])} for i in order],
    }



# --------------------------------------------------------------------------
# Feature-importance artifacts
# --------------------------------------------------------------------------
from make_slides import C_GBDT, C_MLP  # one colour per model, everywhere


def write_importance_figure(report, path, top=20):
    """Ranked bar chart of the top features, both models side by side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gb = report.get("importance")
    if not gb:
        return None
    mlp = {f["name"]: f for f in report.get("importance_mlp", {}).get("features", [])}

    feats = gb["features"][:top]
    names = [f["name"] for f in feats]
    y = np.arange(len(feats))[::-1]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(feats) + 1.8), constrained_layout=True)
    h = 0.38
    ax.barh(y + h / 2, [f["mean"] for f in feats], height=h, xerr=[f["std"] for f in feats],
            color=C_GBDT, error_kw=dict(ecolor="#6b7280", lw=1), label="boosted trees")
    if mlp:
        ax.barh(y - h / 2, [mlp.get(n, {}).get("mean", 0.0) for n in names], height=h,
                xerr=[mlp.get(n, {}).get("std", 0.0) for n in names],
                color=C_MLP, error_kw=dict(ecolor="#6b7280", lw=1), label="MLP")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9, family="monospace")
    ax.set_xlabel("drop in ROC AUC when the column is shuffled")
    ax.set_title(f"Feature importance, top {len(feats)} of {len(gb['features'])}"
                 f"   ({gb['n_events']:,} validation events, 3 repeats)", loc="left", fontweight="bold")
    ax.axvline(0, color="#6b7280", lw=0.8)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path


def write_importance_tables(report, md_path, csv_path):
    """Every feature ranked - the zero-importance tail is the point."""
    gb = report.get("importance")
    if not gb:
        return None
    mlp = {f["name"]: f for f in report.get("importance_mlp", {}).get("features", [])}

    rows = [(i + 1, f["name"], f["mean"], f["std"], mlp.get(f["name"], {}).get("mean", float("nan")))
            for i, f in enumerate(gb["features"])]

    with open(csv_path, "w") as f:
        f.write("rank,feature,gbdt_mean,gbdt_std,mlp_mean\n")
        for r, nm, m, sd, mm in rows:
            f.write(f"{r},{nm},{m:.6f},{sd:.6f},{mm:.6f}\n")

    n_zero = sum(1 for _, _, m, _, _ in rows if m <= 0)
    with open(md_path, "w") as f:
        f.write("# Feature importance\n\n")
        f.write(f"Drop in ROC AUC when a column is shuffled, {gb['n_events']:,} validation events, "
                "3 repeats, measured against the exported ONNX models.\n\n")
        f.write(f"Baseline AUC: boosted trees {gb['baseline_auc']:.4f}")
        if report.get("importance_mlp"):
            f.write(f", MLP {report['importance_mlp']['baseline_auc']:.4f}")
        f.write(f".  {n_zero} of {len(rows)} columns contribute nothing to the boosted trees.\n\n")
        f.write("| rank | feature | GBDT | +- | MLP |\n|---:|---|---:|---:|---:|\n")
        for r, nm, m, sd, mm in rows:
            f.write(f"| {r} | `{nm}` | {m:+.5f} | {sd:.5f} | {mm:+.5f} |\n")
    return md_path


def region_breakdown(truth, score, region, threshold):
    """Per-region rates at the shipped operating threshold, and per-region AUC.

    For the two labelled-decay regions the rate is an efficiency. For "no decay"
    and "downstream" (both label 0) it is the rate at which the model flags them,
    i.e. a false-positive rate - downstream decays are not the target class.

    For a decay region the AUC is that region's decays against the true non-decays
    (region 0), which is the question actually being asked: can this kind of decay
    be told apart from a genuine scatter?
    """
    pred = (score > threshold).astype(np.float32)
    nondecay = region == 0
    out = []
    for code, name in REGIONS.items():
        m = region == code
        n = int(m.sum())
        if n == 0:
            continue
        labelled = bool(truth[m].max() > 0.5)
        row = {"code": code, "name": name, "n": n, "label": int(labelled),
               "accuracy": float((pred[m] == truth[m]).mean()), "flagged": float((pred[m] == 1).mean()),
               "metric": "efficiency" if labelled else "false-positive rate", "auc": None}
        row["value"] = row["flagged"]
        if labelled:
            sel = m | nondecay
            fpr, tpr, _ = roc_curve(truth[sel], score[sel])
            row["auc"] = float(auc(fpr, tpr))
        out.append(row)
    return out


def confusion(truth, score, region, threshold):
    """TP/TN/FP/FN at the threshold, as the cooker's truth comparison counts them."""
    pred = (score > threshold).astype(int)
    yi = truth.astype(int)
    tp = int(((pred == 1) & (yi == 1)).sum()); tn = int(((pred == 0) & (yi == 0)).sum())
    fp = int(((pred == 1) & (yi == 0)).sum()); fn = int(((pred == 0) & (yi == 1)).sum())
    return {
        "threshold": float(threshold),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "efficiency": tp / (tp + fn) if tp + fn else 0.0,
        "purity": tp / (tp + fp) if tp + fp else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "regions": [
            {"name": nm, "n": int((region == c).sum()),
             "correct": float((pred[region == c] == yi[region == c]).mean())}
            for c, nm in REGIONS.items() if (region == c).sum()
        ],
    }


def model_block(truth, score, region, threshold, priors, extra):
    fpr, tpr, _ = roc_curve(truth, score)
    nondecay = region == 0
    wp = reactionPlots.working_points(truth, score, prior=priors["physical"], negatives=nondecay)
    # The same working points at the prior the previous deck used, for comparison.
    for row, old in zip(wp, reactionPlots.working_points(truth, score, prior=priors["previous"], negatives=nondecay)):
        row["precision_at_previous_prior"] = old["precision_at_prior"]
    at_thr = (score > threshold)
    eff = float(at_thr[truth == 1].mean())
    fpr0 = float(at_thr[nondecay].mean())
    p = priors["physical"]
    out = {
        "threshold": float(threshold),
        "accuracy": float((at_thr == truth).mean()),
        "auc": float(auc(fpr, tpr)),
        "auc_vs_nondecay": float(roc_auc_score(truth[(truth == 1) | nondecay], score[(truth == 1) | nondecay])),
        "at_threshold": {"efficiency": eff, "fpr_nondecay": fpr0,
                         "precision_at_prior": p * eff / (p * eff + (1 - p) * fpr0) if (eff + fpr0) > 0 else float("nan")},
        "regions": region_breakdown(truth, score, region, threshold),
        "roc": roc_points(truth, score),
        # Against the true non-decays only - the background real data is made of.
        "roc_nondecay": roc_points(truth[(truth == 1) | nondecay], score[(truth == 1) | nondecay], n=400),
        "working_points": wp,
        "truth": confusion(truth, score, region, threshold),
    }
    out.update(extra)
    return out


def roc_points(truth, score, n=220):
    """Thinned ROC curve, small enough to inline in an SVG."""
    fpr, tpr, _ = roc_curve(truth, score)
    idx = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
    return [[round(float(fpr[i]), 5), round(float(tpr[i]), 5)] for i in idx]


def onnx_meta(path):
    import onnx as _onnx

    return {e.key: e.value for e in _onnx.load(path).metadata_props}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--valid", default="data/valid.csv")
    p.add_argument("--checkpoint", default="model/output.pth")
    p.add_argument("--gbdt-pkl", default="model/decay_gbdt.joblib")
    p.add_argument("--out", default="report.json")
    p.add_argument("--prior", type=float, default=None,
                   help="physical decay prior for precision; default: derived from the beam momentum and the "
                        "label's Z window (reactionData.physical_prior)")
    p.add_argument("--previous-prior", type=float, default=0.003, help="the prior earlier reports quoted, for comparison")
    p.add_argument("--gbdt-onnx", default="model/decay_gbdt.onnx")
    p.add_argument("--mlp-onnx", default="model/decay_mlp.onnx")
    p.add_argument("--plots", action="store_true", help="also write feature_importance.pdf and the ranked tables")
    p.add_argument("--reuse-importance", action="store_true",
                   help="keep the importance blocks from an existing --out file instead of recomputing them")
    p.add_argument("--top-figure", type=int, default=20, help="features shown in the importance figure")
    p.add_argument("--importance-samples", type=int, default=25000)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    # Both shipped models, exactly as shipped: the MLP checkpoint the ONNX was
    # exported from and the fitted trees export_onnx.py saved. Nothing is refitted
    # here, and each threshold is the one its ONNX file carries into the cooker -
    # chosen on the training holdout, never on this file.
    meta_g, meta_m = onnx_meta(args.gbdt_onnx), onnx_meta(args.mlp_onnx)
    thr_g, thr_m = float(meta_g["threshold"]), float(meta_m["threshold"])
    rec = reactionModel.checkpoint_extra(args.checkpoint)
    train_path = rec.get("train", "")

    report = {"valid": args.valid, "train": train_path,
              "manifests": reactionData.read_manifests(train_path) if train_path else [],
              "valid_manifest": reactionData.read_manifest(args.valid)}

    raw_valid = reactionData.load_csv(args.valid, drop_chv_veto=False)
    valid_df = reactionData.load_csv(args.valid)

    # The physical prior: the chance a beam muon decays over the window the label
    # covers, at this beam momentum.
    z_lo, z_hi = reactionData.label_window(valid_df)
    momentum = float(report["valid_manifest"]["momentum_MeV"]) if report["valid_manifest"] else float(valid_df["momentum"].dropna().iloc[0])
    prior = args.prior if args.prior is not None else reactionData.physical_prior(momentum, z_lo, z_hi)
    report["prior"] = {"physical": prior, "previous": args.previous_prior, "momentum_MeV": momentum,
                       "window_mm": [z_lo, z_hi],
                       "decay_length_m": momentum / reactionData.MUON_MASS * reactionData.MUON_CTAU_M}
    priors = report["prior"]

    # ---- MLP ---------------------------------------------------------------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, standardizer, spec, _ = reactionModel.load_model(args.checkpoint, device=device)
    rawX, y, _, meta = reactionData.build_matrix(valid_df, spec)
    region = meta["decay_region"].to_numpy()
    logits, truth, loss = reactionPlots.predict(model, standardizer.transform(rawX), y, device)
    mlp_score = 1.0 / (1.0 + np.exp(-logits))
    ck = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    report["mlp"] = model_block(truth, mlp_score, region, thr_m, priors, {
        "n_features": int(model.n_features), "loss": float(loss), "best_epoch": int(ck["epoch"]) + 1,
        "holdout_loss": float(ck["loss"]),
    })

    # ---- gradient boosting -------------------------------------------------
    import joblib

    shipped = joblib.load(args.gbdt_pkl)
    clf, cols = shipped["model"], shipped["columns"]
    if cols != meta_g["input_columns"].split(","):
        raise SystemExit(f"{args.gbdt_pkl} and {args.gbdt_onnx} disagree on their inputs - re-run export_onnx.py")
    Xva = valid_df[cols].to_numpy(dtype=np.float32)
    gb_score = clf.predict_proba(Xva)[:, 1]
    report["gbdt"] = model_block(truth, gb_score, region, thr_g, priors, {
        "n_features": len(cols), "rounds": int(clf.n_iter_),
        "params": json.loads(meta_g.get("gbdt_params", "{}")),
    })
    report["truth"] = report["gbdt"]["truth"]  # the model the cooker runs by default

    report["split"] = {k: meta_g.get(k) for k in ("threshold_rule", "holdout_frac", "holdout_events", "fit_events",
                                                   "train_decay_fraction", "train_tag", "muse_git_sha", "exported")}

    # ---- feature importance, for both models --------------------------------
    # The full ranking is kept (not just the top N): the columns contributing
    # nothing are the interesting ones when deciding what to drop from the
    # input contract.
    import onnxruntime as ort

    n = min(args.importance_samples, len(Xva))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(Xva), size=n, replace=False)
    Xs, ys = Xva[idx].copy(), truth[idx].astype(int)
    reuse = args.reuse_importance and os.path.exists(args.out)
    if reuse:
        with open(args.out) as f:
            prev = json.load(f)
        for key in ("importance", "importance_mlp"):
            if key in prev:
                report[key] = prev[key]
        print("\nReusing the existing permutation importance.")
    else:
        print("\nPermutation importance...")
        for key, path, tag in (("importance", args.gbdt_onnx, "gbdt"), ("importance_mlp", args.mlp_onnx, "mlp")):
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            report[key] = importance_block(sess, Xs.copy(), ys, cols, n, args.seed, tag)

    # ---- dataset facts ------------------------------------------------------
    rr = raw_valid["decay_region"].to_numpy()
    report["dataset"] = {
        "valid_events_raw": int(len(raw_valid)),
        "valid_events": int(len(truth)),
        "train_events": int(meta_g.get("fit_events", 0)) + int(meta_g.get("holdout_events", 0)),
        "decay_fraction": float(truth.mean()),
        "raw_label_fraction": float(raw_valid[reactionData.LABEL_COLUMN].mean()),
        "raw_muondecay_fraction": float(raw_valid["MuonDecay"].mean()),
        "chv_rate": float(raw_valid["chv_veto"].mean()),
        "chv_rate_by_region": {REGIONS[c]: float(raw_valid["chv_veto"][rr == c].mean()) for c in REGIONS},
        "columns": int(len(valid_df.columns)),
        "feature_columns": len(cols),
        "region_counts": {REGIONS[c]: int((region == c).sum()) for c in REGIONS},
        "reconstructed": {"stt": float(valid_df["stt_valid"].mean()), "gem": float(valid_df["gem_valid"].mean()),
                          "vertex": float((valid_df["n_vertices"] > 0).mean())},
    }

    # roc_curve returns float32 thresholds, which json cannot encode.
    def plain(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if np.isinf(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"cannot serialise {type(o)}")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=plain)

    if args.plots:
        fig = write_importance_figure(report, "feature_importance.pdf", top=args.top_figure)
        tab = write_importance_tables(report, "feature_importance.md", "feature_importance.csv")
        if fig:
            print(f"wrote {fig}")
        if tab:
            print(f"wrote {tab} and feature_importance.csv")
    print(f"\nwrote {args.out}   physical prior {prior:.5f} (window {z_lo:.0f}..{z_hi:.0f} mm)")
    for k in ("mlp", "gbdt"):
        r = report[k]
        print(f"  {k:5s} AUC {r['auc']:.5f}  threshold {r['threshold']:.4f}: efficiency {r['at_threshold']['efficiency']*100:.2f}%"
              f"  FPR(no decay) {r['at_threshold']['fpr_nondecay']*100:.3f}%  precision@prior {r['at_threshold']['precision_at_prior']*100:.1f}%")


if __name__ == "__main__":
    main()
