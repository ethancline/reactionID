"""Tune the boosted-tree settings the shipped model is fitted with.

The search never sees the validation file. It carves a tuning split out of the
training data and picks settings on that; the validation file is used once, at
the end, to compare the result against the current settings. Tuning directly
against the file the numbers are reported on - which is what a quick sweep does -
reports an AUC biased upwards by the search itself.

Several training CSVs can be pooled, e.g. two MC productions of the same beam
setting. The final table separates what the extra data bought from what the
settings bought.

Writes the winning settings to model/gbdt_params.json, which export_onnx.py and
make_report.py read (reactionData.gbdt_params), so the shipped model and the
report both use them.

    python tune.py --train data/train.csv data_15Apr25/train.csv --valid data/valid.csv
"""

import argparse
import json
import time
from datetime import date

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import reactionData

# Search space. Capacity (leaves, rounds) and regularisation (min leaf size, l2)
# pull against each other; the learning rate trades rounds for smoothness.
GRID = {
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "max_iter": [400, 800, 1600],
    "max_leaf_nodes": [31, 63, 127, 255],
    "min_samples_leaf": [20, 50, 100, 200],
    "l2_regularization": [0.0, 0.1, 1.0, 5.0],
    "max_bins": [255],
}


def fit(Xtr, ytr, seed, **params):
    p = dict(reactionData.GBDT_DEFAULTS)
    p.update(params)
    clf = HistGradientBoostingClassifier(random_state=seed, **p)
    clf.fit(Xtr, ytr)
    return clf


def auc(clf, X, y):
    return roc_auc_score(y, clf.predict_proba(X)[:, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", default=["data/train.csv"], help="one or more training CSVs, pooled")
    ap.add_argument("--baseline-train", default="data/train.csv",
                    help="the training set the current model was fitted on, for the before/after table")
    ap.add_argument("--valid", default="data/valid.csv", help="untouched until the final comparison")
    ap.add_argument("--tune-frac", type=float, default=0.15, help="share of the tuning source held out for the search")
    ap.add_argument("--tune-from", type=int, default=None,
                    help="index into --train of the file to carve the tuning split from (default: all of them). "
                         "Use the production the validation file comes from, so the settings are not chosen "
                         "for a different detector simulation")
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=reactionData.GBDT_PARAMS_FILE)
    ap.add_argument("--dry-run", action="store_true", help="search and report, but do not write the settings")
    args = ap.parse_args()

    # Which file each event came from, kept beside the frame rather than in it so
    # it can never be mistaken for an input column.
    frames = [reactionData.load_csv(path) for path in args.train]
    source = np.concatenate([np.full(len(f), i) for i, f in enumerate(frames)])
    pooled = pd.concat(frames, ignore_index=True)
    valid = reactionData.load_csv(args.valid)

    # Which columns carry information is decided on the pooled training data: a
    # column constant in one production may vary in the other.
    cols = reactionData.feature_columns(pooled)
    y = pooled[reactionData.LABEL_COLUMN].to_numpy(np.int32)
    X = pooled[cols].to_numpy(np.float32)
    Xva = valid[cols].to_numpy(np.float32)
    yva = valid[reactionData.LABEL_COLUMN].to_numpy(np.int32)
    print(f"\npooled training: {len(y):,} events from {len(args.train)} file(s), positives {y.mean()*100:.2f}%, {len(cols)} inputs")
    for i, path in enumerate(args.train):
        m = source == i
        print(f"   {path}: {m.sum():,} events, positives {y[m].mean()*100:.2f}%")
    print(f"validation (untouched until the end): {len(yva):,} events")

    # Stratified tuning split, carved from training only - and, with --tune-from,
    # only from the production that the validation file represents.
    rng = np.random.default_rng(args.seed)
    tune = np.zeros(len(y), bool)
    eligible = np.ones(len(y), bool) if args.tune_from is None else (source == args.tune_from)
    for cls in (0, 1):
        idx = np.flatnonzero((y == cls) & eligible)
        tune[rng.choice(idx, int(round(args.tune_frac * len(idx))), replace=False)] = True
    Xs, ys, Xt, yt = X[~tune], y[~tune], X[tune], y[tune]
    where = "all training files" if args.tune_from is None else args.train[args.tune_from]
    print(f"search: fit on {len(ys):,}, score on a held-out tuning split of {len(yt):,} from {where}\n")

    trials = [dict(reactionData.GBDT_DEFAULTS)]  # the current settings are always trial 0
    seen = {json.dumps(trials[0], sort_keys=True)}
    while len(trials) < args.trials:
        kw = {k: rng.choice(v).item() for k, v in GRID.items()}
        key = json.dumps(kw, sort_keys=True)
        if key not in seen:
            seen.add(key)
            trials.append(kw)

    results = []
    t0 = time.perf_counter()
    for i, kw in enumerate(trials):
        t1 = time.perf_counter()
        clf = fit(Xs, ys, args.seed, **kw)
        a = auc(clf, Xt, yt)
        results.append((a, kw, clf.n_iter_))
        tag = " (current settings)" if i == 0 else ""
        print(f"  trial {i:2d}  tuning AUC {a:.5f}  error {1-a:.5f}  rounds {clf.n_iter_:4d}  "
              f"({time.perf_counter()-t1:.0f}s)  {kw}{tag}", flush=True)
    best_auc, best, best_rounds = max(results, key=lambda r: r[0])
    base_auc = results[0][0]
    print(f"\nbest on the tuning split: {best_auc:.5f} vs current settings {base_auc:.5f} "
          f"({(1-base_auc)/(1-best_auc):.2f}x fewer errors), search took {time.perf_counter()-t0:.0f}s")

    # ---- the honest comparison, on the untouched validation file -------------
    print("\n=== validation file, used once ===")
    rows = []
    base = reactionData.load_csv(args.baseline_train)
    Xb, yb = base[cols].to_numpy(np.float32), base[reactionData.LABEL_COLUMN].to_numpy(np.int32)
    # Four fits, so the two gains separate: settings (rows 1->2, 3->4) and data
    # (rows 1->3, 2->4). With productions that differ, more data is not
    # automatically better, and this is where that shows.
    for label, Xf, yf, kw in (("current settings, current training data", Xb, yb, reactionData.GBDT_DEFAULTS),
                              ("tuned settings, current training data", Xb, yb, best),
                              ("current settings, pooled training data", X, y, reactionData.GBDT_DEFAULTS),
                              ("tuned settings, pooled training data", X, y, best)):
        clf = fit(Xf, yf, args.seed, **kw)
        a = auc(clf, Xva, yva)
        rows.append((label, len(yf), a, clf.n_iter_))
        print(f"  {label:<42} {len(yf):>9,} events   AUC {a:.5f}   error {1-a:.5f}   rounds {clf.n_iter_}", flush=True)
    e0 = 1 - rows[0][2]
    for label, n, a, _ in rows[1:]:
        print(f"  -> {label}: {e0/(1-a):.2f}x fewer ranking errors than the current model")

    if args.dry_run:
        print("\n--dry-run: settings not written")
        return
    out = {
        "params": best,
        "tuned": date.today().isoformat(),
        "train": args.train,
        "tuning_split": {"fraction": args.tune_frac, "events": int(len(yt)), "auc": best_auc,
                         "auc_current_settings": base_auc, "trials": len(trials)},
        "validation": [{"model": l, "train_events": n, "auc": a, "rounds": r} for l, n, a, r in rows],
        "n_inputs": len(cols),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out} - export_onnx.py and make_report.py will now fit with these settings")


if __name__ == "__main__":
    main()
