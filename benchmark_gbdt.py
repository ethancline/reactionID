"""Gradient-boosted-trees baseline, and which features actually carry the signal.

A 366-input MLP is a lot of machinery for tabular data. Boosted trees train in
seconds on the same CSVs, handle the exporter's NaN 'not measured' marker natively,
and report permutation importances - so this doubles as the answer to "which
detectors is the classifier actually using?".

    python benchmark_gbdt.py --train data/train.csv --valid data/valid.csv
"""

import argparse
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import auc, roc_curve

import reactionData


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default="data/train.csv")
    p.add_argument("--valid", default="data/valid.csv")
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--top", type=int, default=25, help="how many features to list")
    p.add_argument("--importance-samples", type=int, default=20000, help="subsample size for permutation importance")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    # Trees split on raw values and treat NaN as its own branch, so no one-hot
    # expansion and no standardisation here - just the columns as exported.
    train_df = reactionData.load_csv(args.train)
    valid_df = reactionData.load_csv(args.valid)
    cols = reactionData.feature_columns(train_df)

    Xtr = train_df[cols].to_numpy(dtype=np.float32)
    ytr = train_df[reactionData.LABEL_COLUMN].to_numpy(dtype=np.int32)
    Xva = valid_df[cols].to_numpy(dtype=np.float32)
    yva = valid_df[reactionData.LABEL_COLUMN].to_numpy(dtype=np.int32)
    print(f"train {Xtr.shape}  valid {Xva.shape}  decay fraction {ytr.mean():.4f}")

    clf = HistGradientBoostingClassifier(
        max_iter=args.max_iter,
        learning_rate=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=args.seed,
    )
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    print(f"Trained {clf.n_iter_} boosting rounds in {time.perf_counter()-t0:.1f}s")

    score = clf.predict_proba(Xva)[:, 1]
    pred = (score > 0.5).astype(np.int32)
    fpr, tpr, _ = roc_curve(yva, score)
    print("\n=== Gradient boosting on the validation set ===")
    print(f"  accuracy : {(pred == yva).mean()*100:.3f}%")
    print(f"  ROC AUC  : {auc(fpr, tpr):.4f}")
    print("  (compare against the MLP numbers printed by `reactionID.py eval`)")

    n = min(args.importance_samples, len(Xva))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(Xva), size=n, replace=False)
    print(f"\nComputing permutation importance on {n} events...")
    imp = permutation_importance(clf, Xva[idx], yva[idx], n_repeats=3, random_state=args.seed, scoring="roc_auc")

    order = np.argsort(imp.importances_mean)[::-1][: args.top]
    print(f"\n=== Top {args.top} features by drop in ROC AUC when shuffled ===")
    for i in order:
        print(f"  {cols[i]:<26} {imp.importances_mean[i]:+.5f} +- {imp.importances_std[i]:.5f}")


if __name__ == "__main__":
    main()
