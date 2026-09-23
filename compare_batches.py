"""Are two MC productions of the same beam setting interchangeable for training?

Three checks, before pooling them:

1. Label and decay-region rates, after the same cuts training applies.
2. Adversarial validation: train a classifier to tell which production an event
   came from, using the model's own inputs. AUC ~0.5 means the productions look
   the same to the model; well above that means something differs, and the
   columns that separate them say what. Run *within* each truth class, so a
   difference in how often muons decay cannot pose as a detector difference.
3. Transfer: train on each production, size-matched, and score both on the same
   untouched validation file. If the second production is as good a teacher as
   the first, pooling them is safe.

    python compare_batches.py --a data/train.csv --b data_15Apr25/train.csv --valid data/valid.csv
"""

import argparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import reactionData

REGIONS = {0: "no decay", 1: "upstream", 2: "target", 3: "downstream"}


def hgb(seed):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True,
                                          validation_fraction=0.1, random_state=seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="the production the current model was trained on")
    ap.add_argument("--b", required=True, help="the candidate to pool with it")
    ap.add_argument("--valid", required=True, help="untouched validation file, from production A")
    ap.add_argument("--per-class", type=int, default=40000, help="events per production per truth class, adversarial test")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    A, B, V = reactionData.load_csv(args.a), reactionData.load_csv(args.b), reactionData.load_csv(args.valid)
    cols = reactionData.feature_columns(A, verbose=False)
    lab = reactionData.LABEL_COLUMN

    # ---- 1. rates -------------------------------------------------------------
    print("\n=== 1. rates after the training cuts ===")
    print(f"{'':<14}{'events':>10}{'MuonDecay':>11}{'label':>9}" + "".join(f"{REGIONS[c]:>12}" for c in REGIONS))
    for name, d in (("A", A), ("B", B)):
        n = len(d)
        reg = d["decay_region"].to_numpy()
        print(f"{name:<14}{n:>10,}{d['MuonDecay'].mean()*100:>10.2f}%{d[lab].mean()*100:>8.2f}%"
              + "".join(f"{(reg == c).mean()*100:>11.2f}%" for c in REGIONS))
    for c in (1, 2, 3):
        pa, pb = (A["decay_region"] == c).mean(), (B["decay_region"] == c).mean()
        se = np.sqrt(pa * (1 - pa) / len(A) + pb * (1 - pb) / len(B))
        print(f"   {REGIONS[c]:<11} A-B = {100*(pa-pb):+.2f} points ({(pa-pb)/se:+.1f} sigma)")

    # ---- 2. adversarial validation, within each truth class --------------------
    print("\n=== 2. can a classifier tell the productions apart? (AUC 0.5 = no) ===")
    for c in REGIONS:
        a = A[A["decay_region"] == c]
        b = B[B["decay_region"] == c]
        n = min(args.per_class, len(a), len(b))
        if n < 2000:
            print(f"  {REGIONS[c]:<11} too few events ({n}) - skipped")
            continue
        sa, sb = a.sample(n, random_state=args.seed), b.sample(n, random_state=args.seed)
        X = np.vstack([sa[cols].to_numpy(np.float32), sb[cols].to_numpy(np.float32)])
        y = np.r_[np.zeros(n, int), np.ones(n, int)]
        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]
        half = len(y) // 2
        clf = hgb(args.seed).fit(X[:half], y[:half])
        score = roc_auc_score(y[half:], clf.predict_proba(X[half:])[:, 1])
        line = f"  {REGIONS[c]:<11} {n:>6,} per production   AUC {score:.3f}"
        if score > 0.55:
            # Which inputs give it away: the drop in AUC when each is shuffled.
            from sklearn.inspection import permutation_importance
            imp = permutation_importance(clf, X[half:][:8000], y[half:][:8000], n_repeats=2,
                                         random_state=args.seed, scoring="roc_auc")
            top = np.argsort(imp.importances_mean)[::-1][:5]
            line += "   separated by: " + ", ".join(f"{cols[i]} ({imp.importances_mean[i]:+.3f})" for i in top)
        print(line)

    # ---- 3. transfer ----------------------------------------------------------
    print("\n=== 3. as teachers: size-matched training, same untouched validation ===")
    n = min(len(A), len(B))
    Xv, yv = V[cols].to_numpy(np.float32), V[lab].to_numpy(int)
    for name, d in (("A", A), ("B", B)):
        s = d.sample(n, random_state=args.seed)
        clf = hgb(args.seed).fit(s[cols].to_numpy(np.float32), s[lab].to_numpy(int))
        a = roc_auc_score(yv, clf.predict_proba(Xv)[:, 1])
        print(f"  trained on {name} ({n:,} events)   validation AUC {a:.5f}   error {1-a:.5f}")


if __name__ == "__main__":
    main()
