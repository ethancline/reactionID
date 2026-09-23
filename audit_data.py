"""Data-integrity checks behind the review, written to one JSON.

    python audit_data.py --train data/train.csv data_15Apr25/train.csv --valid data/valid.csv --out audit.json

1. Event overlap: does any validation event also appear in a training file? Rows
   are compared on the model's own input columns, rounded to 1e-6, so a repeated
   simulation seed would show up even if the bookkeeping columns differ.
2. Leakage screen: the ROC AUC of each input on its own. A single column that
   nearly separates the classes by itself would be a truth leak, not physics.
3. MC weights: MuonDecay_W must be uniform, or every unweighted rate is wrong.
4. The production chain's cut-based flags: how often the in-time TOF cut and the
   decay verdict are actually true where defined.
5. Stale rows in a cooked ReactionID branch (entry != tree index), i.e. events a
   stage upstream skipped, and so absent from the feature CSV.
"""

import argparse
import json

import numpy as np
import onnx
import pandas as pd
from sklearn.metrics import roc_auc_score

import reactionData


def row_hashes(df, cols):
    return set(pd.util.hash_pandas_object(df[cols].round(6), index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", nargs="+", default=["data/train.csv", "data_15Apr25/train.csv"])
    p.add_argument("--valid", default="data/valid.csv")
    p.add_argument("--model", default="model/decay_gbdt.onnx", help="its input_columns define the rows compared")
    p.add_argument("--rid", default="cooked/mc17606_210MeV_LH2_2_RID.root")
    p.add_argument("--out", default="audit.json")
    args = p.parse_args()

    cols = {e.key: e.value for e in onnx.load(args.model).metadata_props}["input_columns"].split(",")
    valid = pd.read_csv(args.valid)
    hv = row_hashes(valid, cols)
    out = {"input_columns": len(cols), "valid": args.valid, "valid_rows": len(valid), "valid_unique": len(hv), "overlap": []}

    for path in args.train:
        t = pd.read_csv(path, usecols=cols + ["MuonDecay_W"])
        ht = row_hashes(t, cols)
        w = t["MuonDecay_W"].dropna()
        out["overlap"].append({"train": path, "rows": len(t), "unique": len(ht), "shared_with_valid": len(hv & ht),
                               "weight_min": float(w.min()), "weight_max": float(w.max())})
        print(f"{path}: {len(t):,} rows, {len(hv & ht)} shared with {args.valid}, MuonDecay_W in [{w.min():g}, {w.max():g}]")

    # Single-column separation on the events the model is trained and scored on.
    v = reactionData.load_csv(args.valid)
    y = v[reactionData.LABEL_COLUMN].to_numpy()
    aucs = []
    for c in cols:
        x = v[c].to_numpy(dtype=np.float64)
        x = np.where(np.isnan(x), np.nanmin(x) - 1 if np.isfinite(x).any() else 0, x)  # NaN as its own lowest value
        a = roc_auc_score(y, x) if len(np.unique(x)) > 1 else 0.5
        aucs.append((max(a, 1 - a), c))
    aucs.sort(reverse=True)
    out["single_feature_auc"] = [{"name": c, "auc": float(a)} for a, c in aucs[:10]]
    print(f"strongest single input: {aucs[0][1]} AUC {aucs[0][0]:.4f}")

    # The production chain's own cut-based flags (ReactionID's allScattering, as
    # exported). On MC they are only meaningful if the chain's TOF alignment is.
    flags = {}
    for c in ("rid_is_intime_tof", "rid_is_decay"):
        x = valid[c].dropna()
        flags[c] = {"defined": int(len(x)), "true": int((x > 0.5).sum())}
    out["rid_flags"] = flags
    print(f"rid flags: {flags}")

    try:
        import uproot

        with uproot.open(args.rid) as f:
            e = np.asarray(f["PathLength"]["ReactionID"].array()["entry"])
        out["stale"] = {"rid": args.rid, "tree_entries": int(len(e)), "stale": int((e != np.arange(len(e))).sum())}
        print(f"{args.rid}: {out['stale']['stale']} of {len(e)} rows stale")
    except Exception as exc:  # the RID file is optional
        print(f"stale-row check skipped: {exc}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
