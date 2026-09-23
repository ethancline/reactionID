"""Compare the cooker's ReactionID scores against Python's, event by event.

This is the check that catches train/serve skew. The feature vector is built by
one shared C++ class, so the two *should* be identical by construction - but
"should be" is exactly the assumption that silently costs accuracy when it stops
holding. A model fed a subtly different vector than it was trained on does not
error; it just gets worse.

    python verify_inference.py --rid <run>_RID.root --csv <run>_features.csv

Requires uproot to read the cooked tree.
"""

import argparse
import os
import sys

import numpy as np
import onnxruntime as ort
import pandas as pd

import reactionData


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rid", required=True, help="the _RID.root file written by the ReactionID stage")
    p.add_argument("--csv", required=True, help="the _features.csv written by muonDecay_out for the same run")
    p.add_argument("--model", default="model/decay_gbdt.onnx")
    p.add_argument("--tree", default="PathLength", help="tree name in the RID file")
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--max-outlier-frac", type=float, default=1e-3)
    p.add_argument("--json", help="also write the agreement numbers to this file (read by make_presentation.py)")
    args = p.parse_args()

    try:
        import uproot
    except ImportError:
        sys.exit("verify_inference.py needs uproot:  pip install uproot")

    # --- cooked side --------------------------------------------------------
    # makeBranch writes the object whole (splitlevel 0), so the members come back
    # as fields of one branch rather than as separate sub-branches.
    with uproot.open(args.rid) as f:
        t = f[args.tree]
        r = t["ReactionID"].array(library="ak")
        cooked = pd.DataFrame(
            {
                # Join on the tree entry number, not EventInfo: Chef creates a
                # fresh EventInfo per output tree and only chains containing a
                # stage that fills it carry a real event number.
                "entry": np.asarray(r["entry"], dtype=np.int64),
                "cooked_score": np.asarray(r["score"], dtype=np.float64),
                "cooked_flag": np.asarray(r["is_decay"], dtype=bool),
                "valid": np.asarray(r["inputs_valid"], dtype=bool),
                "n_inputs": np.asarray(r["n_inputs"], dtype=np.int32),
                "model_id": np.asarray(r["model_id"], dtype=np.int32),
            }
        )
    # Chef fills the tree for every event, but process() is skipped for some
    # (cryptor's blinding), and those rows carry the previous event's verdict.
    # A row is genuinely this event's only when entry == its tree index.
    n_all = len(cooked)
    cooked = cooked[cooked["entry"].to_numpy() == np.arange(n_all)]
    print(f"cooked: {n_all} tree entries, {len(cooked)} actually scored "
          f"({n_all - len(cooked)} skipped upstream and carrying stale values), "
          f"n_inputs={sorted(set(cooked['n_inputs']))}")

    # --- python side, from the CSV the same chain wrote ----------------------
    df = pd.read_csv(args.csv)
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    # The cooked branch records which model produced it (ReactionIDResult::model_id:
    # 1 gbdt, 2 mlp). Comparing against a different model reports a large, entirely
    # meaningless disagreement, so refuse instead.
    tag = sess.get_modelmeta().custom_metadata_map.get("model_tag", "")
    want = {"gbdt": 1, "mlp": 2}.get(tag)
    have = sorted(set(cooked["model_id"]))
    if want is not None and have != [want]:
        print(f"the cooked branch was scored by model_id {have}, but --model is the {tag} export (id {want}).\n"
              f"Re-cook with -c ReactionID:setModel:<path to {os.path.basename(args.model)}> to verify it.")
        return 2
    # The model's own declared input order - the same list the cooker maps onto
    # the builder at startup. Re-deriving it from this CSV would be wrong: which
    # columns carry no information is decided on the training set, and a single
    # file can differ (one more column varies, or one fewer), which is exactly
    # the kind of drift this check exists to catch rather than reproduce.
    declared = sess.get_modelmeta().custom_metadata_map.get("input_columns", "")
    cols = declared.split(",") if declared else reactionData.feature_columns(df)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"CSV lacks {len(missing)} of the model's inputs, e.g. {missing[:5]} - schema mismatch")
        return 1
    X = df[cols].to_numpy(dtype=np.float32)
    df = df.assign(py_score=sess.run(["probability"], {"features": X})[0].ravel().astype(np.float64))

    merged = cooked.merge(df[["entry", "py_score"] + cols], on="entry", how="inner")
    merged = merged[merged["valid"]]
    if merged.empty:
        sys.exit("no events in common between the RID file and the CSV - are they from the same run?")
    print(f"joined on event number: {len(merged)} events in common")

    # --- compare ------------------------------------------------------------
    d = np.abs(merged["cooked_score"] - merged["py_score"])
    n_out = int((d > args.tol).sum())
    frac = n_out / len(d)
    print("\n=== score agreement ===")
    print(f"  mean|dP| = {d.mean():.3e}   max|dP| = {d.max():.3e}")
    print(f"  |dP| > {args.tol:g}: {n_out} of {len(d)} ({frac*100:.4f}%)")

    # Which columns are NaN in the cooker but not in the CSV? That is the usual
    # cause of a disagreement: the plugin ran on a shorter chain than the export.
    # With only a handful of outliers this comparison is noise - the "worst 200"
    # would be mostly well-agreeing rows. Only report it when there is a real tail.
    if n_out >= 20:
        worst = merged.loc[d.sort_values(ascending=False).index[:200]]
        rest = merged.loc[d.sort_values().index[: len(merged) // 2]]
        print("\n  columns most associated with disagreement (NaN rate, worst vs typical):")
        rows = []
        for c in cols:
            a, b = worst[c].isna().mean(), rest[c].isna().mean()
            if abs(a - b) > 0.2:
                rows.append((abs(a - b), c, a, b))
        for _, c, a, b in sorted(rows, reverse=True)[:8]:
            print(f"    {c:22s} {a*100:5.1f}% vs {b*100:5.1f}%")
        if not rows:
            print("    none - the disagreement is not driven by a missing column")

    ok = frac < args.max_outlier_frac
    if args.json:
        import json

        with open(args.json, "w") as f:
            json.dump({"model": os.path.basename(args.model), "model_tag": tag, "rid": args.rid, "csv": args.csv,
                       "tree_entries": int(n_all), "scored": int(len(cooked)), "stale": int(n_all - len(cooked)),
                       "compared": int(len(d)), "mean_abs": float(d.mean()), "max_abs": float(d.max()),
                       "tol": args.tol, "over_tol": n_out, "pass": bool(ok)}, f, indent=1)
    print("\n  " + ("PASS - the cooker feeds the model what Python does" if ok else "FAIL - investigate before trusting the branch"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
