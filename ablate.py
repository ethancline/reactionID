"""What each group of inputs is actually worth, and how much of the score is an artifact.

Three questions this answers, all of which the AUC on its own hides:

1. **How much is reconstruction failure?** Training on nothing but the pattern of
   which columns are NaN - no measured values at all - reaches AUC ~0.91 on this MC.
   Decays break reconstruction, so that correlation is real here, but it is a
   property of the reconstruction rather than of the physics and will not transfer
   to data whose failure modes differ. The `nanonly` control measures it directly,
   and it has to be re-run after every change: if it climbs alongside the full
   model, the "improvement" is more of the same artifact.

2. **What does each detector add?** Leave-one-group-out on the validation set. With
   AUC already past 0.99, report the change in *error* rate, not in AUC - a drop
   from 0.9963 to 0.9950 looks like nothing and is a 35% increase in mistakes.

3. **How much of the performance is only available when reconstruction worked?**
   Every number is quoted twice: over all events, and over the subset with a full
   set of measurements. The gap is the part that does not transfer.

Uses the boosted trees with the shipped settings (model/gbdt_params.json), so
every number describes the model that ships; NaN is handled natively. One fit
per configuration, scored on all events and on the fully reconstructed subset.

    python ablate.py                      # every group, plus the controls
    python ablate.py --only vertex_pull   # one group
"""

import argparse
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import reactionData

# Input groups, by column-name prefix or exact name. Each is a thing a physicist
# would decide to add or not add, rather than an individual column.
GROUPS = {
    "vertex_pull": ("vertex angle/DOCA pulls", ("vtx_theta_err", "vtx_doca_err", "vtx_ex", "vtx_ey", "vtx_ez", "vtx_theta_pull", "vtx_doca_pull")),
    "vertex_geom": ("vertex position, angle, DOCA", ("vtx_x", "vtx_y", "vtx_z", "vtx_theta", "vtx_doca", "vtx_side", "n_vertices")),
    "bm": ("beam monitor walls", tuple(f"bm{i}_" for i in range(3))),
    "oot": ("out-of-time scintillator hits", ("_oot_",)),
    "doca_signed": ("signed DOCA components", ("_doca_x", "_doca_y")),
    "scint_time": ("per-wall in-time time", ("bhc_time", "bhd_time", "spslf_time", "spslr_time", "spsrf_time", "spsrr_time", "veto_time")),
    "rid_flags": ("ReactionID selection flags", ("rid_",)),
    "tof": ("time of flight and BH/SPS correlation", ("tof_raw", "bh_avg_time", "bh_both_planes", "sps_corr_time", "sps_corr_rf", "bhc_corr_time", "bhd_corr_time")),
    "kink": ("front/rear STT half-chamber split", ("stt_nhits_front", "stt_nhits_rear", "stt_nhits_asym",
                                                   "stt_dist_front", "stt_dist_rear", "stt_dist_diff",
                                                   "stt_tracklet", "stt_ntracklets")),
    "stt": ("STT track", ("stt_",)),
    "gem": ("GEM track", ("gem_",)),
}


# Groups nested inside a broader prefix. "stt_" also matches every kink column, so
# without this, "without stt" silently removed the kink group too and the two
# rows could not be compared.
CARVED_OUT = {"stt": ("kink",)}


def matches(col, patterns):
    return any(col == p or col.startswith(p) or p in col for p in patterns)


def in_group(col, key):
    if not matches(col, GROUPS[key][1]):
        return False
    return not any(matches(col, GROUPS[other][1]) for other in CARVED_OUT.get(key, ()))


def fit(Xtr, ytr, seed):
    # The settings the shipped model is fitted with (model/gbdt_params.json via
    # reactionData), so every number here describes that model, not a proxy.
    clf = HistGradientBoostingClassifier(random_state=seed, **reactionData.gbdt_params())
    return clf.fit(Xtr, ytr)


def fit_auc(Xtr, ytr, Xva, yva, seed, max_iter=None):
    clf = fit(Xtr, ytr, seed)
    return roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default="data/train.csv", help="training CSV, or several joined with commas")
    p.add_argument("--valid", default="data/valid.csv")
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--only", help="run just this group")
    p.add_argument("--reconstructed-on", default="sps_corr_time",
                   help="column whose presence defines a fully-reconstructed event")
    p.add_argument("--json", help="also write every number printed here to this file")
    args = p.parse_args()
    out = {"label": reactionData.LABEL_COLUMN, "groups": []}

    train_df = reactionData.load_csv(args.train)
    valid_df = reactionData.load_csv(args.valid)
    cols = reactionData.feature_columns(train_df)

    ytr = train_df[reactionData.LABEL_COLUMN].to_numpy(dtype=np.int32)
    yva = valid_df[reactionData.LABEL_COLUMN].to_numpy(dtype=np.int32)
    Xtr = train_df[cols].to_numpy(dtype=np.float32)
    Xva = valid_df[cols].to_numpy(dtype=np.float32)

    reco = ~np.isnan(valid_df[args.reconstructed_on].to_numpy(dtype=np.float64))
    print(f"\ntrain {Xtr.shape}  valid {Xva.shape}  positives {ytr.mean()*100:.2f}%")
    print(f"fully reconstructed ({args.reconstructed_on} present): {reco.sum()} of {len(reco)} "
          f"({100*reco.mean():.1f}%), positives there {yva[reco].mean()*100:.2f}%")

    def evaluate(name, keep_mask, note=""):
        # One fit, scored on all events and on the fully reconstructed subset.
        t0 = time.perf_counter()
        clf = fit(Xtr[:, keep_mask], ytr, args.seed)
        p = clf.predict_proba(Xva[:, keep_mask])[:, 1]
        a_all = roc_auc_score(yva, p)
        a_rec = roc_auc_score(yva[reco], p[reco])
        print(f"  {name:<28} {int(keep_mask.sum()):4d} cols   AUC {a_all:.4f}   reco-only {a_rec:.4f}   "
              f"({time.perf_counter()-t0:.0f}s) {note}")
        return a_all, a_rec

    print("\n=== Controls ===")
    baseline, baseline_rec = evaluate("all inputs", np.ones(len(cols), dtype=bool))

    # The artifact control: hand the model only which columns are missing.
    nan_tr = np.isnan(train_df[cols].to_numpy(dtype=np.float64)).astype(np.float32)
    nan_va = np.isnan(valid_df[cols].to_numpy(dtype=np.float64)).astype(np.float32)
    varies = nan_tr.std(axis=0) > 0
    t0 = time.perf_counter()
    nan_auc = fit_auc(nan_tr[:, varies], ytr, nan_va[:, varies], yva, args.seed)
    print(f"  {'ONLY which cols are NaN':<28} {int(varies.sum()):4d} cols   AUC {nan_auc:.4f}   "
          f"{'':>15} ({time.perf_counter()-t0:.0f}s)  <- no measured values at all")

    # The same inputs asked the old question - decay anywhere, including metres
    # downstream of the target - so the effect of the relabel is measured in the
    # same run as everything else rather than quoted from an earlier model.
    old_tr = train_df["MuonDecay"].to_numpy(dtype=np.int32)
    old_va = valid_df["MuonDecay"].to_numpy(dtype=np.int32)
    old_auc = fit_auc(Xtr, old_tr, Xva, old_va, args.seed)
    print(f"  {'old label (decay anywhere)':<28} {len(cols):4d} cols   AUC {old_auc:.4f}")

    # Every group added in this round of work, removed together: the new label on
    # roughly the old input set.
    added = ("vertex_pull", "bm", "oot", "doca_signed", "scint_time", "kink")
    in_added = np.array([any(in_group(c, g) for g in added) for c in cols])
    no_add_auc, no_add_rec = evaluate("without any new inputs", ~in_added, f"[{int(in_added.sum())} removed]")

    # RF phase is excluded from the inputs (see reactionData.EXCLUDED_COLUMNS).
    # Measure what that costs, in this run: RF added back, and RF on its own.
    rf_cols = [c for c, why in reactionData.EXCLUDED_COLUMNS.items()
               if why.startswith("RF phase") and c in train_df.columns
               and not np.isnan(train_df[c].to_numpy(dtype=np.float64)).all()]
    rf = {}
    if rf_cols:
        Xtr_rf = train_df[cols + rf_cols].to_numpy(dtype=np.float32)
        Xva_rf = valid_df[cols + rf_cols].to_numpy(dtype=np.float32)
        rf["with"] = fit_auc(Xtr_rf, ytr, Xva_rf, yva, args.seed)
        rf["only"] = fit_auc(train_df[rf_cols].to_numpy(np.float32), ytr, valid_df[rf_cols].to_numpy(np.float32), yva, args.seed)
        rf["without"] = baseline
        rf["n_cols"] = len(rf_cols)
        print(f"  {'RF phase added back':<28} {len(cols)+len(rf_cols):4d} cols   AUC {rf['with']:.4f}")
        print(f"  {'RF phase only':<28} {len(rf_cols):4d} cols   AUC {rf['only']:.4f}")

    out.update(n_train=int(len(ytr)), n_valid=int(len(yva)), n_inputs=len(cols), rf=rf, train=args.train,
               gbdt_params=reactionData.gbdt_params(),
               positive_fraction=float(yva.mean()), reco_fraction=float(reco.mean()),
               all_inputs={"auc": baseline, "auc_reco": baseline_rec},
               nan_only={"auc": nan_auc, "n_cols": int(varies.sum())},
               old_label={"auc": old_auc},
               without_new_inputs={"auc": no_add_auc, "auc_reco": no_add_rec, "n_removed": int(in_added.sum())})

    groups = {args.only: GROUPS[args.only]} if args.only else GROUPS
    print("\n=== Leave one group out (change in error rate, 1 - AUC) ===")
    rows = []
    for key, (label, patterns) in groups.items():
        mask = np.array([in_group(c, key) for c in cols])
        if not mask.any():
            print(f"  {key:<28} no matching columns - skipped")
            continue
        a_all, a_rec = evaluate(f"without {key}", ~mask, f"[{int(mask.sum())} removed]")
        rows.append((key, label, int(mask.sum()), a_all, a_rec))

    print(f"\n=== Worth of each group, vs all-inputs AUC {baseline:.4f} ===")
    print(f"  {'group':<14} {'cols':>4}  {'dAUC':>8}  {'error rate change':>19}   what it is")
    for key, label, n, a_all, _ in sorted(rows, key=lambda r: r[3]):
        d = a_all - baseline
        ratio = (1 - a_all) / (1 - baseline) if baseline < 1 else float("nan")
        print(f"  {key:<14} {n:>4}  {d:+8.4f}  {ratio:>18.2f}x   {label}")
    print("\n  A ratio above 1 means removing the group costs accuracy; 1.35x means 35% more mistakes.")
    print(f"  NaN-pattern-only control: {nan_auc:.4f}. If this rises with the full model, the gain is")
    print("  more reconstruction-failure correlation rather than physics.")

    if args.json:
        import json
        for key, label, n, a_all, a_rec in rows:
            out["groups"].append({"key": key, "label": label, "n_cols": n, "auc": a_all, "auc_reco": a_rec,
                                  "error_ratio": (1 - a_all) / (1 - baseline)})
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
