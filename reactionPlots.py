import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader

import reactionModel

# Region codes written by muonDecay_out. The binary label lumps decays in the target
# together with decays metres downstream of SPS; these are very different problems.
REGION_NAMES = {0: "no decay", 1: "upstream", 2: "target", 3: "downstream"}


def predict(model, X, y, device, batch_size=512):
    """Run the model over X in file order.

    The validation loader must not shuffle: the returned arrays are matched
    row-by-row against the metadata frame by the callers below.
    """
    model.eval()
    ds = reactionModel.inputLineData(data_values=torch.tensor(X, dtype=torch.float32), line_parameters=torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    logits, truth = [], []
    criterion = nn.BCEWithLogitsLoss()
    tot_loss, tot_n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            n = yb.size(0)
            tot_loss += criterion(out, yb.unsqueeze(1)).item() * n
            tot_n += n
            logits.append(torch.flatten(out).cpu())
            truth.append(torch.flatten(yb).cpu())

    logits = torch.cat(logits).numpy()
    truth = torch.cat(truth).numpy()
    return logits, truth, tot_loss / max(tot_n, 1)


def working_points(truth, score, prior=None, targets=(0.50, 0.80, 0.90, 0.95, 0.99), negatives=None):
    """Efficiency/rejection at fixed signal efficiency, optionally re-weighted.

    The label is positive for 13% of the MC events (16% after the chv cut); for a
    210 MeV/c muon the physical probability of decaying over the labelled window
    is about 0.1%. Precision measured on the sample as generated is therefore
    meaningless for real data, so quote it at the physical prior instead
    (reactionData.physical_prior).
    """
    fpr, tpr, thr = roc_curve(truth, score)
    rows = []
    for target in targets:
        i = int(np.searchsorted(tpr, target))
        i = min(i, len(tpr) - 1)
        eff, fa = tpr[i], fpr[i]
        # The rate that matters for precision on real data is the one on the
        # background real data is made of. With `negatives` (a mask of the
        # label-0 events to count), fpr is measured on that subset only.
        if negatives is not None:
            fa = float((score[negatives] >= thr[i]).mean())
        row = {"target_eff": target, "eff": eff, "fpr": fa, "rejection": (1.0 / fa if fa > 0 else np.inf), "threshold": thr[i]}
        if prior is not None:
            # Precision at an arbitrary signal prior pi: pi*eff / (pi*eff + (1-pi)*fpr)
            denom = prior * eff + (1.0 - prior) * fa
            row["precision_at_prior"] = (prior * eff / denom) if denom > 0 else float("nan")
        rows.append(row)
    return rows


def report_metrics(truth, logits, meta=None, prior=None, benchmark=None, benchmark_name="ReactionID cuts", threshold=0.5):
    score = 1.0 / (1.0 + np.exp(-logits))
    pred = (score > threshold).astype(np.float32)

    print("\n=== Classification ===")
    print(f"  events           : {len(truth)}")
    print(f"  decay fraction   : {truth.mean():.4f}   (sample prior)")
    print(f"  accuracy         : {(pred == truth).mean()*100:.3f}%   (score > {threshold:g})")
    fpr, tpr, _ = roc_curve(truth, score)
    print(f"  ROC AUC          : {auc(fpr, tpr):.4f}")
    print(f"  avg precision    : {average_precision_score(truth, score):.4f}  (at the sample prior, not physical)")

    print("\n=== Working points ===")
    hdr = f"  {'target':>7} {'eff':>7} {'fpr':>9} {'rejection':>10} {'thresh':>9}"
    if prior is not None:
        hdr += f" {'prec@' + format(prior, '.4f'):>12}"
    print(hdr)
    for r in working_points(truth, score, prior=prior):
        line = f"  {r['target_eff']:>7.2f} {r['eff']:>7.3f} {r['fpr']:>9.4f} {r['rejection']:>10.1f} {r['threshold']:>9.3f}"
        if prior is not None:
            line += f" {r['precision_at_prior']:>12.4f}"
        print(line)

    # Break the result out by where the muon actually decayed. A decay 5 m past SPS
    # is a different problem from one in the target, and averaging them hides both.
    if meta is not None and "decay_region" in meta:
        print(f"\n=== By decay region (score > {threshold:g}) ===")
        region = meta["decay_region"].to_numpy()
        for code, name in REGION_NAMES.items():
            m = region == code
            if m.sum() == 0:
                continue
            print(f"  {name:>11} (n={int(m.sum()):>7}): correctly tagged {(pred[m] == truth[m]).mean()*100:6.2f}%")

    # The cooker already has a cut-based answer; a classifier that cannot beat it is
    # not earning its keep.
    if benchmark is not None:
        bm = np.asarray(benchmark, dtype=np.float64)
        valid = ~np.isnan(bm)
        if valid.sum() > 0:
            bt, bp = truth[valid], bm[valid]
            b_eff = bp[bt == 1].mean() if (bt == 1).any() else float("nan")
            b_fpr = bp[bt == 0].mean() if (bt == 0).any() else float("nan")
            print(f"\n=== Benchmark: {benchmark_name} ===")
            print(f"  defined for {valid.sum()} of {len(truth)} events ({valid.sum()/len(truth)*100:.2f}%)")
            print(f"  accuracy {(bp == bt).mean()*100:.3f}%   eff {b_eff:.4f}   fpr {b_fpr:.4f}")
            m_eff = pred[valid][bt == 1].mean() if (bt == 1).any() else float("nan")
            m_fpr = pred[valid][bt == 0].mean() if (bt == 0).any() else float("nan")
            print(f"  model on the same events: accuracy {(pred[valid] == bt).mean()*100:.3f}%   eff {m_eff:.4f}   fpr {m_fpr:.4f}")

    return score, pred


def plot_and_test_model_BCE(
    model,
    losses=None,
    valid_losses=None,
    num_epochs=None,
    device=None,
    straws=None,
    truth=None,
    lr=None,
    meta=None,
    prior=None,
    outprefix="bce",
    show=False,
    threshold=0.5,
):
    logits, truth, avg_loss = predict(model, straws, truth, device)
    print(f"Average validation loss: {avg_loss:.5f}")

    # ReactionID's cut-based verdict, the thing a classifier has to beat. It is
    # only a fair comparison where it was actually evaluated: the flag is NaN for
    # events with no reconstructed vertex, and on this MC it is additionally stuck
    # at "decay" for every vertex because the run-17606 TOF/beta alignment puts
    # out_beta below the muon cut for everything. report_metrics handles the NaNs;
    # a degenerate benchmark shows up as a rate of 1.0 rather than being hidden.
    benchmark = meta["rid_is_decay"].to_numpy() if (meta is not None and "rid_is_decay" in meta) else None
    score, pred = report_metrics(truth, logits, meta=meta, prior=prior, benchmark=benchmark, threshold=threshold)
    correct = truth == pred

    # --- training curves and score distributions ----------------------------
    _, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10), constrained_layout=True)

    if losses is not None:
        n = len(losses)
        ax[0, 0].plot(range(n), losses, color="r", label="Training loss")
        print(f"Minimum training loss: {min(losses):.5f}")
        if valid_losses:
            ax[0, 0].plot(range(len(valid_losses)), valid_losses, color="b", label="Holdout loss (early stopping)")
            print(f"Minimum holdout loss: {min(valid_losses):.5f}")
        if lr is not None:
            a = ax[0, 0].twinx()
            a.set_ylabel("Learning Rate", color="g")
            a.plot(range(len(lr)), lr, color="g", alpha=0.5, linestyle="--", label="Learning rate")
            a.tick_params(axis="y", labelcolor="g")
            a.yaxis.set_major_formatter("{:.3g}".format)
            lines, labels = ax[0, 0].get_legend_handles_labels()
            l2, lb2 = a.get_legend_handles_labels()
            ax[0, 0].legend(lines + l2, labels + lb2, fontsize=7.5)
        else:
            ax[0, 0].legend()
        ax[0, 0].set_yscale("log")
    ax[0, 0].set_title("Loss vs Epoch")
    ax[0, 0].set_xlabel("Epoch")
    ax[0, 0].set_ylabel("Loss")

    h = ax[0, 1].hist2d(truth, pred, bins=2, cmin=1)
    ax[0, 1].set_title("Truth vs Predicted Decay")
    ax[0, 1].set_xlabel("Truth")
    ax[0, 1].set_ylabel("Predicted")
    plt.colorbar(h[3], ax=ax[0, 1], label="Entries")

    ax[1, 0].hist(logits[truth == 0], bins=60, histtype="step", label="Truth: no decay")
    ax[1, 0].hist(logits[truth == 1], bins=60, histtype="step", label="Truth: decay")
    ax[1, 0].set_title("Logit by Truth Class")
    ax[1, 0].set_xlabel("Logit")
    ax[1, 0].set_ylabel("Counts")
    ax[1, 0].legend()

    ax[1, 1].hist(logits[correct], bins=60, histtype="step", alpha=0.7, label="Correct")
    ax[1, 1].hist(logits[~correct], bins=60, histtype="step", alpha=0.7, label="Wrong")
    ax[1, 1].set_title("Logit by Outcome")
    ax[1, 1].set_xlabel("Logit")
    ax[1, 1].set_ylabel("Counts")
    ax[1, 1].legend()

    plt.savefig(f"{outprefix}_loss_residuals.pdf", bbox_inches="tight", dpi=300)

    # --- ROC / PR -----------------------------------------------------------
    _, ax2 = plt.subplots(ncols=2, nrows=2, figsize=(10, 10), constrained_layout=True)

    ax2[0, 0].hist(score, bins=50, label=rf"$\mu={np.mean(score):.3f}$" + "\n" + rf"$\sigma={np.std(score):.3f}$")
    ax2[0, 0].set_title("Sigmoid Distribution")
    ax2[0, 0].set_xlabel("Score")
    ax2[0, 0].set_ylabel("Counts")
    ax2[0, 0].legend()

    if meta is not None and "decay_region" in meta:
        region = meta["decay_region"].to_numpy()
        for code, name in REGION_NAMES.items():
            m = region == code
            if m.sum() > 20:
                ax2[0, 1].hist(score[m], bins=50, histtype="step", density=True, label=f"{name} (n={int(m.sum())})")
        ax2[0, 1].set_title("Score by Decay Region")
        ax2[0, 1].set_xlabel("Score")
        ax2[0, 1].set_ylabel("Density")
        ax2[0, 1].legend(fontsize=8)

    fpr, tpr, _ = roc_curve(truth, score)
    ax2[1, 0].plot(fpr, tpr, label=f"ROC AUC = {auc(fpr, tpr):.3f}")
    ax2[1, 0].fill_between(fpr, tpr, alpha=0.1)
    ax2[1, 0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    if benchmark is not None:
        bm = np.asarray(benchmark, dtype=np.float64)
        v = ~np.isnan(bm)
        if v.sum() > 0 and (truth[v] == 1).any() and (truth[v] == 0).any():
            ax2[1, 0].plot(
                bm[v][truth[v] == 0].mean(), bm[v][truth[v] == 1].mean(), "r*", markersize=14, label="ReactionID cuts"
            )
    ax2[1, 0].set_title("ROC Curve")
    ax2[1, 0].set_xlabel("False Positive Rate")
    ax2[1, 0].set_ylabel("True Positive Rate")
    ax2[1, 0].legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(truth, score)
    ax2[1, 1].plot(recall, precision, label=f"Avg Precision = {average_precision_score(truth, score):.3f}")
    ax2[1, 1].fill_between(recall, precision, alpha=0.1)
    ax2[1, 1].axhline(truth.mean(), color="k", linestyle="--", alpha=0.5, label=f"Sample prior = {truth.mean():.3f}")
    ax2[1, 1].set_title("Precision-Recall (at the sample prior)")
    ax2[1, 1].set_xlabel("Recall")
    ax2[1, 1].set_ylabel("Precision")
    ax2[1, 1].legend()

    plt.savefig(f"{outprefix}_sigmoid.pdf", bbox_inches="tight", dpi=300)

    # --- where in the detector do the mistakes happen? ----------------------
    # `logits`, `pred` and `meta` are all in file order because the loader above
    # does not shuffle, so these masks line up with the decay coordinates.
    if meta is not None and {"MuonDecay_X", "MuonDecay_Y", "MuonDecay_Z"} <= set(meta.columns):
        loc = meta[["MuonDecay_X", "MuonDecay_Y", "MuonDecay_Z"]].to_numpy(dtype=np.float64)
        wrong = (~correct) & np.isfinite(loc).all(axis=1)
        if wrong.sum() > 0:
            _, lax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10), constrained_layout=True)
            for a, (i, j, xl, yl, xlim, ylim) in zip(
                [lax[0, 0], lax[0, 1], lax[1, 0]],
                [
                    (0, 1, "X (mm)", "Y (mm)", (-1000, 1000), (-1000, 1000)),
                    (2, 0, "Z (mm)", "X (mm)", (-2000, 6000), (-1000, 1000)),
                    (2, 1, "Z (mm)", "Y (mm)", (-2000, 6000), (-1000, 1000)),
                ],
            ):
                hh = a.hist2d(loc[wrong, i], loc[wrong, j], bins=200, cmap="viridis", cmin=1, range=[xlim, ylim])
                a.set_title(f"Misidentified decays: {yl} vs {xl}")
                a.set_xlabel(xl)
                a.set_ylabel(yl)
                plt.colorbar(hh[3], ax=a, label="Counts")

            lax[1, 1].hist(loc[np.isfinite(loc).all(axis=1), 2], bins=100, histtype="step", label="All decays")
            lax[1, 1].hist(loc[wrong, 2], bins=100, histtype="step", label="Misidentified")
            lax[1, 1].set_title("Decay Z Distribution")
            lax[1, 1].set_xlabel("Z (mm)")
            lax[1, 1].set_ylabel("Counts")
            lax[1, 1].legend()
            plt.savefig(f"{outprefix}_decay_location.pdf", bbox_inches="tight", dpi=300)

    print(f"\nWrote {outprefix}_loss_residuals.pdf, {outprefix}_sigmoid.pdf, {outprefix}_decay_location.pdf")
    if show:
        plt.show()
    plt.close("all")
