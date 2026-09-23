"""Train or evaluate the muon decay-in-flight classifier.

    python reactionID.py train --train data/train.csv --valid data/valid.csv
    python reactionID.py eval  --valid data/valid.csv --checkpoint model/output.pth

Inputs come from the cooker's muonDecay_out plugin via script/cook_mc_chain.sh in
the muse repository; see reactionData.py for the CSV contract.
"""

import argparse
import os

import torch

import reactionData
import reactionModel
import reactionPlots


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["train", "eval"], help="train a new model, or evaluate a checkpoint")
    p.add_argument("--train", default="data/train.csv", help="training feature CSV")
    p.add_argument("--valid", default="data/valid.csv", help="validation feature CSV")
    p.add_argument("--checkpoint", default="model/output.pth", help="checkpoint path (written by train, read by eval)")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--patience", type=int, default=25, help="early stopping patience in epochs; 0 disables")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--holdout-frac", type=float, default=0.15,
                   help="share of the training data held out for early stopping (and, in export_onnx.py, thresholds)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument(
        "--physical-prior",
        type=float,
        default=None,
        help="decay probability to quote precision at. The MC is enriched far above the "
        "physical rate, so precision on the sample as generated is not a physics number.",
    )
    p.add_argument("--outprefix", default="bce", help="prefix for the output PDFs")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="decision threshold for accuracy and the per-region table; the shipped one is in the ONNX metadata")
    p.add_argument("--no-categorical", action="store_true", help="feed bar/PID indices as numbers instead of one-hot")
    p.add_argument("--show", action="store_true", help="open the plots interactively")
    return p.parse_args()


def main():
    args = parse_args()
    reactionModel.set_seed(args.seed)
    device = pick_device(args.device)
    print(f"Device: {device}")

    categorical = not args.no_categorical

    if args.mode == "train":
        # Early stopping picks an epoch, which is a fit to whatever set it watches.
        # Watch a holdout carved from training, so the validation file is scored
        # once, by a model that was never selected on it.
        fit_df, hold_df = reactionData.holdout_split(reactionData.load_csv(args.train), args.holdout_frac, args.seed)
        trainX, trainY, columns, _, spec, standardizer = reactionData.load_dataset(None, df=fit_df, categorical=categorical)
        # The same spec and standardiser are reused for the holdout and validation,
        # so the sets cannot silently expand to different feature layouts.
        holdX, holdY, hcols, _, _, _ = reactionData.load_dataset(
            None, spec=spec, standardizer=standardizer, categorical=categorical, df=hold_df
        )
        validX, validY, vcols, validMeta, _, _ = reactionData.load_dataset(
            args.valid, spec=spec, standardizer=standardizer, categorical=categorical
        )
        assert vcols == columns and hcols == columns, "spec reuse failed"
        print(f"fit {trainX.shape}  holdout {holdX.shape}  valid {validX.shape}  decay fraction {trainY.mean():.4f}")

        os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
        model = reactionModel.reactionLearner(n_features=trainX.shape[1]).to(device)
        model, epochs, losses, valid_losses, lr = reactionModel.train_model(
            model,
            data=torch.tensor(trainX),
            truth=torch.tensor(trainY),
            num_epochs=args.epochs,
            device=device,
            learning_rate=args.lr,
            savePath=args.checkpoint,
            validData=torch.tensor(holdX),
            validDecay=torch.tensor(holdY),
            patience=args.patience,
            standardizer=standardizer,
            spec=spec,
            batch_size=args.batch_size,
            extra={"train": args.train, "holdout_frac": args.holdout_frac, "seed": args.seed},
        )
    else:
        model, standardizer, spec, lr = reactionModel.load_model(args.checkpoint, device=device)
        if standardizer is None or spec is None:
            raise ValueError(f"{args.checkpoint} carries no feature spec or standardisation; retrain it.")
        losses = valid_losses = epochs = None

        df = reactionData.load_csv(args.valid)
        rawX, validY, names, validMeta = reactionData.build_matrix(df, spec)
        if rawX.shape[1] != model.n_features:
            raise ValueError(
                f"the CSV expands to {rawX.shape[1]} features but the checkpoint expects {model.n_features}. "
                "The exporter schema changed since this model was trained."
            )
        validX = standardizer.transform(rawX)

    reactionPlots.plot_and_test_model_BCE(
        model=model,
        losses=losses,
        valid_losses=valid_losses,
        num_epochs=epochs,
        device=device,
        straws=validX,
        truth=validY,
        lr=lr if args.mode == "train" else None,
        meta=validMeta,
        prior=args.physical_prior,
        outprefix=args.outprefix,
        show=args.show,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
