import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class inputLineData(Dataset):
    def __init__(self, data_values, line_parameters):
        if torch.is_tensor(data_values):
            self.data_values = data_values.clone().detach()
        else:
            self.data_values = torch.tensor(data_values, dtype=torch.float32)
        if torch.is_tensor(line_parameters):
            self.line_parameters = line_parameters.clone().detach()
        else:
            self.line_parameters = torch.tensor(line_parameters, dtype=torch.float32)

    def __len__(self):
        return len(self.data_values)

    def __getitem__(self, idx):
        return self.data_values[idx], self.line_parameters[idx]


# The network's hidden-layer sizes and dropout. Change them here (or pass
# widths=/dropout= to reactionLearner); they are saved in every checkpoint, so a
# model trained with other sizes still loads.
DEFAULT_WIDTHS = (256, 512, 1024, 256)
DEFAULT_DROPOUT = 0.1


class reactionLearner(nn.Module):
    """Binary decay-in-flight classifier over the flat feature vector.

    The input width comes from the data rather than a literal, so a change to the
    exporter's schema is caught at load time instead of silently mismatching.
    """

    def __init__(self, n_features, widths=DEFAULT_WIDTHS, dropout=DEFAULT_DROPOUT):
        super(reactionLearner, self).__init__()
        self.n_features = n_features
        self.widths = tuple(int(w) for w in widths)
        self.dropout = float(dropout)
        layers = []
        prev = n_features
        for w in widths:
            layers += [nn.Linear(prev, w), nn.ReLU(inplace=True), nn.BatchNorm1d(w)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = w
        layers.append(nn.Linear(prev, 1))
        self.sequence = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequence(x)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_model(model, optimizer, epoch, loss, lr, filename, standardizer=None, spec=None, extra=None):
    """Everything needed to reapply the model to new data goes in the checkpoint.

    That includes the standardisation statistics and the feature names: a model
    saved without them cannot be used on a fresh CSV, because the scaling it was
    trained under is not recoverable.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "loss": loss,
        "lr": lr,
        "n_features": getattr(model, "n_features", None),
        # The architecture, so load_model rebuilds the same network.
        "widths": list(getattr(model, "widths", DEFAULT_WIDTHS)),
        "dropout": getattr(model, "dropout", DEFAULT_DROPOUT),
        "spec": spec.state_dict() if spec is not None else None,
        "standardizer": standardizer.state_dict() if standardizer is not None else None,
        # How the data was split, so export_onnx.py can rebuild the same holdout
        # and choose the operating threshold on events the reported numbers never see.
        "extra": extra or {},
    }
    torch.save(checkpoint, filename)


def load_model(checkpoint_path, device="cpu"):
    """Rebuild the model, its standardiser and its feature spec from a checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)

    n_features = checkpoint.get("n_features")
    if n_features is None:
        raise ValueError(
            f"{checkpoint_path} predates the schema-aware checkpoints and does not record "
            "its input width or standardisation. Retrain to produce a usable checkpoint."
        )

    # Checkpoints written before the architecture was recorded used the defaults.
    model = reactionLearner(n_features, widths=checkpoint.get("widths", DEFAULT_WIDTHS),
                            dropout=checkpoint.get("dropout", DEFAULT_DROPOUT)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    from reactionData import FeatureSpec, Standardizer

    standardizer = Standardizer.from_state(checkpoint["standardizer"]) if checkpoint.get("standardizer") else None
    spec = FeatureSpec.from_state(checkpoint["spec"]) if checkpoint.get("spec") else None

    print(f"Model loaded ({n_features} input features).")
    return model, standardizer, spec, checkpoint.get("lr", 1e-2)


def checkpoint_extra(checkpoint_path):
    """The split bookkeeping save_model() recorded, or {} for older checkpoints."""
    return torch.load(checkpoint_path, weights_only=False, map_location="cpu").get("extra") or {}


def train_model(
    model,
    data,
    truth,
    num_epochs=150,
    device="cpu",
    learning_rate=5e-3,
    savePath="model/output.pth",
    validData=None,
    validDecay=None,
    patience=25,
    standardizer=None,
    spec=None,
    batch_size=128,
    extra=None,
):
    input_data = inputLineData(data_values=data, line_parameters=truth)
    train_loader = DataLoader(
        input_data, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False, num_workers=1, prefetch_factor=1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-8)

    valid_loader = None
    if validData is not None:
        valid_data = inputLineData(data_values=validData, line_parameters=validDecay)
        valid_loader = DataLoader(valid_data, batch_size=512, shuffle=False, drop_last=False, pin_memory=False, num_workers=1, prefetch_factor=1)

    losses, valid_losses, lr = [], ([] if valid_loader else None), []
    best_loss, best_epoch, epochs_run = float("inf"), -1, 0
    criterion = nn.BCEWithLogitsLoss()

    print("Training...")
    start = time.perf_counter()
    model.train()
    for epoch in range(num_epochs):
        epochs_run = epoch + 1
        avg_loss, total_samples = 0.0, 0
        for batch_data, batch_values in train_loader:
            batch_data, batch_values = batch_data.to(device), batch_values.to(device)
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_values.unsqueeze(1))
            loss.backward()
            optimizer.step()

            n = batch_values.size(0)
            avg_loss += loss.item() * n
            total_samples += n
        avg_loss /= total_samples
        lr.append(scheduler.get_last_lr()[0])
        scheduler.step()
        losses.append(avg_loss)

        monitored = avg_loss
        if valid_loader is not None:
            model.eval()
            with torch.no_grad():
                v_loss, v_total = 0.0, 0
                for d, v in valid_loader:
                    d, v = d.to(device), v.to(device)
                    n = v.size(0)
                    v_loss += criterion(model(d), v.unsqueeze(1)).item() * n
                    v_total += n
                v_loss /= v_total
                valid_losses.append(v_loss)
            model.train()
            monitored = v_loss

        # Keep the best epoch, not the last one. Cosine restarts mean the final
        # epoch is frequently not the best.
        if monitored < best_loss - 1e-6:
            best_loss, best_epoch = monitored, epoch
            save_model(model, optimizer, epoch, monitored, lr[-1], savePath, standardizer=standardizer, spec=spec, extra=extra)

        elapsed = time.perf_counter() - start
        eta = (elapsed / (epoch + 1)) * (num_epochs - epoch - 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            msg = f"Epoch {epoch+1}/{num_epochs}  lr={lr[-1]:.3g}  train={losses[-1]:.5f}"
            if valid_losses is not None:
                msg += f"  holdout={valid_losses[-1]:.5f}"
            msg += f"  best={best_loss:.5f}@{best_epoch+1}  ETA {eta/60:.1f} min"
            print(msg)

        if patience and epoch - best_epoch >= patience:
            print(f"No improvement for {patience} epochs - stopping at epoch {epoch+1}.")
            break

    print(f"Trained in {(time.perf_counter()-start)/60:.2f} minutes.")
    print(f"Best checkpoint: epoch {best_epoch+1}, loss {best_loss:.5f} -> {savePath}")

    # Return the best weights, not whatever the last epoch happened to leave behind.
    best = torch.load(savePath, weights_only=False, map_location=device)
    model.load_state_dict(best["model_state_dict"])
    return model, epochs_run, losses, valid_losses, lr
