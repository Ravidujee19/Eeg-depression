import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
from src.snn_model import EEGDepression_SNN, predict_snn
from src.utils import get_logger

logger = get_logger()

def make_dataloader(
    spikes: np.ndarray,
    y: np.ndarray,
    batch_size: int = 16,
    shuffle: bool = True
) -> DataLoader:
    spk_tensor = torch.tensor(spikes, dtype=torch.float32)   # (T, N, F)
    y_tensor   = torch.tensor(y,      dtype=torch.long)      # (N,)

    dataset = TensorDataset(
        spk_tensor.permute(1, 0, 2),  
        y_tensor
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# SNN Training Loop
def train_snn(
    model: EEGDepression_SNN,
    spikes_train: np.ndarray,
    y_train: np.ndarray,
    spikes_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    n_epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42
) -> Dict[str, List[float]]:

    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()

    train_loader = make_dataloader(spikes_train, y_train, batch_size=batch_size, shuffle=True)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  []
    }

    logger.info(f"Training SNN for {n_epochs} epochs on {device} …")

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0

        for batch_spk, batch_y in train_loader:
            batch_spk = batch_spk.permute(1, 0, 2).to(device)
            batch_y   = batch_y.to(device)

            optimizer.zero_grad()
            _, mem_rec = model(batch_spk)          # (T, batch, n_outputs)
            logits     = mem_rec.sum(dim=0)        # (batch, n_outputs)

            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_y.size(0)
            correct    += (logits.argmax(dim=1) == batch_y).sum().item()
            total      += batch_y.size(0)

        train_loss = epoch_loss / total
        train_acc  = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation
        val_loss_val = 0.0
        val_acc_val  = 0.0
        if spikes_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                spk_t   = torch.tensor(spikes_val, dtype=torch.float32).to(device)
                y_t     = torch.tensor(y_val, dtype=torch.long).to(device)
                _, mem  = model(spk_t)
                lgts    = mem.sum(dim=0)
                val_loss_val = loss_fn(lgts, y_t).item()
                val_acc_val  = (lgts.argmax(dim=1) == y_t).float().mean().item()

        history["val_loss"].append(val_loss_val)
        history["val_acc"].append(val_acc_val)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:>3}/{n_epochs} | "
                f"Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss_val:.4f} | Val Acc: {val_acc_val:.4f}"
            )

    logger.info("SNN training complete done")
    return history


# Patient Adaptation 
def patient_adaptation(
    model: EEGDepression_SNN,
    spikes_patient: np.ndarray,
    y_patient: np.ndarray,
    n_epochs: int = 3,
    lr: float = 5e-4,
    device: str = "cpu"
) -> Tuple[EEGDepression_SNN, Dict[str, List[float]]]:
    
    logger.info(f"Patient adaptation: {spikes_patient.shape[1]} samples, {n_epochs} epochs")

    # Use a lower LR to preserve general knowledge (catastrophic forgetting mitigation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()
    model     = model.to(device)

    history: Dict[str, List[float]] = {"adapt_loss": [], "adapt_acc": []}

    spk_t = torch.tensor(spikes_patient, dtype=torch.float32).to(device)
    y_t   = torch.tensor(y_patient,      dtype=torch.long).to(device)

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        _, mem_rec = model(spk_t)
        logits     = mem_rec.sum(dim=0)
        loss       = loss_fn(logits, y_t)

        loss.backward()
        optimizer.step()

        acc = (logits.argmax(dim=1) == y_t).float().mean().item()
        history["adapt_loss"].append(loss.item())
        history["adapt_acc"].append(acc)
        logger.info(f"  Adapt Epoch {epoch}/{n_epochs} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    logger.info("Patient adaptation complete done")
    return model, history
