"""
Training Pipeline for BioSkillFormer.
Two-stage optimization: Frozen ResNet-18 (epochs 1-10) -> Full end-to-end unfreezing (epochs 11-60).
"""

import os
import argparse
from typing import Dict
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.bioskillformer import BioSkillFormer
from losses import BioSkillFormerJointLoss
from dataset import MultiSenseBadmintonDataset, get_participant_disjoint_splits
from metrics import evaluate_predictions, ordinal_prediction_from_thresholds


def train_one_epoch(
    model: BioSkillFormer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: BioSkillFormerJointLoss,
    device: torch.device
) -> Dict[str, float]:
    model.train()
    running_losses = {"total_loss": 0.0, "loss_ce": 0.0, "loss_ord": 0.0, "loss_con": 0.0, "loss_bio": 0.0, "loss_rec": 0.0}

    for batch in loader:
        optimizer.zero_grad()
        labels = batch["label"].to(device)
        bio = batch["bio"].to(device)

        outputs = model(
            pose=batch["pose"].to(device),
            emg=batch["emg"].to(device),
            pressure=batch["pressure"].to(device),
            gaze=batch["gaze"].to(device),
            video=batch["video"].to(device),
            cwt=batch["cwt"].to(device),
            bio=bio,
            modality_mask=batch["mask"].to(device)
        )

        loss, breakdown = criterion(outputs, labels, bio)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in breakdown.items():
            running_losses[k] += v * labels.size(0)

    dataset_size = len(loader.dataset)
    return {k: v / dataset_size for k, v in running_losses.items()}


@torch.no_grad()
def evaluate(
    model: BioSkillFormer,
    loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        labels = batch["label"].to(device)
        outputs = model(
            pose=batch["pose"].to(device),
            emg=batch["emg"].to(device),
            pressure=batch["pressure"].to(device),
            gaze=batch["gaze"].to(device),
            video=batch["video"].to(device),
            cwt=batch["cwt"].to(device),
            bio=batch["bio"].to(device),
            modality_mask=batch["mask"].to(device)
        )

        ord_probs = outputs["ordinal_probs"].cpu().numpy()
        preds = ordinal_prediction_from_thresholds(ord_probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    return evaluate_predictions(np.array(all_labels), np.array(all_preds))


def main():
    parser = argparse.ArgumentParser(description="Train BioSkillFormer")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Synthetic verification dataset for testing the complete pipeline
    dummy_samples = [
        {
            "pose": np.random.randn(125, 99).astype(np.float32),
            "emg": np.random.randn(125, 8).astype(np.float32),
            "pressure": np.random.randn(125, 32).astype(np.float32),
            "gaze": np.random.randn(125, 3).astype(np.float32),
            "video": np.random.randn(38, 3, 224, 224).astype(np.float32),
            "cwt": np.random.randn(142, 64, 64).astype(np.float32),
            "bio": np.random.randn(18).astype(np.float32),
            "label": np.random.choice([0, 1, 2]),
            "participant_id": f"P{np.random.randint(1, 26):02d}"
        }
        for _ in range(300)
    ]

    train_data, val_data, test_data = get_participant_disjoint_splits(dummy_samples)

    train_ds = MultiSenseBadmintonDataset(train_data, is_training=True)
    val_ds = MultiSenseBadmintonDataset(val_data, is_training=False)
    test_ds = MultiSenseBadmintonDataset(test_data, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = BioSkillFormer(d_model=128, num_layers=4, n_heads=8, dim_feedforward=512, dropout=0.1).to(device)
    criterion = BioSkillFormerJointLoss(lambda_ce=1.0, lambda_ord=1.0, lambda_con=0.2, lambda_bio=0.5, lambda_rec=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.video_encoder.freeze_backbone()

    best_val_macro_f1 = -1.0
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "bioskillformer_final.pth")

    for epoch in range(1, args.epochs + 1):
        if epoch == 11:
            model.video_encoder.unfreeze_backbone()

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Loss: {train_metrics['total_loss']:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f} | "
            f"Val MAE: {val_metrics['ordinal_mae']:.4f} | "
            f"Val QWK: {val_metrics['qwk']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), best_checkpoint_path)

    # Final Test Set Evaluation
    model.load_state_dict(torch.load(best_checkpoint_path))
    test_metrics = evaluate(model, test_loader, device)
    print("\n--- Test Set Evaluation ---")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Ordinal MAE: {test_metrics['ordinal_mae']:.4f}")
    print(f"QWK: {test_metrics['qwk']:.4f}")


if __name__ == "__main__":
    main()