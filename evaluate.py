"""
Post-Hoc Permutation Attribution & Participant-Level Spearman Association.
Permutation analysis (K=30) across input streams and correlation against biomechanical indicators.
"""

from typing import Dict, List, Tuple
import copy
import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from models.bioskillformer import BioSkillFormer
from metrics import evaluate_predictions, ordinal_prediction_from_thresholds


@torch.no_grad()
def run_permutation_attribution(
    model: BioSkillFormer,
    dataset: List[Dict],
    device: torch.device,
    K: int = 30
) -> Dict[str, float]:
    """
    Shuffles each input modality independently across test samples to measure Macro-F1 drop.
    """
    model.eval()
    modality_keys = ["bio", "cwt", "pose", "emg", "pressure", "gaze", "video"]
    display_names = {
        "bio": "Biomechanical Tokens",
        "cwt": "Time-Frequency Tokens",
        "pose": "Pose Kinematics",
        "emg": "EMG Activation",
        "pressure": "Plantar Pressure",
        "gaze": "Eye Gaze",
        "video": "Video Appearance"
    }

    labels = np.array([s["label"] for s in dataset])
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    base_preds = []
    for b in loader:
        out = model(
            pose=b["pose"].to(device), emg=b["emg"].to(device), pressure=b["pressure"].to(device),
            gaze=b["gaze"].to(device), video=b["video"].to(device), cwt=b["cwt"].to(device),
            bio=b["bio"].to(device), modality_mask=b["mask"].to(device)
        )
        base_preds.extend(ordinal_prediction_from_thresholds(out["ordinal_probs"].cpu().numpy()))
    
    base_f1 = evaluate_predictions(labels, np.array(base_preds))["macro_f1"]
    degradations = {}

    for mod in modality_keys:
        iter_drops = []
        for _ in range(K):
            permuted_data = copy.deepcopy(dataset)
            perm_indices = np.random.permutation(len(dataset))
            for i, p_idx in enumerate(perm_indices):
                permuted_data[i][mod] = dataset[p_idx][mod]

            p_loader = DataLoader(permuted_data, batch_size=32, shuffle=False)
            p_preds = []
            for b in p_loader:
                out = model(
                    pose=b["pose"].to(device), emg=b["emg"].to(device), pressure=b["pressure"].to(device),
                    gaze=b["gaze"].to(device), video=b["video"].to(device), cwt=b["cwt"].to(device),
                    bio=b["bio"].to(device), modality_mask=b["mask"].to(device)
                )
                p_preds.extend(ordinal_prediction_from_thresholds(out["ordinal_probs"].cpu().numpy()))

            perm_f1 = evaluate_predictions(labels, np.array(p_preds))["macro_f1"]
            iter_drops.append(base_f1 - perm_f1)

        degradations[mod] = max(float(np.mean(iter_drops)), 0.0)

    total_drop = sum(degradations.values()) + 1e-8
    return {display_names[k]: (v / total_drop) * 100.0 for k, v in degradations.items()}


def compute_participant_level_associations(
    participant_predictions: Dict[str, List[int]],
    participant_biomechanics: Dict[str, np.ndarray]
) -> Dict[str, Tuple[float, float]]:
    """
    Computes out-of-fold Spearman rank correlation between predictions and biomechanics.
    """
    indicator_names = [
        "Joint ROM",
        "SPARC Smoothness",
        "EMG Impact Offset",
        "Loading Symmetry",
        "Peak Foot Pressure",
        "Coordination Velocity"
    ]

    p_ids = list(participant_predictions.keys())
    mean_preds = [np.mean(participant_predictions[p]) for p in p_ids]
    bio_arrays = np.stack([np.mean(participant_biomechanics[p], axis=0) for p in p_ids])

    results = {}
    for i, name in enumerate(indicator_names):
        rho, p_val = spearmanr(mean_preds, bio_arrays[:, i])
        results[name] = (float(rho), float(p_val))

    return results