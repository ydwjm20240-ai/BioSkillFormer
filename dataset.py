"""
Dataset and Preprocessing Pipelines for MultiSenseBadminton.
Features participant-disjoint splitting, online sensor augmentations, and modality masking.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiSenseBadmintonDataset(Dataset):
    """
    Participant-Disjoint Multimodal Badminton Dataset.
    """
    def __init__(
        self,
        samples: List[Dict],
        is_training: bool = False,
        modality_dropout_prob: float = 0.15,
        stats: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
    ):
        self.samples = samples
        self.is_training = is_training
        self.modality_dropout_prob = modality_dropout_prob
        self.stats = stats

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_noise(self, x: np.ndarray, sigma: float = 0.02) -> np.ndarray:
        return x + np.random.normal(0, sigma, size=x.shape)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]

        pose = item["pose"].copy()
        emg = item["emg"].copy()
        pressure = item["pressure"].copy()
        gaze = item["gaze"].copy()
        video = item["video"].copy()
        cwt = item["cwt"].copy()
        bio = item["bio"].copy()
        label = item["label"]
        participant_id = item["participant_id"]

        if self.stats is not None:
            for mod_name, arr in [("pose", pose), ("emg", emg), ("pressure", pressure), ("gaze", gaze)]:
                mean, std = self.stats[mod_name]
                arr[...] = (arr - mean) / (std + 1e-8)

        if self.is_training:
            pose = self._apply_noise(pose)
            emg = self._apply_noise(emg)
            pressure = self._apply_noise(pressure)
            gaze = self._apply_noise(gaze)

        mask = np.ones(7, dtype=np.float32)
        if self.is_training and np.random.rand() < self.modality_dropout_prob:
            drop_indices = np.random.choice(7, size=np.random.randint(1, 3), replace=False)
            mask[drop_indices] = 0.0

        return {
            "pose": torch.tensor(pose, dtype=torch.float32),
            "emg": torch.tensor(emg, dtype=torch.float32),
            "pressure": torch.tensor(pressure, dtype=torch.float32),
            "gaze": torch.tensor(gaze, dtype=torch.float32),
            "video": torch.tensor(video, dtype=torch.float32),
            "cwt": torch.tensor(cwt, dtype=torch.float32),
            "bio": torch.tensor(bio, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "participant_id": participant_id
        }


def get_participant_disjoint_splits(all_samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Partitions 25 participants strictly into 17 Train, 4 Validation, and 4 Test individuals.
    """
    train_ids = {f"P{i:02d}" for i in range(1, 18)}
    val_ids = {f"P{i:02d}" for i in range(18, 22)}
    test_ids = {"P22", "P23", "P24", "P25"}

    train_data = [s for s in all_samples if s["participant_id"] in train_ids]
    val_data = [s for s in all_samples if s["participant_id"] in val_ids]
    test_data = [s for s in all_samples if s["participant_id"] in test_ids]

    return train_data, val_data, test_data