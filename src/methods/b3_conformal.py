from typing import Dict

import numpy as np
import torch


class ConformalThreshold:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.threshold = None

    def fit(self, normal_scores: np.ndarray) -> None:
        normal_scores = np.asarray(normal_scores, dtype=np.float32).ravel()
        if len(normal_scores) == 0:
            raise ValueError("Conformal calibration needs normal target images")
        rank = int(np.ceil((len(normal_scores) + 1) * (1 - self.alpha)))
        rank = min(rank, len(normal_scores))
        self.threshold = float(np.sort(normal_scores)[rank - 1])

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Conformal threshold is not calibrated")
        return (np.asarray(scores) > self.threshold).astype(np.int64)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        value = -1.0 if self.threshold is None else self.threshold
        return {"threshold": torch.tensor(value), "alpha": torch.tensor(self.alpha)}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        value = float(state["threshold"].item())
        self.threshold = None if value < 0 else value
        self.alpha = float(state["alpha"].item())
