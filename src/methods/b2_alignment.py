from typing import Dict

import torch


class StatisticalAlignment:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.registered = False
        self.source_mean = None
        self.source_std = None
        self.target_mean = None
        self.target_std = None

    def fit(self, reference_features: torch.Tensor) -> None:
        features = reference_features.detach().cpu()
        self.source_mean = features.mean(dim=0)
        self.source_std = features.std(dim=0).clamp(min=1e-6)
        self.registered = True

    def fit_target(self, target_features: torch.Tensor) -> None:
        features = target_features.detach().cpu()
        self.target_mean = features.mean(dim=0)
        self.target_std = features.std(dim=0).clamp(min=1e-6)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self.registered or self.target_mean is None:
            return features
        mean = self.target_mean.to(features.device)
        std = self.target_std.to(features.device)
        source_mean = self.source_mean.to(features.device)
        source_std = self.source_std.to(features.device)
        standardized = (features - mean) / std
        return standardized * source_std + source_mean

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "registered": torch.tensor(self.registered),
            "enabled": torch.tensor(self.enabled),
            "source_mean": self.source_mean if self.source_mean is not None else torch.tensor([]),
            "source_std": self.source_std if self.source_std is not None else torch.tensor([]),
            "target_mean": self.target_mean if self.target_mean is not None else torch.tensor([]),
            "target_std": self.target_std if self.target_std is not None else torch.tensor([]),
        }

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.registered = bool(state["registered"].item())
        self.enabled = bool(state["enabled"].item())
        self.source_mean = state["source_mean"]
        self.source_std = state["source_std"]
        self.target_mean = state["target_mean"]
        self.target_std = state["target_std"]
