from typing import Optional

import numpy as np
import torch
from sklearn.random_projection import SparseRandomProjection


class MemoryBank:
    def __init__(self, features: Optional[torch.Tensor] = None):
        self.features = features

    def build(self, features: torch.Tensor) -> "MemoryBank":
        self.features = features.detach().cpu()
        return self

    def greedy_coreset(self, ratio: float = 0.05, seed: int = 42) -> "MemoryBank":
        if self.features is None:
            raise RuntimeError("The memory bank is empty")
        n_samples = max(1, int(len(self.features) * ratio))
        if n_samples >= len(self.features):
            return self
        rng = np.random.default_rng(seed)
        remaining = np.arange(len(self.features))
        selected = []
        first = int(rng.integers(len(self.features)))
        selected.append(first)
        remaining = np.delete(remaining, np.where(remaining == first))
        features_np = self.features.numpy()

        # Sparse projection speeds the greedy selection for large MVTec banks.
        output_dim = min(128, features_np.shape[1])
        projected = SparseRandomProjection(n_components=output_dim, random_state=seed).fit_transform(features_np)
        nearest = np.full(len(projected), np.inf, dtype=np.float32)
        first_projected = projected[first]
        while len(selected) < n_samples:
            batch = projected[remaining]
            distances = np.sum((batch - first_projected) ** 2, axis=1)
            nearest[remaining] = np.minimum(nearest[remaining], distances)
            chosen_position = int(np.argmax(nearest[remaining]))
            chosen = int(remaining[chosen_position])
            selected.append(chosen)
            remaining = np.delete(remaining, chosen_position)
            first_projected = projected[chosen]
        self.features = self.features[torch.tensor(selected)]
        return self

    def save(self, path: str) -> None:
        torch.save(self.features, path)

    def load(self, path: str) -> "MemoryBank":
        self.features = torch.load(path, map_location="cpu", weights_only=True)
        return self
