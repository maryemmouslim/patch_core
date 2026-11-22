from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .backbone import WideResNetFeatureExtractor
from .memory_bank import MemoryBank
from ..methods.b2_alignment import StatisticalAlignment
from ..methods.b3_conformal import ConformalThreshold


class PatchCore:
    def __init__(self, coreset_ratio: float = 0.05, knn_k: int = 1,
                 device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = WideResNetFeatureExtractor().to(self.device)
        self.coreset_ratio = coreset_ratio
        self.knn_k = knn_k
        self.memory = MemoryBank()
        self.alignment = StatisticalAlignment()
        self.threshold = ConformalThreshold()

    def extract(self, loader: DataLoader) -> torch.Tensor:
        batches = []
        with torch.inference_mode():
            for images, *_ in loader:
                features = self.backbone(images.to(self.device))
                batches.append(features.reshape(-1, features.shape[-1]).cpu())
        return torch.cat(batches, dim=0)

    def fit(self, train_loader: DataLoader) -> None:
        features = self.extract(train_loader)
        self.alignment.fit(features)
        self.memory.build(features).greedy_coreset(self.coreset_ratio)

    def fit_conformal(self, target_loader: DataLoader) -> None:
        _, image_scores, _ = self.predict(target_loader)
        self.threshold.fit(image_scores)

    def fit_alignment_target(self, target_loader: DataLoader) -> None:
        features = self.extract(target_loader)
        self.alignment.fit_target(features)

    def enable_alignment(self, enabled: bool) -> None:
        self.alignment.enabled = enabled

    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        labels, image_scores, pixel_scores = [], [], []
        memory = self.memory.features.to(self.device)
        with torch.inference_mode():
            for images, batch_labels, masks, _ in loader:
                features = self.backbone(images.to(self.device))
                features = features.reshape(-1, features.shape[-1])
                features = self.alignment.transform(features)

                distances = torch.cdist(features, memory)
                nearest = torch.topk(distances, k=self.knn_k, dim=1, largest=False).values.mean(dim=1)

                batch_size = images.shape[0]
                locations = nearest.reshape(batch_size, -1)
                score = locations.max(dim=1).values
                image_scores.extend(score.cpu().numpy().tolist())

                h = w = int(np.sqrt(locations.shape[1]))
                heatmap = locations.reshape(batch_size, h, w)
                heatmap = torch.nn.functional.interpolate(
                    heatmap[:, None], size=images.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(1)
                pixel_scores.extend(heatmap.cpu().numpy())
                labels.extend(batch_labels.numpy().tolist())
        return np.asarray(labels), np.asarray(image_scores), np.asarray(pixel_scores)

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.alignment.state_dict(), path / "alignment.pt")
        torch.save(self.threshold.state_dict(), path / "threshold.pt")
        self.memory.save(str(path / "memory_bank.pt"))

    def load(self, directory: str) -> None:
        path = Path(directory)
        self.alignment.load_state_dict(torch.load(path / "alignment.pt", map_location="cpu", weights_only=True))
        self.threshold.load_state_dict(torch.load(path / "threshold.pt", map_location="cpu", weights_only=True))
        self.memory.load(str(path / "memory_bank.pt"))
