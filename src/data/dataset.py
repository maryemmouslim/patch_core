from pathlib import Path
from typing import Iterator

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MVTecDataset(Dataset):
    def __init__(self, root: str, category: str, split: str = "train",
                 image_size: int = 224, transform=None):
        self.root = Path(root) / category
        split_dir = self.root / split
        if split == "train":
            self.image_paths = sorted((split_dir / "good").glob("*.png"))
        else:
            self.image_paths = sorted(split_dir.glob("*/*.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No MVTec images found in {split_dir}")
        self.mask_dir = self.root / "ground_truth"
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        transformed = self.transform(image)

        label = 0
        mask = torch.zeros((1, 224, 224), dtype=torch.float32)
        if "test" in path.parts:
            label = 0 if path.parent.name == "good" else 1
            if label:
                mask_path = self.mask_dir / path.parent.name / f"{path.stem}_mask.png"
                if mask_path.exists():
                    mask_image = Image.open(mask_path).convert("L")
                    mask_image = mask_image.resize((224, 224), Image.Resampling.NEAREST)
                    mask = torch.from_numpy(__import__("numpy").array(mask_image)).float()[None]
                    mask /= mask.max().clamp(min=1)
        return transformed, label, mask, str(path)

