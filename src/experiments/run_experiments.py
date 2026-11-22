import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from ..data.augmentations import apply_photometric, LIGHTING_PRESETS
from ..data.dataset import MVTecDataset
from ..evaluation.metrics import image_metrics, pixel_auroc
from ..models.patchcore import PatchCore


class PhotometricTransform:
    def __init__(self, preset: dict, image_size: int = 224):
        self.preset = preset
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = apply_photometric(np.array(image), **self.preset)
        tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return (tensor - mean) / std


def make_loader(dataset: MVTecDataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_category(root: str, category: str, args) -> dict:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = PatchCore(args.coreset, args.knn, device=device)

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if args.b1:
        def invariant_transform(image: Image.Image) -> torch.Tensor:
            presets = list(LIGHTING_PRESETS.values())
            preset = random.choice(presets)
            return PhotometricTransform(preset, args.image_size)(image)
        train_transform = invariant_transform

    train = MVTecDataset(root, category, "train", args.image_size, train_transform)
    model.fit(make_loader(train, args.batch_size, True))

    normal_reference = MVTecDataset(root, category, "test", args.image_size)
    good_paths = [path for path in normal_reference.image_paths if path.parent.name == "good"]
    normal_reference.image_paths = good_paths[:args.target_images]

    if args.b2:
        target_transform = PhotometricTransform(LIGHTING_PRESETS[args.lighting], args.image_size)
        target_normal = MVTecDataset(root, category, "test", args.image_size, target_transform)
        target_normal.image_paths = [p for p in target_normal.image_paths if p.parent.name == "good"]
        target_normal.image_paths = target_normal.image_paths[:args.target_images]
        model.fit_alignment_target(make_loader(target_normal, args.batch_size))
        model.enable_alignment(True)

    if args.b3:
        target_normal = MVTecDataset(
            root, category, "test", args.image_size,
            PhotometricTransform(LIGHTING_PRESETS[args.lighting], args.image_size),
        )
        target_normal.image_paths = [p for p in target_normal.image_paths if p.parent.name == "good"]
        target_normal.image_paths = target_normal.image_paths[:args.target_images]
        model.fit_conformal(make_loader(target_normal, args.batch_size))

    test_transform = PhotometricTransform(LIGHTING_PRESETS[args.lighting], args.image_size)
    test = MVTecDataset(root, category, "test", args.image_size, test_transform)
    labels, scores, heatmaps = model.predict(make_loader(test, args.batch_size))

    masks = []
    for _, _, mask, _ in test:
        masks.append(mask.numpy())
    masks = np.asarray(masks)
    threshold = model.threshold.threshold if args.b3 else None
    metrics = image_metrics(labels, scores, threshold)
    metrics["pixel_auroc"] = pixel_auroc(labels, masks, heatmaps)
    metrics["category"] = category
    metrics["lighting"] = args.lighting

    output = Path(args.output) / category / args.lighting
    output.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    np.save(output / "scores.npy", scores)
    np.save(output / "labels.npy", labels)
    np.save(output / "heatmaps.npy", heatmaps)
    with open(output / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="LightShift-PatchCore experiments")
    parser.add_argument("--root", default="mvtec_anomaly_detection")
    parser.add_argument("--categories", nargs="+", default=["bottle"])
    parser.add_argument("--lighting", choices=list(LIGHTING_PRESETS), default="reference")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--coreset", type=float, default=0.05)
    parser.add_argument("--knn", type=int, default=1)
    parser.add_argument("--target-images", type=int, default=10)
    parser.add_argument("--b1", action="store_true")
    parser.add_argument("--b2", action="store_true")
    parser.add_argument("--b3", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", default="artifacts")
    return parser.parse_args()


def main():
    args = parse_args()
    all_metrics = []
    for category in args.categories:
        all_metrics.append(run_category(args.root, category, args))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)


if __name__ == "__main__":
    main()
