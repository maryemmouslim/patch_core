from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
import torch


def generate_photometric_variants(source_dir: str, output_dir: str,
                                  presets: Iterable[dict]) -> list[str]:
    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    created = []
    for image_path in sorted(source.glob("*.png")):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for index, preset in enumerate(presets):
            result = image.astype(np.float32)
            gamma = preset.get("gamma", 1.0)
            result = 255 * np.power(np.clip(result / 255.0, 0, 1), gamma)
            contrast = preset.get("contrast", 1.0)
            brightness = preset.get("brightness", 0.0)
            result = (result - 127.5) * contrast + 127.5 + brightness
            temperature = preset.get("temperature", 0.0)
            if temperature:
                result[..., 0] += temperature
                result[..., 2] -= temperature
            result = np.clip(result, 0, 255).astype(np.uint8)
            target = output / f"{image_path.stem}_variant_{index}.png"
            Image.fromarray(result).save(target)
            created.append(str(target))
    return created

