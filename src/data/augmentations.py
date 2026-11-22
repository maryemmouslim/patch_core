import cv2
import numpy as np


def apply_photometric(image: np.ndarray, brightness: float = 0.0,
                      contrast: float = 1.0, gamma: float = 1.0,
                      temperature: float = 0.0, shadow: float = 0.0) -> np.ndarray:
    result = image.astype(np.float32)
    result = 255 * np.power(np.clip(result / 255.0, 0, 1), gamma)
    result = (result - 127.5) * contrast + 127.5 + brightness

    if temperature != 0:
        result[..., 0] += temperature
        result[..., 2] -= temperature

    if shadow > 0:
        h, w = result.shape[:2]
        x = np.linspace(-1, 1, w)[None, :]
        y = np.linspace(-1, 1, h)[:, None]
        mask = np.clip((0.5 + 0.5 * (x + y)) * shadow, 0, shadow)
        result *= 1 - mask[..., None]
    return np.clip(result, 0, 255).astype(np.uint8)


LIGHTING_PRESETS = {
    "reference": dict(brightness=0, contrast=1.0, gamma=1.0),
    "dim": dict(brightness=-35, contrast=0.85, gamma=1.4),
    "bright": dict(brightness=35, contrast=1.15, gamma=0.7),
    "blue": dict(temperature=18),
    "yellow": dict(temperature=-18),
}
