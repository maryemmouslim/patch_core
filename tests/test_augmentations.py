import numpy as np

from src.data.augmentations import apply_photometric


def test_photometric_output_shape_and_range():
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    result = apply_photometric(image, brightness=20, contrast=1.2, gamma=0.8)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255
