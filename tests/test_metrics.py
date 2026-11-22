import numpy as np

from src.evaluation.metrics import image_metrics


def test_image_metrics_perfect_predictions():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    result = image_metrics(labels, scores, threshold=0.5)
    assert result["image_auroc"] == 1.0
    assert result["accuracy"] == 1.0

