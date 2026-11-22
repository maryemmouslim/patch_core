from typing import Dict

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def pixel_auroc(labels: np.ndarray, masks: np.ndarray, heatmaps: np.ndarray) -> float:
    if masks.size == 0:
        return float("nan")
    masks = np.squeeze(np.asarray(masks))
    heatmaps = np.asarray(heatmaps)
    if masks.shape[-2:] != heatmaps.shape[-2:]:
        return float("nan")
    count = min(len(masks), len(heatmaps))
    mask = masks[:count].ravel().astype(np.uint8)
    scores = heatmaps[:count].ravel()
    valid = np.isfinite(scores)
    if len(np.unique(mask[valid])) < 2:
        return float("nan")
    return float(roc_auc_score(mask[valid], scores[valid]))


def image_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float | None = None) -> Dict[str, float]:
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores).astype(np.float32)
    result = {}
    if len(np.unique(labels)) >= 2:
        result["image_auroc"] = float(roc_auc_score(labels, scores))
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        result["image_ap"] = float(average_precision_score(labels, scores))
        result["max_f1"] = float(np.max(f1))
        if threshold is None:
            threshold = float(thresholds[np.argmax(f1[:-1])]) if len(thresholds) else 0.0
    else:
        result["image_auroc"] = float("nan")
        result["image_ap"] = float("nan")
        result["max_f1"] = float("nan")
        if threshold is None:
            threshold = float(np.mean(scores))

    predictions = (scores >= threshold).astype(np.int64)
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    result["threshold"] = float(threshold)
    result["precision"] = float(tp / max(tp + fp, 1))
    result["recall"] = float(tp / max(tp + fn, 1))
    result["specificity"] = float(tn / max(tn + fp, 1))
    result["accuracy"] = float((tp + tn) / max(len(labels), 1))
    return result
