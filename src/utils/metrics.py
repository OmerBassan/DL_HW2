import math

import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def calculate_accuracy(predictions, labels):
    """
    Calculates accuracy.
    predictions: Tensor of shape [Batch, 1] (probabilities 0-1)
    labels: Tensor of shape [Batch, 1] (0 or 1)
    """
    predicted_classes = (predictions > 0.5).float()
    correct = (predicted_classes == labels).float().sum()
    return correct / labels.size(0)

def calculate_precision_recall_f1(predictions, labels, threshold=0.5):
    """
    Calculates Precision, Recall, and F1 score.
    Returns a dictionary.
    """
    # Convert to numpy for sklearn
    preds_np = (np.asarray(_np(predictions)) > threshold).astype(int).flatten()
    labels_np = np.asarray(_np(labels)).astype(int).flatten()

    precision = precision_score(labels_np, preds_np, zero_division=0)
    recall = recall_score(labels_np, preds_np, zero_division=0)
    f1 = f1_score(labels_np, preds_np, zero_division=0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x


# ------------------------------------------------------------------ inference
def predict_probs(model, loader, device, criterion=None):
    """The ONE eval loop: eval() + inference_mode() together, so neither can be
    forgotten. Returns (probs[N] float64, labels[N] int64, mean_loss or None)."""
    model.eval()
    probs, labels, loss_sum, n = [], [], 0.0, 0
    with torch.inference_mode():
        for img1, img2, y in loader:
            img1, img2 = img1.to(device), img2.to(device)
            y = y.to(device).float().view(-1, 1)
            logits, _ = model(img1, img2)
            assert logits.shape == y.shape, (logits.shape, y.shape)
            if criterion is not None:
                loss_sum += criterion(logits, y).item() * y.size(0)
            probs.append(torch.sigmoid(logits).view(-1).double().cpu())
            labels.append(y.view(-1).long().cpu())
            n += y.size(0)
    probs = torch.cat(probs).numpy()
    if not np.isfinite(probs).all():
        raise FloatingPointError("non-finite probabilities from model")
    return probs, torch.cat(labels).numpy(), (loss_sum / n if criterion is not None else None)


# -------------------------------------------------------------------- metrics
def accuracy_at(labels, probs, threshold):
    return float(np.mean((np.asarray(probs) > threshold) == (np.asarray(labels) == 1)))


def roc_auc(labels, probs):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        raise ValueError("AUC is undefined with a single class")
    return float(roc_auc_score(labels, probs))


def roc_points(labels, probs):
    fpr, tpr, thr = roc_curve(labels, probs)
    return fpr, tpr, thr


def best_threshold(labels, probs):
    """Threshold maximising accuracy; predict SAME when prob > threshold.

    Tune on VALIDATION only. Candidates are midpoints between consecutive
    distinct scores (plus one below the min and one above the max), so the
    returned value sits in a gap, not on a training score. Ties are broken by
    taking the middle of the tied candidates. With balanced classes,
    max-accuracy is equivalent to maximising Youden's J (TPR - FPR).
    """
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    u = np.unique(probs)
    cands = np.concatenate(([u[0] - 1e-6], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-6]))
    cands = np.clip(cands, 0.0, 1.0)
    accs = np.array([np.mean((probs > t) == (labels == 1)) for t in cands])
    best = np.flatnonzero(accs == accs.max())
    t = float(cands[best[len(best) // 2]])
    return t, float(accs.max())


def wilson_ci(k, n, confidence=0.95):
    """Wilson score interval for a binomial proportion k/n.

    Chosen over the normal (Wald) interval, which undercovers and can leave
    [0, 1] near the extremes. Assumes independent trials; LFW pairs share
    identities, so the true interval is somewhat wider (stated limitation).
    """
    if n <= 0 or not (0 <= k <= n):
        raise ValueError(f"invalid k={k}, n={n}")
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # Analytically lo == 0 when k == 0 and hi == 1 when k == n; set them exactly,
    # since float rounding leaves ~1e-19 and the interval would exclude k/n.
    lo = 0.0 if k == 0 else max(0.0, center - half)
    hi = 1.0 if k == n else min(1.0, center + half)
    return lo, hi
