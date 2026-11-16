"""Evaluate a checkpoint on the FIXED val and test pairs.

Protocol:
  1. val pairs  -> tune the decision threshold (max accuracy) and report val AUC.
  2. test pairs -> report accuracy at 0.5 and at the val threshold (with 95%
     Wilson CIs), AUC, precision/recall/F1. The test set is never used to choose
     anything.

Output: results JSON (deterministic: no timestamps; includes SHA-256 of the
checkpoint and of both pair files so every number is traceable), ROC plot,
and up to 10 failure cases.

Usage:
  python evaluate.py --checkpoint checkpoints/best_model.pth --out logs/eval/results.json
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.lfw_dataset import PairListDataset
from src.data.splits import read_pairs
from src.data.transforms import get_transforms
from src.models.siamese_main import AttentionalSiameseNetwork
from src.utils.metrics import (
    accuracy_at, best_threshold, calculate_precision_recall_f1, predict_probs,
    roc_auc, roc_points, wilson_ci,
)
from src.utils.seed import set_seed
from src.utils.visualization import denormalize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(model, path):
    """Accepts a bare state_dict or a dict with a 'model' key (full training checkpoint)."""
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)


def save_failure_case(img1, img2, prediction, label, idx, save_dir):
    """Saves a failure case image (side by side)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(denormalize(img1))
    axes[0].axis('off')
    axes[1].imshow(denormalize(img2))
    axes[1].axis('off')
    type_str = "False Positive" if label == 0 else "False Negative"
    plt.suptitle(f"{type_str}\nP(same): {prediction:.4f}, Label: {label}")
    plt.savefig(f"{save_dir}/fail_{idx}_{type_str.replace(' ', '_')}.png")
    plt.close(fig)


def split_report(labels, probs, threshold):
    n = len(labels)
    k05 = int(round(accuracy_at(labels, probs, 0.5) * n))
    kt = int(round(accuracy_at(labels, probs, threshold) * n))
    prf = calculate_precision_recall_f1(np.asarray(probs), np.asarray(labels), threshold=threshold)
    probs = np.asarray(probs)
    return {
        "n_pairs": n,
        "n_positive": int(np.sum(labels)),
        # Saturation diagnostics: a model whose sigmoid outputs collapse to one
        # value scores AUC 0.5 and looks merely "bad"; these make it visible.
        "prob_min": float(probs.min()),
        "prob_max": float(probs.max()),
        "n_distinct_probs": int(np.unique(probs).size),
        "auc": roc_auc(labels, probs),
        "acc_at_0.5": k05 / n,
        "acc_at_0.5_ci95": list(wilson_ci(k05, n)),
        "acc_at_val_threshold": kt / n,
        "acc_at_val_threshold_ci95": list(wilson_ci(kt, n)),
        "precision_at_val_threshold": float(prf["precision"]),
        "recall_at_val_threshold": float(prf["recall"]),
        "f1_at_val_threshold": float(prf["f1"]),
    }


def write_json_atomic(path, obj):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def plot_roc(curves, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (labels, probs, auc) in curves.items():
        fpr, tpr, _ = roc_points(labels, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate_model(checkpoint, out_path, batch_size=32, num_workers=2, n_failures=10, config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    set_seed(config["seed"])

    if not os.path.isfile(checkpoint):
        # No silent fallback to random weights: those numbers would look like a result.
        sys.exit(f"ERROR: checkpoint not found: {checkpoint}")

    root = config["dataset_path"]
    splits_dir = config["splits_dir"]
    val_csv, test_csv = os.path.join(splits_dir, "val_pairs.csv"), os.path.join(splits_dir, "test_pairs.csv")
    tf = get_transforms(split="test", input_size=tuple(config.get("input_shape", [224, 224])))
    val_ds = PairListDataset(root, read_pairs(val_csv), transform=tf)
    test_ds = PairListDataset(root, read_pairs(test_csv), transform=tf)
    mk = lambda ds: DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = AttentionalSiameseNetwork().to(DEVICE)
    load_checkpoint(model, checkpoint)
    print(f"Evaluating {checkpoint} on {DEVICE}")

    # 1) VAL: choose the threshold here, never on test
    val_p, val_y, _ = predict_probs(model, mk(val_ds), DEVICE)
    threshold, val_best_acc = best_threshold(val_y, val_p)

    # 2) TEST: touched once, nothing is chosen from it
    test_p, test_y, _ = predict_probs(model, mk(test_ds), DEVICE)

    val_rep = split_report(val_y, val_p, threshold)
    test_rep = split_report(test_y, test_p, threshold)
    results = {
        "checkpoint": checkpoint.replace("\\", "/"),
        "checkpoint_sha256": sha256(checkpoint),
        "val_pairs_sha256": sha256(val_csv),
        "test_pairs_sha256": sha256(test_csv),
        "val_threshold": threshold,
        "val": val_rep,
        "test": test_rep,
    }
    write_json_atomic(out_path, results)

    out_dir = os.path.dirname(out_path) or "."
    plot_roc({"val": (val_y, val_p, val_rep["auc"]), "test": (test_y, test_p, test_rep["auc"])},
             os.path.join(out_dir, "roc.png"))

    # Failure cases at the val threshold, in fixed test order (deterministic)
    fail_dir = os.path.join(out_dir, "failures")
    os.makedirs(fail_dir, exist_ok=True)
    wrong = np.flatnonzero((test_p > threshold) != (test_y == 1))[:n_failures]
    for j, i in enumerate(wrong):
        img1, img2, _ = test_ds[int(i)]
        save_failure_case(img1, img2, float(test_p[i]), int(test_y[i]), j, fail_dir)

    t = test_rep
    print("\n===== Test results (official LFW View 1 DevTest pairs) =====")
    print(f"pairs: {t['n_pairs']}  (positives {t['n_positive']})")
    print(f"AUC:                     {t['auc']:.4f}")
    print(f"Acc @ 0.5:               {t['acc_at_0.5']:.4f}  95% CI [{t['acc_at_0.5_ci95'][0]:.4f}, {t['acc_at_0.5_ci95'][1]:.4f}]")
    print(f"Acc @ val thr {threshold:.4f}:  {t['acc_at_val_threshold']:.4f}  95% CI "
          f"[{t['acc_at_val_threshold_ci95'][0]:.4f}, {t['acc_at_val_threshold_ci95'][1]:.4f}]")
    print(f"P / R / F1 @ val thr:    {t['precision_at_val_threshold']:.4f} / {t['recall_at_val_threshold']:.4f} / {t['f1_at_val_threshold']:.4f}")
    print(f"(val AUC {val_rep['auc']:.4f}, val acc @ thr {val_best_acc:.4f})")
    print(f"results: {out_path}  roc: {os.path.join(out_dir, 'roc.png')}  failures: {fail_dir}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    ap.add_argument("--out", default="logs/eval/results.json")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()
    evaluate_model(args.checkpoint, args.out, args.batch_size, args.num_workers)
