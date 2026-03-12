from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Optional

import numpy as np



def average_position_error(
    estimated_positions: np.ndarray,
    true_positions:      np.ndarray,
) -> float:

    assert estimated_positions.shape == true_positions.shape
    errors = np.linalg.norm(estimated_positions - true_positions, axis=1)
    return float(errors.mean())


def rmse_position(
    estimated_positions: np.ndarray,
    true_positions:      np.ndarray,
) -> tuple[float, float, float, float]:

    diff = estimated_positions - true_positions
    rmse_x = float(np.sqrt(np.mean(diff[:, 0]**2)))
    rmse_y = float(np.sqrt(np.mean(diff[:, 1]**2)))
    rmse_z = float(np.sqrt(np.mean(diff[:, 2]**2)))
    rmse_total = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
    return rmse_x, rmse_y, rmse_z, rmse_total


def gospa_distance(
    estimated: list[np.ndarray],
    truth:     list[np.ndarray],
    p:         float = 2.0,
    c:         float = 2000.0,    # cutoff distance [m]
    alpha:     float = 2.0,
) -> float:

    m = len(estimated)
    n = len(truth)

    if m == 0 and n == 0:
        return 0.0
    if m == 0:
        return float(n * (c / alpha) ** p) ** (1 / p)
    if n == 0:
        return float(m * (c / alpha) ** p) ** (1 / p)

    # Build cost matrix
    cost = np.zeros((m, n))
    for i, e in enumerate(estimated):
        for j, t in enumerate(truth):
            dist = np.linalg.norm(np.array(e) - np.array(t))
            cost[i, j] = min(dist, c) ** p

    # Greedy assignment (sufficient for small n, m)
    assigned_est: set[int] = set()
    assigned_tru: set[int] = set()
    total_cost = 0.0

    for _ in range(min(m, n)):
        flat_idx = np.argmin(cost)
        i, j = divmod(flat_idx, n)
        total_cost += cost[i, j]
        cost[i, :] = np.inf
        cost[:, j] = np.inf
        assigned_est.add(i)
        assigned_tru.add(j)

    # Penalise unmatched (false alarms + missed detections)
    n_missed = n - len(assigned_tru)
    n_false  = m - len(assigned_est)
    total_cost += (n_missed + n_false) * (c / alpha) ** p

    return float(total_cost ** (1.0 / p))




def classification_accuracy(
    true_labels:  list[str],
    pred_labels:  list[str],
) -> float:
    """Simple accuracy: fraction of correctly classified targets."""
    if not true_labels:
        return 0.0
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    return correct / len(true_labels)


def per_class_metrics(
    true_labels: list[str],
    pred_labels: list[str],
    classes:     Optional[list[str]] = None,
) -> dict[str, dict[str, float]]:

    if classes is None:
        classes = sorted(set(true_labels) | set(pred_labels))

    metrics: dict[str, dict[str, float]] = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)
        support = sum(1 for t in true_labels if t == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        metrics[cls] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   support,
        }
    return metrics


def print_classification_report(
    true_labels: list[str],
    pred_labels: list[str],
) -> None:
    """Pretty-print a classification report to stdout."""
    metrics = per_class_metrics(true_labels, pred_labels)
    acc = classification_accuracy(true_labels, pred_labels)

    print(f"\n  {'Class':<24} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>9}")
    print("  " + "-" * 65)
    for cls, m in metrics.items():
        print(f"  {cls:<24} {m['precision']:>10.3f} {m['recall']:>10.3f} "
              f"{m['f1']:>8.3f} {m['support']:>9}")
    print("  " + "-" * 65)
    print(f"  {'Accuracy':<24} {'':>10} {'':>10} {acc:>8.3f} {len(true_labels):>9}")




def ade(
    predicted:   np.ndarray,    # (H, 3) or (B, H, 3)
    ground_truth: np.ndarray,   # (H, 3) or (B, H, 3)
) -> float:

    diff = predicted - ground_truth
    step_errors = np.linalg.norm(diff.reshape(-1, diff.shape[-1]), axis=-1)
    return float(step_errors.mean())


def fde(
    predicted:    np.ndarray,   # (H, 3) or (B, H, 3)
    ground_truth: np.ndarray,
) -> float:

    return float(np.linalg.norm(predicted[..., -1, :] - ground_truth[..., -1, :]))


def nll_score(
    pred_means:  np.ndarray,    # (H, 3)
    pred_log_std: np.ndarray,   # (H, 3)
    ground_truth: np.ndarray,   # (H, 3)
) -> float:

    var = np.exp(2 * pred_log_std) + 1e-8
    nll = pred_log_std + 0.5 * ((ground_truth - pred_means) ** 2) / var
    return float(nll.mean())


def prediction_score_summary(
    predictions:   list,        # list[PredictedTrajectory]
    ground_truths: list[np.ndarray],
) -> dict:

    ades, fdes, nlls = [], [], []

    for pred, gt in zip(predictions, ground_truths):
        if pred is None:
            continue
        a = ade(pred.pred_positions, gt)
        f = fde(pred.pred_positions, gt)
        log_std = np.log(pred.pred_std + 1e-8)
        n = nll_score(pred.pred_positions, log_std, gt)
        ades.append(a);  fdes.append(f);  nlls.append(n)

    if not ades:
        return {}

    return {
        "mean_ade_m":  float(np.mean(ades)),
        "mean_fde_m":  float(np.mean(fdes)),
        "mean_nll":    float(np.mean(nlls)),
        "best_ade_m":  float(np.min(ades)),
        "worst_ade_m": float(np.max(ades)),
        "n_evaluated": len(ades),
    }



def detection_metrics(
    detected_ranges:  list[float],   # [m] — ranges of CFAR detections
    true_ranges:      list[float],   # [m] — ranges of true targets
    gate_m:           float = 1500.0, # association gate [m]
) -> dict[str, float]:

    matched_targets: set[int] = set()
    false_alarms = 0

    for det_r in detected_ranges:
        matched = False
        for ti, tr in enumerate(true_ranges):
            if abs(det_r - tr) < gate_m and ti not in matched_targets:
                matched_targets.add(ti)
                matched = True
                break
        if not matched:
            false_alarms += 1

    pd  = len(matched_targets) / len(true_ranges) if true_ranges else 0.0
    far = false_alarms / max(len(detected_ranges), 1)

    return {
        "pd":              round(pd, 4),
        "n_true_targets":  len(true_ranges),
        "n_detected":      len(matched_targets),
        "n_false_alarms":  false_alarms,
        "far":             round(far, 4),
    }




class LatencyProfiler:

    def __init__(self) -> None:
        self._records: dict[str, list[float]] = {}

    @contextmanager
    def measure(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - t0) * 1000.0   # ms
            self._records.setdefault(name, []).append(elapsed)

    def mean_ms(self, name: str) -> float:
        recs = self._records.get(name, [])
        return float(np.mean(recs)) if recs else 0.0

    def report(self) -> None:
        total = sum(self.mean_ms(n) for n in self._records)
        print("\n  ┌─────────────────────────────────────────────────┐")
        print(  "  │  SkySentinel — Pipeline Latency Report          │")
        print(  "  ├──────────────────────────┬──────────┬───────────┤")
        print(  "  │  Stage                   │  Mean ms │  Calls    │")
        print(  "  ├──────────────────────────┼──────────┼───────────┤")
        for name, recs in sorted(self._records.items(),
                                  key=lambda kv: np.mean(kv[1]), reverse=True):
            print(f"  │  {name:<26}│ {np.mean(recs):>8.2f} │ {len(recs):>9} │")
        print(  "  ├──────────────────────────┼──────────┼───────────┤")
        print(f"  │  {'TOTAL (per scan)':<26}│ {total:>8.2f} │           │")
        print(  "  └──────────────────────────┴──────────┴───────────┘\n")

    def to_dict(self) -> dict:
        return {
            name: {
                "mean_ms":  round(float(np.mean(recs)), 3),
                "max_ms":   round(float(np.max(recs)),  3),
                "n_calls":  len(recs),
            }
            for name, recs in self._records.items()
        }