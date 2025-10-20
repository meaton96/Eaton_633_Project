import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from typing import Tuple


def best_threshold_by_fbeta(y_true, scores, beta=2.0):
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    precisions_t = precisions[:-1]
    recalls_t = recalls[:-1]
    fbeta = (1 + beta**2) * (precisions_t * recalls_t) / (beta**2 * precisions_t + recalls_t + 1e-12)
    best_idx = int(np.argmax(fbeta))
    return {
        "threshold": thresholds[best_idx],
        "precision": float(precisions_t[best_idx]),
        "recall": float(recalls_t[best_idx]),
        "fbeta": float(fbeta[best_idx]),
        "curve": (thresholds, precisions_t, recalls_t, fbeta, best_idx),
    }

def get_scores(estimator, X):
    # Try probabilities first
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        # assume positive class is column 1
        return proba[:, 1]
    # Fall back to decision_function
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    # Last resort: use predicted labels (not great for sweeping)
    return estimator.predict(X).astype(float)

def plot_scoring(curve_info: Tuple):
    
    thr, P, R, F, idx = curve_info
    plt.figure(figsize=(8,5))
    plt.plot(thr, R, label="Recall")
    plt.plot(thr, P, label="Precision")
    plt.plot(thr, F, label="F2")
    plt.axvline(thr[idx], linestyle="--", label=f"Best thr={thr[idx]:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold vs Precision/Recall/F2")
    plt.legend()
    plt.grid(True)
    plt.show()


