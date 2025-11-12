import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.metrics import make_scorer, fbeta_score

def cost_curves_from_scores(
    y_true, y_scores, 
    R=16300.0, e=0.50, c_m_list=(500.0, 1000.0, 1500.0)
):
    """
    Returns PR arrays, thresholds, prevalence, flag_rate, and per-patient savings 
    curves for each c_m in c_m_list.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision, recall = precision[:-1], recall[:-1]  # align with thresholds
    p = float(np.mean(y_true))

    # flag_rate = P(flag) = (p * recall) / precision
    with np.errstate(divide='ignore', invalid='ignore'):
        flag_rate = np.where(precision > 0, (p * recall) / precision, 0.0)

    savings = {}
    for c_m in c_m_list:
        per_patient = flag_rate * (precision * e * R - c_m)
        savings[c_m] = per_patient

    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "prevalence": p,
        "flag_rate": flag_rate,
        "savings": savings,   
    }

def best_threshold_by_cost(curves, prefer_c_m=None):
    """
    Pick the threshold that maximizes per-patient savings.
    If prefer_c_m is provided, optimize for that single cost.
    Otherwise, pick the argmax over the average savings across c_m_list.
    """
    c_ms = sorted(curves["savings"].keys())
    mat = np.column_stack([curves["savings"][c] for c in c_ms]) 

    if prefer_c_m is None:
        # maximize mean savings across the cost scenarios
        agg = np.nanmean(mat, axis=1)
        j = int(np.nanargmax(agg))
        chosen_c_m = None
    else:
        # pick threshold optimal for the chosen cost
        idx = c_ms.index(prefer_c_m)
        j = int(np.nanargmax(mat[:, idx]))
        chosen_c_m = prefer_c_m

    return {
        "idx": j,
        "threshold": float(curves["thresholds"][j]),
        "precision": float(curves["precision"][j]),
        "recall": float(curves["recall"][j]),
        "flag_rate": float(curves["flag_rate"][j]),
        "savings_at_c": {c: float(curves["savings"][c][j]) for c in c_ms},
        "chosen_c_m": chosen_c_m,
        "c_m_list": c_ms,
    }

def cost_savings_by_threshold(y_true, y_scores, R=16300, c_m=1000, e=0.50):
    # PR curve returns precision and recall for descending thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # Align lengths
    precision, recall = precision[:-1], recall[:-1]
    thresholds = thresholds

    y_true = np.asarray(y_true)
    p = y_true.mean()

    # flag_rate = (p * recall) / precision
    # Guard division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        flag_rate = np.where(precision > 0, (p * recall) / precision, 0.0)

    # Expected net savings per patient at each threshold
    per_patient_savings = flag_rate * (precision * e * R - c_m)

    # Package results
    return {
        "thresholds": thresholds,
        "precision": precision,
        "recall": recall,
        "flag_rate": flag_rate,
        "per_patient_savings": per_patient_savings,
        "best_idx": int(np.nanargmax(per_patient_savings)),
    }


# def make_f2_scorer():
#     return make_scorer(fbeta_score, beta=2, average='binary')



# def best_threshold_by_fbeta(y_true, scores, beta=2.0):
#     precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
#     precisions_t = precisions[:-1]
#     recalls_t = recalls[:-1]
#     fbeta = (1 + beta**2) * (precisions_t * recalls_t) / (beta**2 * precisions_t + recalls_t + 1e-12)
#     best_idx = int(np.argmax(fbeta))
#     return {
#         "threshold": thresholds[best_idx],
#         "precision": float(precisions_t[best_idx]),
#         "recall": float(recalls_t[best_idx]),
#         "fbeta": float(fbeta[best_idx]),
#         "curve": (thresholds, precisions_t, recalls_t, fbeta, best_idx),
#     }

def get_scores(estimator, X):
    # Try probabilities first
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        # positive class is column 1
        return proba[:, 1]
    # Fall back to decision_function
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
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


