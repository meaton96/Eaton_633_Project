from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import pandas as pd
from typing import Any


METRICS_SQL = """
INSERT INTO metrics (
    run_id,
    model,
    data,
    threshold_notes,
    pipeline_notes,
    hyperparam_notes,
    notes,
    roc_auc,
    accuracy,
    precision,
    recall,
    f1,
    TN,
    TP,
    FP,
    FN,
    est_pp_savings
) VALUES (
    :run_id,
    :model,
    :data,
    :threshold_notes,
    :pipeline_notes,
    :hyperparam_notes,
    :notes,
    :roc_auc,
    :accuracy,
    :precision,
    :recall,
    :f1,
    :TN,
    :TP,
    :FP,
    :FN,
    :est_pp_savings
)
ON CONFLICT (run_id, model, data, threshold_notes, pipeline_notes)
DO UPDATE SET
    hyperparam_notes = EXCLUDED.hyperparam_notes,
    notes = EXCLUDED.notes,
    roc_auc = EXCLUDED.roc_auc,
    accuracy = EXCLUDED.accuracy,
    precision = EXCLUDED.precision,
    recall = EXCLUDED.recall,
    f1 = EXCLUDED.f1,
    TN = EXCLUDED.TN,
    TP = EXCLUDED.TP,
    FP = EXCLUDED.FP,
    FN = EXCLUDED.FN,
    est_pp_savings = EXCLUDED.est_pp_savings,
    created_at = CURRENT_TIMESTAMP;
"""


HYPERPARAM_SQL = """
INSERT INTO hyperparameters (
    run_id,
    model,
    hyperparam_text
) VALUES (
    :run_id,
    :model,
    :hyperparam_text
)
ON CONFLICT (run_id, model) DO UPDATE
SET
    hyperparam_text = EXCLUDED.hyperparam_text,
    created_at = CURRENT_TIMESTAMP;
"""



load_dotenv()

DB_URI = os.getenv('DB_URI')



if not DB_URI:
    raise ValueError("No DB_URI found in environment variables (DB_URI).")

engine = create_engine(DB_URI, pool_pre_ping=True, pool_recycle=1800)

print('connected to database')

def _to_float(x):
    return None if x is None or (hasattr(x, "__float__") and pd.isna(x)) else float(x)

def _to_int(x):
    return None if x is None or (hasattr(x, "__int__") and pd.isna(x)) else int(x)

def log_hyperparameters(
    run_id: int,
    model: str,
    hyperparam_text: str = "None"
) -> pd.DataFrame:
    params = {
        "run_id": _to_int(run_id),
        "model": str(model),
        "hyperparam_text": str(hyperparam_text),  
    }

    with engine.begin() as conn:
        conn.execute(text(HYPERPARAM_SQL), params)


    print('logged hyperparameters')
    return pd.DataFrame([params])


def log_metric(
    run_id: int,
    model: str,
    notes: str = "None",
    pipeline_notes: str = "None",
    hyperparam_notes: str = "None",
    data: str = "test",      
    threshold_notes: str = "base",
    roc_auc: float | None = None,
    accuracy: float | None = None,
    precision: float | None = None,
    recall: float | None = None,
    f1: float | None = None,
    TN: int | None = None,
    TP: int | None = None,
    FN: int | None = None,
    FP: int | None = None,
    est_pp_savings: float = 0.0,
    write: bool = True,
) -> pd.DataFrame:

    params = {
        "run_id": _to_int(run_id),
        "model": str(model),
        "data": str(data),  # must match the enum label exactly
        "threshold_notes": str(threshold_notes) if threshold_notes is not None else None,
        "pipeline_notes": str(pipeline_notes) if pipeline_notes is not None else None,
        "hyperparam_notes": str(hyperparam_notes) if hyperparam_notes is not None else None,
        "notes": str(notes) if notes is not None else None,
        "roc_auc": _to_float(roc_auc),
        "accuracy": _to_float(accuracy),
        "precision": _to_float(precision),
        "recall": _to_float(recall),
        "f1": _to_float(f1),
        "TN": _to_int(TN),
        "TP": _to_int(TP),
        "FP": _to_int(FP),
        "FN": _to_int(FN),
        "est_pp_savings": _to_float(est_pp_savings)
    }

    if not write:
        return pd.DataFrame([params])

    with engine.begin() as conn:
        conn.execute(text(METRICS_SQL), params)

    return pd.DataFrame([params])