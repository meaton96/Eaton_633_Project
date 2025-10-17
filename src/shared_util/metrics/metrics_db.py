from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd


COLS = [
    "id",
    "model",
    "pipeline_notes",
    "hyperparam_notes",
    "notes",
    "roc_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "TN",
    "TP",
    "FP",
    "FN",
]

DEFAULT_DATA_DIR = Path("data")
DEFAULT_FILENAME = "metrics.csv"

CSV_PATH: Path = DEFAULT_DATA_DIR / DEFAULT_FILENAME
table: pd.DataFrame | None = None


def _normalise_path(csv_path: str | Path | None) -> Path:
    """Return a concrete path to the metrics CSV."""
    if csv_path is None:
        return CSV_PATH

    path = Path(csv_path)
    if path.suffix.lower() != ".csv":
        return path / DEFAULT_FILENAME
    return path


def _ensure_table_loaded() -> pd.DataFrame:
    if table is None:
        raise RuntimeError("Metrics table not loaded. Call read_in() before using CRUD helpers.")
    return table


def _set_state(new_table: pd.DataFrame, new_path: Path) -> pd.DataFrame:
    global table, CSV_PATH
    table = new_table
    CSV_PATH = new_path
    return table


def create_table(data_dir: str | Path | None = None, overwrite: bool = True) -> pd.DataFrame:
    """
    Create a metrics CSV (and directories if needed) and prime the in-memory table.
    """
    csv_path = _normalise_path(data_dir)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and not overwrite:
        df = pd.read_csv(csv_path)
        df = df.reindex(columns=COLS, fill_value=None)
    else:
        df = pd.DataFrame(columns=COLS)
        df.to_csv(csv_path, index=False)

    return _set_state(df, csv_path).copy()


def read_in(csv_path: str | Path | None = None, create_if_missing: bool = True) -> pd.DataFrame:
    """
    Load metrics from disk into memory, creating the CSV if instructed.
    """
    resolved_path = _normalise_path(csv_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    if not resolved_path.exists():
        if not create_if_missing:
            raise FileNotFoundError(f"Metrics CSV not found at {resolved_path}")
        df = pd.DataFrame(columns=COLS)
        df.to_csv(resolved_path, index=False)
    else:
        df = pd.read_csv(resolved_path)
        df = df.reindex(columns=COLS, fill_value=None)

    return _set_state(df, resolved_path).copy()


def csv_dump(csv_path: str | Path | None = None) -> None:
    """
    Persist the in-memory metrics table to disk.
    """
    df = _ensure_table_loaded()
    path = _normalise_path(csv_path)
    if path != CSV_PATH:
        _set_state(df, path)
    df.to_csv(CSV_PATH, index=False)


def get_table(copy: bool = True) -> pd.DataFrame:
    """
    Return the current in-memory table.
    """
    df = _ensure_table_loaded()
    return df.copy() if copy else df


def get_metric(metric_id: Any, model: str) -> pd.DataFrame | None:
    """
    Retrieve metrics for a given (id, model) pair.
    """
    df = _ensure_table_loaded()
    mask = (df["id"] == metric_id) & (df["model"] == model)
    if not mask.any():
        return None
    return df.loc[mask].copy()


def _build_row(
    metric_id: Any,
    model: str,
    pipeline_notes: str,
    hyperparam_notes: str,
    notes: str,
    roc_auc: float | None,
    accuracy: float | None,
    precision: float | None,
    recall: float | None,
    f1: float | None,
    TN: int | None,
    TP: int | None,
    FP: int | None,
    FN: int | None,
) -> list[Any]:
    """Construct a row with the canonical column order."""
    row_map = {
        "id": metric_id,
        "model": model,
        "pipeline_notes": pipeline_notes,
        "hyperparam_notes": hyperparam_notes,
        "notes": notes,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TN": TN,
        "TP": TP,
        "FP": FP,
        "FN": FN,
    }
    return [row_map[col] for col in COLS]


def log_metric(
    metric_id: Any,
    model: str,
    notes: str = "None",
    pipeline_notes: str = "None",
    hyperparam_notes: str = "None",
    roc_auc: float | None = None,
    accuracy: float | None = None,
    precision: float | None = None,
    recall: float | None = None,
    f1: float | None = None,
    TN: int | None = None,
    TP: int | None = None,
    FN: int | None = None,
    FP: int | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """
    Upsert a metric row identified by (id, model).
    """
    df = _ensure_table_loaded()
    mask = (df["id"] == metric_id) & (df["model"] == model)
    row = _build_row(
        metric_id,
        model,
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
    )

    if mask.any():
        df.loc[mask, COLS] = row
        selection = mask
    else:
        df.loc[len(df)] = row
        selection = df.index == len(df) - 1

    if write:
        csv_dump()

    return df.loc[selection].copy()


def delete_metric(metric_id: Any, model: str, write: bool = True) -> int:
    """
    Delete rows matching the (id, model) key. Returns number of rows removed.
    """
    df = _ensure_table_loaded()
    mask = (df["id"] == metric_id) & (df["model"] == model)
    if not mask.any():
        return 0

    removed_indices = df.index[mask]
    df.drop(index=removed_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if write:
        csv_dump()

    return len(removed_indices)


def list_metrics(columns: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Convenience accessor to list metrics with optional column sub-selection.
    """
    df = _ensure_table_loaded().copy()
    if columns is not None:
        return df.loc[:, list(columns)]
    return df
