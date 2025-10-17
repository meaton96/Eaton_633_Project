from __future__ import annotations

import pandas as pd
import pytest

from src import metrics_db


@pytest.fixture
def temp_metrics_csv(tmp_path):
    """
    Provide a fresh metrics CSV path and isolate metrics_db module state per test.
    """
    csv_root = tmp_path / "metrics_store"
    original_table = metrics_db.table
    original_path = metrics_db.CSV_PATH

    metrics_db.create_table(csv_root, overwrite=True)
    csv_path = csv_root / metrics_db.DEFAULT_FILENAME

    try:
        yield csv_path
    finally:
        metrics_db.table = original_table
        metrics_db.CSV_PATH = original_path


def test_create_table_initialises_empty_csv(temp_metrics_csv):
    df = metrics_db.get_table()
    assert list(df.columns) == metrics_db.COLS
    assert df.empty

    loaded = pd.read_csv(temp_metrics_csv)
    assert list(loaded.columns) == metrics_db.COLS
    assert loaded.empty


def test_log_metric_inserts_and_persists(temp_metrics_csv):
    metrics_db.read_in(temp_metrics_csv)

    metrics_db.log_metric(
        metric_id=1,
        model="baseline",
        accuracy=0.91,
        precision=0.88,
        recall=0.9,
        f1=0.89,
        TN=50,
        TP=45,
        FP=5,
        FN=4,
    )

    table = metrics_db.get_table()
    assert len(table) == 1
    row = table.iloc[0]
    assert row["id"] == 1
    assert row["model"] == "baseline"
    assert pytest.approx(row["accuracy"]) == 0.91

    persisted = pd.read_csv(temp_metrics_csv)
    assert len(persisted) == 1
    assert persisted.iloc[0]["model"] == "baseline"


def test_log_metric_upserts_by_primary_key(temp_metrics_csv):
    metrics_db.read_in(temp_metrics_csv)

    metrics_db.log_metric(metric_id=42, model="rf", accuracy=0.75, precision=0.7, write=False)
    metrics_db.log_metric(metric_id=42, model="rf", accuracy=0.8, precision=0.72, write=False)

    table = metrics_db.get_table()
    assert len(table) == 1

    row = table.iloc[0]
    assert pytest.approx(row["accuracy"]) == 0.8
    assert pytest.approx(row["precision"]) == 0.72


def test_get_and_delete_metric(temp_metrics_csv):
    metrics_db.read_in(temp_metrics_csv)

    metrics_db.log_metric(metric_id="exp-1", model="xgb", accuracy=0.84, write=False)
    metrics_db.log_metric(metric_id="exp-2", model="xgb", accuracy=0.85, write=False)

    match = metrics_db.get_metric("exp-1", "xgb")
    assert match is not None
    assert len(match) == 1
    assert pytest.approx(match.iloc[0]["accuracy"]) == 0.84

    removed = metrics_db.delete_metric("exp-1", "xgb", write=False)
    assert removed == 1

    table = metrics_db.get_table()
    assert len(table) == 1
    assert table.iloc[0]["id"] == "exp-2"


def test_list_metrics_column_subset(temp_metrics_csv):
    metrics_db.read_in(temp_metrics_csv)

    metrics_db.log_metric(metric_id=1, model="lr", accuracy=0.7, precision=0.69, write=False)
    metrics_db.log_metric(metric_id=2, model="svm", accuracy=0.73, precision=0.7, write=False)

    subset = metrics_db.list_metrics(columns=["id", "model", "accuracy"])
    assert list(subset.columns) == ["id", "model", "accuracy"]
    assert len(subset) == 2
