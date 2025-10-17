import pandas as pd
from pathlib import Path
from typing import List
import numpy as np
import os

COLS = ['model', 'pipeline_notes', 'hyperparam_notes', 
        'roc_auc', 'accuracy', 'precision', 'recall', 'f1',
        'TN','TP','FP','FN']

CSV_DIR: Path 

def make_metrics_file(data_dir: str, overwrite=True):
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)

    csv_path = path / 'metrics.csv'
    if overwrite and csv_path.exists():
        csv_path.unlink()  # remove existing file
    df = pd.DataFrame(columns=COLS)
    CSV_DIR = csv_path
    df.to_csv(CSV_DIR, index=False)
    print(f'made metrics log file {os.path.abspath(CSV_DIR)}')
    return df

def set_csv_path(dir: str | Path):
    if dir is Path:
        CSV_DIR = dir
    else:
        CSV_DIR = Path(dir)

def log_metric(
        model: str,
        pipeline_notes: str = "None",
        hyperparam_notes: str = "None",
        roc_auc: float | np.float64 | None = None,
        accuracy: float | np.float64 | None = None,
        precision: float | np.float64 | None = None,
        recall: float | np.float64 | None = None,
        f1: float | np.float64 | None = None,
        TN: int | None = None,
        TP: int | None = None,
        FN: int | None = None,
        FP: int | None = None,
):
    _insert_document(
        f'{model},{pipeline_notes},{hyperparam_notes},{roc_auc},{accuracy},{precision},{recall},{f1},{TN},{TP},{FP},{FN}'
    )

def _insert_document(document: str):
    """
    takes an 8 parameter comma-deliminated value list in the form

    'model', 'pipeline_notes', 'hyperparam_notes', 
    'roc_auc', 'accuracy', 'precision', 'recall', 'f1',
    'TN', 'TP', 'FP', 'FN'

    """
    try:
        with open(CSV_DIR, 'a') as f:
            f.write(document+"\n")
    except:
        print('error writing csv document')

    



