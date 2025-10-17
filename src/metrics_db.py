import pandas as pd
from pathlib import Path
from typing import List
import numpy as np
import os

COLS = ['model', 'pipeline_notes', 'hyperparam_notes', 
        'roc_auc', 'accuracy', 'precision', 'recall', 'f1',
        'TN','TP','FP','FN']

DATA_DIR = Path('../data')
CSV_DIR: Path 

table: pd.DataFrame

def read_in():
    global table
    table = pd.read_csv(CSV_DIR)

def csv_dump():
    table.to_csv(CSV_DIR)

def get_table() -> pd.DataFrame:
    return table



def create_table(overwrite=True):
    path = DATA_DIR
    path.mkdir(parents=True, exist_ok=True)

    csv_path = path / 'metrics.csv'
    if overwrite and csv_path.exists():
        csv_path.unlink()  # remove existing file
    df = pd.DataFrame(columns=COLS)
    global CSV_DIR
    CSV_DIR = csv_path
    df.to_csv(CSV_DIR, index=False)
    print(f'made metrics log file {os.path.abspath(CSV_DIR)}')
    global table
    table = df
    return df

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