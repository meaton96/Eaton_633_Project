from __future__ import annotations
from importlib.resources import files
from pathlib import Path
import pandas as pd

_PKG = "shared_util"

def data_path(name: str) -> Path:
    return Path(files(f"{_PKG}.data") / name) #type: ignore

def load_csv(name: str, **read_csv_kwargs) -> pd.DataFrame:
    return pd.read_csv(data_path(f'{name}.csv'), index_col=0, **read_csv_kwargs)

def list_data() -> list[str]:
    p = files(f"{_PKG}.data")
    return sorted([entry.name for entry in p.iterdir() if entry.name.endswith(".csv")])
