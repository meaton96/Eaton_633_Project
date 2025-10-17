import pandas as pd
from src.utils import make_metrics_file, COLS, log_metric
import pytest
from pathlib import Path

DATA_DIR: Path = Path('data/temp')

def test_make_metrics_file_creates_csv(tmp_path: Path | str = "data/temp"):
    # tmp_path is a pathlib.Path for a fresh temporary directory
    if tmp_path is Path:
        data_dir = tmp_path
    else:
        data_dir = Path(f'{tmp_path}')
    # call function (it returns the created DataFrame)
    df = make_metrics_file(str(data_dir), overwrite=True)

    csv_path = data_dir / "metrics.csv"
    # file should exist
    assert csv_path.exists()

    # reading back the CSV should give an empty dataframe with correct columns
    read_df = pd.read_csv(csv_path)
    assert list(read_df.columns) == COLS
    assert read_df.shape[0] == 0


def test_insert():
    log_metric(
        model='test_model',
        pipeline_notes='pipe_notes_none',
        hyperparam_notes='param_notes_note',
        roc_auc=0.0,
        accuracy=0.0,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        TN=0,
        TP=0,
        FN=0,
        FP=0
    )

    csv_path = DATA_DIR / "metrics.csv"
    # file should exist
    assert csv_path.exists()

    # reading back the CSV should give an empty dataframe with correct columns
    read_df = pd.read_csv(csv_path)
    assert list(read_df.columns) == COLS
    assert len(read_df) == 1
    


    row = read_df.iloc[0]

    # string fields
    assert row['model'] == 'test_model'
    assert row['pipeline_notes'] == 'pipe_notes_none'
    assert row['hyperparam_notes'] == 'param_notes_note'

    # float fields (cast to float to be robust to dtype)
    assert float(row['roc_auc']) == pytest.approx(0.0)
    assert float(row['accuracy']) == pytest.approx(0.0)
    assert float(row['precision']) == pytest.approx(0.0)
    assert float(row['recall']) == pytest.approx(0.0)
    assert float(row['f1']) == pytest.approx(0.0)

    # integer fields (cast to int)
    assert int(row['TN']) == 0
    assert int(row['TP']) == 0
    assert int(row['FP']) == 0
    assert int(row['FN']) == 0



    









# def run_tests():
#     test_make_metrics_file_creates_csv()
#     test_insert()


# if __name__ == "__main__":
#     run_tests()