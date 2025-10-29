from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

DB_URI = os.getenv('DB_URI')



if not DB_URI:
    raise ValueError("No DB_URI found in environment variables (DB_URI).")

engine = create_engine(DB_URI, pool_pre_ping=True, pool_recycle=1800)


def fetch_metric(run_id: int) -> pd.DataFrame:
    """
    Retrieve all metric records for a given run identifier.

    Parameters
    ----------
    run_id: int
        Unique identifier used when logging metrics (metrics.run_id).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing any matching rows ordered by newest first. An empty
        DataFrame is returned when no metrics exist for the requested run.
    """
    query = text(
        """
        SELECT
            id,
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
            tn,
            tp,
            fp,
            fn,
            created_at
        FROM metrics
        WHERE run_id = :run_id
        ORDER BY created_at DESC
        """
    )

    return _connect_and_get(query, run_id=run_id).sort_values(by='roc_auc', ascending=False)

def fetch_all_metrics() -> pd.DataFrame:
    query = text(
        """
        SELECT
            id,
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
            tn,
            tp,
            fp,
            fn,
            created_at
        FROM metrics
        ORDER BY created_at DESC
        """
    )

    return _connect_and_get(query).sort_values(by='roc_auc', ascending=False)

def fetch_all_params() -> pd.DataFrame:
    query=text("""
    SELECT
    model,
    hyperparam_text
    FROM hyperparameters
    ORDER BY created_at DESC
    """)

    return _connect_and_get(query)

def fetch_params_by_id(run_id:int) -> pd.DataFrame:
    query=text("""
    SELECT
    model,
    hyperparam_text
    FROM hyperparameters
    WHERE run_id = :run_id
    ORDER BY created_at DESC
    """)

    return _connect_and_get(query, run_id=run_id)
    
    
def _connect_and_get(query, run_id:int = -1):

    with engine.connect() as conn:
        if run_id == -1:
            result = conn.execute(query)
        else:
            result = conn.execute(query, ({"run_id": run_id}))
        rows = result.fetchall()
        columns = result.keys()

    if not rows:
        return pd.DataFrame()
    
    return pd.DataFrame(rows, columns=columns) #type: ignore 