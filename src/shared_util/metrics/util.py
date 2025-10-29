import pandas as pd
def filter_duplicates_by_id(df, keep_id):
    
    key_cols = ['model', 'data', 'threshold_notes', 'pipeline_notes']
    
    
    duplicated_mask = df.duplicated(subset=key_cols, keep=False)

    duplicates = df[duplicated_mask]
    uniques = df[~duplicated_mask]
    
    kept_duplicates = duplicates[duplicates['id'] == keep_id]
    
    result = pd.concat([uniques, kept_duplicates], ignore_index=True)
    
    return result.sort_values(by='roc_auc', ascending=False).reset_index(drop=True)