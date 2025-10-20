# split initial train, validate, test
import pandas as pd
from sklearn.model_selection import train_test_split

def train_val_split(df, target_col = 'target', test_size = 0.15, val_size = 0.15):
    X_train, X_, y_train, y_ = train_test_split(
        df.drop(columns=['target']), 
        df['target'], 
        test_size=(test_size + val_size), 
        stratify=df['target']
        )
    
    X_validate, X_test, y_validate, y_test = train_test_split(X_, y_, test_size=(test_size + val_size) / test_size, stratify=y_)

