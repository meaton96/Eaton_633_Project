from sklearn.model_selection import train_test_split

def train_val_test_split(df, target_col = 'target', train_size = 0.7):
    X_train, X_, y_train, y_ = train_test_split(
        df.drop(columns=[target_col]), 
        df[target_col], 
        test_size=1-train_size, 
        stratify=df[target_col]
        )
    
    X_validate, X_test, y_validate, y_test = train_test_split(X_, y_, test_size=0.5, stratify=y_)

    return X_train, X_validate, X_test, y_train, y_validate, y_test

