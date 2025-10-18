from sklearn.model_selection import StratifiedKFold, cross_val_score



def make_cv():
    return StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
    )