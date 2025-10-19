from sklearn.metrics import make_scorer, roc_auc_score
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

def make_cv():
    return StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
    )

def get_baseline_score(
        model,
        sampling_tech,
        X_train,
        y_train,
        scorer=roc_auc_score,
        scaler=None
        ) -> float:
    """
    Calculates a baseline score using the designated scorer function, default roc_auc

    Creates a 10 fold CV and uses cross_val_score to calculate a score for each fold.

    Returns the mean score between folds
    """
    cv = make_cv()

    if scaler != None:
       pipe = Pipeline([
        ('sampler', sampling_tech),
        ('scaler', scaler),
        ('model', model)
    ]) 
    else:
        pipe = Pipeline([
        ('sampler', sampling_tech),
        ('model', model)
    ]) 

    

    scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv = cv,
        n_jobs=-1,
        scoring=make_scorer(scorer)
    )

    print(f'{scorer.__name__} score per fold: ', scores)
    print(f'mean {scorer.__name__} score: ', scores.mean())
    return scores.mean()



