from sklearn.metrics import make_scorer, roc_auc_score
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
from shared_util.pipeline import CleaningPipeline

def make_group_cv():
    return StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
    )


def _build_pipe(model, sampling_tech, preprocessor=None, scaler=None):
    if preprocessor is None:
        preprocessor = CleaningPipeline(create_interactions=False)

    steps = [('preprocessor', preprocessor),
             ('sampler', sampling_tech)]

    if scaler is not None:
        steps.append(('scaler', scaler))

    steps.append(('model', model))
    pipe = Pipeline(steps)
    return pipe

def get_baseline_score(
        model,
        sampling_tech,
        X_train,
        y_train,
        scorer='roc_auc',
        scaler=None,
        preprocessor=None,
        patient_col='patient_nbr'
        ) -> float:
    """
    Calculates a baseline score using the designated scorer function, default roc_auc

    Creates a 5 fold CV and uses cross_val_score to calculate a score for each fold.

    Returns the mean score between folds
    """
    if preprocessor is None:
        preprocessor = CleaningPipeline(create_interactions=False)


    cv = make_group_cv()

    pipe = _build_pipe(model, sampling_tech, preprocessor, scaler)
    

    scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        groups=X_train[patient_col],
        cv = cv,
        n_jobs=-1,
        scoring=scorer,
        error_score="raise"
    )

    print(f'{scorer} score per fold: ', scores)
    print(f'mean {scorer} score: ', scores.mean())
    return scores.mean()



