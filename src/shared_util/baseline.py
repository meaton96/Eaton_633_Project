from sklearn.metrics import make_scorer, roc_auc_score
from imblearn.pipeline import Pipeline
from shared_util.utils import make_cv
from sklearn.model_selection import cross_val_score



def get_baseline_score(
        model,
        sampling_tech,
        X_train,
        y_train,
        scorer=roc_auc_score
        ) -> float:
    """
    Calculates a baseline score using the designated scorer function, default roc_auc

    Creates a 10 fold CV and uses cross_val_score to calculate a score for each fold.

    Returns the mean score between folds
    """
    cv = make_cv()
    pipe = Pipeline([
        ('sampler', sampling_tech),
        ('model', model)
    ])

    scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv = cv,
        scoring=make_scorer(scorer)
    )

    print(f'{scorer} score per fold: ', scores)
    print(f'mean {scorer} score: ', scores.mean())
    return scores.mean()



