import numpy as np
import pandas as pd
from typing import Dict, List, Any
import time
# High-level helpers for configuring preprocessing and running model baselines.


MODELS = ['rf', 'svc', 'sgd', 'xgb']

class ModelPipeline:
    """
    Represents a pipeline to test a model

    Contains setup in a constructor including reading data in and splitting test,val,train sets

    Contains class methods for the following:
        Perumation Importance of Features + graphing
        Baseline testing SMOTE vs Random Undersampling
        Hyperparameter sweeping with desired parameter distribution
        Validation using best estimator from sweep, with optional f2 score calculation
        Test using held out test data using either base threshold of f2 weighted score

        Printing metrics of each step ROC_AUC, prec, acc, recall, confusion matrics, classification report

        Baseline is tested in 10 fold cross validation
        
        Hyperparam tuning is done using RandomizedSearch with 5 folds

        All data is grouped by patient_nbr to handle duplicate patient encounters
    """

    

    class PreprocessorSettings:
        """
        Holds information about preprocessor settings for this pipeline

        Contains factory() to create a new preprocesser with the set settings as needed
        """
        create_interactions: bool = True
        log_transform_cols: np.ndarray | None

        def __init__(self, create_interactions=True, log_transform_cols=None) -> None:
            self.create_interactions = create_interactions
            self.log_transform_cols = log_transform_cols
            
            
        def factory(self):
            from shared_util.pipelines.cleaning import CleaningPipeline
            return CleaningPipeline(
                create_interactions=self.create_interactions,
                log_transform_cols=self.log_transform_cols
                )


    LOG: bool = False
    METRICS_DB_ID: int = 0
    METRIC_NOTES: str = "None"
    data_path: str = "cleaned"
    X_train: Any = None
    X_validate: Any | None = None
    X_test: Any | None = None
    y_train: Any | None = None
    y_validate: Any | None = None
    y_test: Any | None = None
    preprocessor_settings: PreprocessorSettings
    model:str = ""
    random_state: int = 42
    scaler: Any | None = None
    hyperparam_dist: Dict[str, Any]
    tuned: bool = False
    validated: bool = False
    param_scoring:str='roc_auc'
    use_pca: bool = False

        


        

    def __init__(self,
                 *
                 , 
                 log: bool = False, 
                 metrics_db_id: int = 0, 
                 metrics_notes: str = "None", 
                 data_path: str = "cleaned",
                 plot_y_dist: bool = True,
                 random_state: int = 42,
                 create_interactions: bool = True,
                 log_transform_cols: np.ndarray | None = None,
                 model: str = 'rf',
                 scaler: str | None = None, # standard, power
                 group_by_col='patient_nbr',
                 use_pca: bool = False,
                 ) -> None:
        """
        Create a new pipeline
        """
        # Persist runtime configuration so we can trace every experiment.
        self.LOG = log
        self.METRIC_NOTES = metrics_notes
        self.data_path = data_path
        self.METRICS_DB_ID = metrics_db_id 
        self.random_state = random_state
        self.group_by_col = group_by_col
        self.use_pca = use_pca

        # Lazily attach a scaler because only some models benefit from it.
        if scaler == 'standard':
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
        if scaler == 'power':
            from sklearn.preprocessing import PowerTransformer
            self.scaler = PowerTransformer(method='yeo-johnson', standardize=True)

        # Load the requested dataset and perform the grouped train/validate/test split.
        from shared_util.dataio import load_csv
        from shared_util.split import group_split
        _df = load_csv(data_path)

        self.X_train, self.X_validate, self.X_test, self.y_train, self.y_validate, self.y_test = group_split(_df)


        if plot_y_dist:
            # Quick sanity check: make sure the target distribution is stable across splits.
            from shared_util.metrics.printing import check_y_dist
            check_y_dist(
                self.y_train,
                self.y_validate,
                self.y_test
            )

        # Freeze preprocessing options inside a helper so we can rebuild identical cleaners on demand.
        self.preprocessor_settings = self.PreprocessorSettings(
            create_interactions=create_interactions,
            log_transform_cols=log_transform_cols
        )

        
        self.model = model

    def permute_importance(self, print_num=15, plot=True,show_duration_info:bool = True,):
        from sklearn.inspection import permutation_importance
        t0 = time.perf_counter()

        def plot_feature_importance(df, col1, col2, size, title):
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,6))
            plt.barh(df[col1][:size][::-1], df[col2][:size][::-1])
            plt.title(title)
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.show()

        def _transform_through_inference_steps(pipe, X, model_step='model'):
            """
            Apply only transform-capable steps in a fitted imblearn Pipeline.
            Skip any sampler (fit_resample-only). Stop before the final estimator.
            """
            Xt = X
            for name, step in pipe.steps:
                if name == model_step:
                    break
                # Skip samplers (no transform)
                if hasattr(step, "fit_resample"):
                    continue
                if hasattr(step, "transform"):
                    Xt = step.transform(Xt)
                else:
                    # Non-sampler but also no transform? treat as passthrough
                    pass
            return Xt
        

        def cv_permutation_importance(
            pipe_factory, X, y, cv, groups=None,
            scoring="roc_auc", n_repeats=10, n_jobs=-1, random_state=42,
            model_step='model', prep_step='preprocessor'
        ):
            # Compute permutation importance fold by fold so samplers respect the group structure.
            perm_means = []
            feat_names = None

            for tr_idx, va_idx in cv.split(X, y, groups):
                X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

                # Rebuild a fresh pipeline each iteration to avoid cross-fold leakage.
                pipe = pipe_factory(self.model)
                pipe.fit(X_tr, y_tr)

                # Transform validation through inference-capable steps
                Xt_va = _transform_through_inference_steps(pipe, X_va, model_step=model_step)

                # Get names from the preprocessor if available; otherwise fallback
                preproc = pipe.named_steps.get(prep_step)
                if hasattr(preproc, "get_feature_names_out"):
                    feat_names_fold = np.asarray(preproc.get_feature_names_out(), dtype=object)
                else:
                    feat_names_fold = np.array([f"x{i}" for i in range(Xt_va.shape[1])], dtype=object)

                if feat_names is None:
                    feat_names = feat_names_fold
                else:
                    if len(feat_names) != len(feat_names_fold):
                        raise RuntimeError("Transformed feature count changed across folds; pipeline not schema-stable.")

                model = pipe.named_steps[model_step]

                # Measure how sensitive the fitted model is to each feature on this fold.
                r = permutation_importance(
                    model, Xt_va, y_va,
                    scoring=scoring, n_repeats=n_repeats,
                    n_jobs=n_jobs, random_state=random_state
                )
                perm_means.append(r.importances_mean) #type: ignore

            arr = np.vstack(perm_means)
            df = pd.DataFrame({
                "feature": feat_names,
                "importance_mean": arr.mean(axis=0),
                "importance_std": arr.std(axis=0)
            }).sort_values("importance_mean", ascending=False)
            return df
        
        # Aggregate the per-fold importances into a single ranked table.
        _perm_imp = cv_permutation_importance(
            pipe_factory=self._pipe_factory,
            X=self.X_train,
            y=self.y_train,
            cv=self._make_group_cv(),
            groups=self.X_train[self.group_by_col]
        )

        from IPython.display import display

        display(_perm_imp.head(print_num))

        if plot:
            plot_feature_importance(
                _perm_imp, 
                'feature', 
                'importance_mean', 
                print_num, 
                f'Permutation Importance ({self.model})')
            
        if show_duration_info:
            self._print_dur(t0)

    def test(self, 
             use_f2_threshold=True, 
             show_plot=True,
             show_duration_info:bool = False,
             use_local_model=False,
             **kwargs
             ):
        
        
        if (not self.validated or not self.tuned) and not use_local_model:
            raise EnvironmentError("Must validate before test")
        
        

        from shared_util.metrics.scoring import get_scores, best_threshold_by_fbeta, plot_scoring
        from shared_util.metrics.printing import print_metrics


        if use_local_model:
            _model = self._pipe_factory(
                model=self.model,
                random_state=self.random_state,
                sampler='under',
                **kwargs
                )
            _model.fit(self.X_train, self.y_train)
        else:
            _model = self.randomized_search.best_estimator_
        t0 = time.perf_counter()
        # Score the validation split so we can choose a threshold off the ROC curve.
        _val_scores = get_scores(_model, self.X_validate)

        if use_f2_threshold:
            # Treat recall as more important than precision by default (F2).
            info = best_threshold_by_fbeta(self.y_validate, _val_scores)

            best_thresh = info['threshold']

            print(f"Best threshold (F2): {best_thresh:.4f}")
            print(f"Precision: {info['precision']:.3f}  Recall: {info['recall']:.3f}  F2: {info['fbeta']:.3f}")

            if show_plot:
                plot_scoring(info['curve'])
        

            _test_scores = get_scores(_model, self.X_test)
            
            y_pred = (_test_scores >= best_thresh).astype(int)
        else:
            # Fall back to the estimator's native threshold if we skip F2 tuning.
            _test_scores = get_scores(_model, self.X_test)
            y_pred = _model.predict(self.X_test) #type: ignore

        # Reuse the same metric printer that logs as needed.
        print_metrics(
            y_true=self.y_test,
            y_pred=y_pred,
            y_proba=_test_scores,
            run_id=self.METRICS_DB_ID,
            metrics_notes=self.METRIC_NOTES,
            data='test',
            threshold_notes='f2_weighted' if use_f2_threshold else 'baseline',
            model=self.model,
            hyperparam_notes='tuned_best_roc_auc',
            pipeline_notes='under_sample',
            log=self.LOG
        )
        if show_duration_info:
            self._print_dur(t0)

    def validate(self, 
                 show_duration_info:bool = False,
                 use_local_model=False,
                 **kwargs
                 ):
        if not self.tuned and not use_local_model:
            raise EnvironmentError("Must run hyperparam search first")
            
        from shared_util.metrics.scoring import get_scores
        t0 = time.perf_counter()
        
        if use_local_model:
            _model = self._pipe_factory(
                model=self.model,
                random_state=self.random_state,
                sampler='under',
                **kwargs
                )
            _model.fit(self.X_train, self.y_train)
        else:
            _model = self.randomized_search.best_estimator_


        # Pull calibrated probabilities and convert them to hard labels via the tuned estimator.
        y_proba = get_scores(_model, self.X_validate)
        y_pred = _model.predict(self.X_validate) #type: ignore

        print(f"Held out validation metrics: {(self.model)}")

        from shared_util.metrics.printing import print_metrics

        print_metrics(
            y_true=self.y_validate,
            y_pred=y_pred,
            y_proba=y_proba,
            metrics_notes=self.METRIC_NOTES,
            run_id=self.METRICS_DB_ID,
            model=self.model,
            data='validate',
            hyperparam_notes=f'tuned best {(self.param_scoring)}',
            pipeline_notes='under_sample',
            log=self.LOG,
            
        )

        self.validated = True

        if show_duration_info:
            self._print_dur(t0)

    def run_hyperparam_search(self,
                 hyperparam_dist = {},
                 param_scoring='roc_auc',
                 n_iter = 30,
                 n_jobs = -1,
                 refit=True,
                 show_duration_info=True,
                 verbose=1
                              ):
        t0 = time.perf_counter()
        from sklearn.model_selection import RandomizedSearchCV

        # Build a full pipeline so tuning evaluates preprocessing, sampling, and the estimator together.
        rs = RandomizedSearchCV(
            estimator=self._pipe_factory(self.model),
            param_distributions=hyperparam_dist,
            n_iter=n_iter,
            refit=refit,
            random_state=self.random_state,
            n_jobs=n_jobs,
            scoring=param_scoring,
            cv=self._make_group_cv(),
            verbose=verbose
        )

        # Group-aware fit ensures we never leak a patient's encounters across folds.
        rs.fit(
            self.X_train,
            self.y_train,
            groups=self.X_train[self.group_by_col]
        )


        print("Best ROC_AUC (cv):", rs.best_score_)
        print("Best params:", rs.best_params_)

        # Summarize top results
        res_df = pd.DataFrame(rs.cv_results_).sort_values('rank_test_score').reset_index(drop=True)
        res_df[['rank_test_score','mean_test_score','std_test_score','params']].head(10)

        # Persist tuning artifacts so downstream steps (validate/test) can reuse the best estimator.
        self.randomized_search = rs
        self.rs_df = res_df
        self.param_scoring = param_scoring
        self.tuned = True

        if self.LOG:
            from shared_util.log import log_hyperparameters
            log_hyperparameters(
                run_id=self.METRICS_DB_ID,
                model=self.model,
                hyperparam_text=str(rs.best_params_)
            )

        
        if show_duration_info:
            self._print_dur(t0)

    def run_baseline(self,
                 run_undersampler: bool = True,
                 run_smote: bool = True,
                 show_duration_info:bool = True,
                 ):
        
        t0 = time.perf_counter()

        # Build a single estimator that each sampling strategy will pair with.
        _model = self._model_factory(
            model=self.model,
            random_state=self.random_state
        )

        def _run_model(pipeline_notes, sampler):
            # Local helper so each sampler shares the same preprocessing + logging logic.
            from shared_util.baseline import get_baseline_score

        
            _roc_auc = get_baseline_score(
                _model,
                sampler,
                X_train=self.X_train,
                y_train=self.y_train,
                scaler=self.scaler,
                preprocessor=self.preprocessor_settings.factory()
            )

            if self.LOG:
                from shared_util.log import log_metric
                log_metric(
                    run_id=self.METRICS_DB_ID,
                    model=self.model,
                    threshold_notes='base',
                    data='train',
                    pipeline_notes=pipeline_notes,
                    notes=self.METRIC_NOTES,
                    roc_auc=_roc_auc
                )

        if run_smote:
            from imblearn.over_sampling import SMOTE
            print('--------SMOTE Baseline--------')
            # Compare against an oversampling strategy that synthesizes minority examples.
            _run_model('smote', SMOTE(random_state=self.random_state))        

        if run_undersampler:
            from imblearn.under_sampling import RandomUnderSampler
            print('-----Undersample Baseline-----')
            # Also measure performance when we downsample the majority class.
            _run_model('under_sample', RandomUnderSampler(random_state=self.random_state))
    

        if show_duration_info:
            self._print_dur(t0)
        
    def _make_group_cv(self, n_splits=5):
        from sklearn.model_selection import StratifiedGroupKFold
        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state
        )
    
    def _print_dur(self, t0 = 0.0):

        def _fmt_time(dur):
            total_seconds_int = int(dur)

            minutes, seconds = divmod(total_seconds_int, 60)
            hours, minutes = divmod(minutes, 60)

            fractional_seconds = dur - total_seconds_int
            seconds += fractional_seconds

            return f'{hours} hrs, {minutes} mins, {seconds:.1f} secs'

        

        print(f'Duration: {_fmt_time(time.perf_counter() - t0)}')
    
    def _model_factory(self, model:str, **kwargs):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import SGDClassifier
        from sklearn.svm import SVC
        from xgboost import XGBClassifier

        match model:
            case 'rf':
                return RandomForestClassifier(**kwargs)
            case 'sgd':
                return SGDClassifier(**kwargs)
            case 'svc':
                return SVC(kernel='rbf', **kwargs)
            case 'xgb':
                return XGBClassifier(objective='binary:logistic',
                                     tree_method='hist', 
                                     eval_metric='auc',
                                     device='cuda',
                                     **kwargs)
            case _:
                raise ValueError(f"Unknown model '{model}'")
            
    def _pipe_factory(self, model:str, random_state=42, sampler='under', **kwargs):
        from imblearn.pipeline import Pipeline

        # Choose the sampling strategy while keeping the interface consistent.
        if sampler == 'under':
            from imblearn.under_sampling import RandomUnderSampler
            _sampler = RandomUnderSampler(random_state=random_state)

        elif sampler == 'smote':
            from imblearn.over_sampling import SMOTE
            _sampler = SMOTE(random_state=random_state)

        _model = self._model_factory(model=model, **kwargs)
        _pre = self.preprocessor_settings.factory()

        steps = [("preprocessor",_pre), ("sampler", _sampler)]

        if self.scaler is not None:
            steps.append(("scaler", self.scaler))

        if self.use_pca:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=0.95)
            steps.append(('PCA', pca))

        steps.append(("model", _model))

        return Pipeline(steps=steps)
