import numpy as np
import pandas as pd
from typing import Dict, List, Any
import time
MODELS = ['rf', 'svc', 'sgd', 'xgb', 'nb', 'knn', 'lr', 'mlp']

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

        Baseline is tested in 5 fold cross validation
        
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
       # one_hot_ordinal: bool = True

        def __init__(self, 
                     create_interactions=True, 
                     log_transform_cols=None,
                     one_hot_ordinal=True
                     ) -> None:
            self.create_interactions = create_interactions
            self.log_transform_cols = log_transform_cols
           # self.one_hot_ordinal = one_hot_ordinal
            
            
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
    resample: bool = True
        


        

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
               #  one_hot_ordinal: bool = True,
                 resample: bool = True
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
        self.resample = resample
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
            from shared_util.metrics.plotting import check_y_dist
            check_y_dist(
                self.y_train,
                self.y_validate,
                self.y_test
            )

        # Freeze preprocessing options inside a helper so we can rebuild identical cleaners on demand.
        self.preprocessor_settings = self.PreprocessorSettings(
            create_interactions=create_interactions,
            log_transform_cols=log_transform_cols,
           # one_hot_ordinal=one_hot_ordinal
        )

        
        self.model = model

        # freeze feature schema on full train for later use in permutation importance
        _clean_pipe = self.preprocessor_settings.factory()
        _clean_pipe.fit(self.X_train)
        self._feature_schema = _clean_pipe.get_feature_names_out()

    def permute_importance(self, print_num=15, plot=True,show_duration_info:bool = True,):
        from sklearn.inspection import permutation_importance
        t0 = time.perf_counter()

        print(f"----------Running Feature Importance for model: {self.model}----------")

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
            name_to_idx = {name: i for i, name in enumerate(self._feature_schema)}
            for tr_idx, va_idx in cv.split(X, y, groups):
                X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]



                # Rebuild a fresh pipeline each iteration to avoid cross-fold leakage.
                pipe = pipe_factory(self.model)
                pipe.fit(X_tr, y_tr)

                # Transform validation through inference-capable steps
                Xt_va = _transform_through_inference_steps(pipe, X_va, model_step=model_step)
                preproc = pipe.named_steps.get(prep_step)
                feat_names_fold = np.asarray(preproc.get_feature_names_out(), dtype=object)


                # Xt_va = pd.DataFrame(Xt_va, columns=feat_names_fold)

                # # reindex to the global schema: add missing cols = 0, drop extras
                # Xt_va = Xt_va.reindex(columns=self._feature_schema, fill_value=0)
                # feat_names = np.asarray(self._feature_schema, dtype=object)

                model = pipe.named_steps[model_step]

                # Measure how sensitive the fitted model is to each feature on this fold.
                r = permutation_importance(
                    model, Xt_va, y_va,
                    scoring=scoring, n_repeats=n_repeats,
                    n_jobs=n_jobs, random_state=random_state
                )


                fold_importance = np.zeros(len(self._feature_schema))
                for j, name in enumerate(feat_names_fold):
                    idx = name_to_idx.get(name)
                    if idx is not None:
                        fold_importance[idx] = r.importances_mean[j] #type: ignore

                perm_means.append(fold_importance)

               # perm_means.append(r.importances_mean) #type: ignore

            arr = np.vstack(perm_means)
            df = pd.DataFrame({
                "feature": self._feature_schema,
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
         use_f2_threshold=False,
         use_cost_threshold=True,
         R: float = 16300.0,
         e: float = 0.33,
         c_m_list=(500.0, 1000.0, 1500.0),
         prefer_c_m: float | None = 330.0,
         show_plot=True,
         show_duration_info: bool = False,
         use_local_model=False,
         **kwargs):
        """
        New: use_cost_threshold selects threshold maximizing $ savings on the validate set.
        - Overlays: PR curve with break-even PPV lines, and savings vs threshold per c_m.
        - If both use_f2_threshold and use_cost_threshold are True, cost wins. Because money.
        """
        if (not self.validated or not self.tuned) and not use_local_model:
            raise EnvironmentError("Must validate before test")

        import time
        import numpy as np
        from shared_util.metrics.scoring import (
            get_scores,
            cost_curves_from_scores, best_threshold_by_cost
        )
        from shared_util.metrics.plotting import plot_pr_with_break_even, plot_savings_vs_threshold
        from shared_util.metrics.printing import print_metrics

        # Fit or load model
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

        # 1) Scores on validate for threshold selection
        _val_scores = get_scores(_model, self.X_validate)

        threshold_notes = 'baseline'
        chosen_thresh = None
        est_pp_savings = 0.0

        # Build cost curves on validate
        curves = cost_curves_from_scores(
            y_true=self.y_validate,
            y_scores=_val_scores,
            R=R, e=e, c_m_list=c_m_list
        )
        choice = best_threshold_by_cost(curves, prefer_c_m=prefer_c_m)

        chosen_thresh = choice["threshold"]
        threshold_notes = f'cost_opt_R={int(R)}_e={e:.2f}_c={("avg" if choice["chosen_c_m"] is None else int(choice["chosen_c_m"]))}'

        if choice["chosen_c_m"] is None:
            # maximizing average across c_m_list
            est_pp_savings = float(np.mean(list(choice["savings_at_c"].values())))
        else:
            # using a specific c_m
            est_pp_savings = float(choice["savings_at_c"][choice["chosen_c_m"]])
        
        # Optional overlays
        if show_plot:
            title = f"PR with break-even PPV lines (R=${int(R)}, e={e:.2f})"
            plot_pr_with_break_even(curves, R=R, e=e, c_m_list=c_m_list, title=title)

            title2 = f"Savings vs Threshold on validate (per 1000 patients)"
            plot_savings_vs_threshold(curves, c_m_list=c_m_list, per_1000=True, title=title2)


        print(
            f"[COST] Chosen threshold: {chosen_thresh:.4f} | "
            f"PPV={choice['precision']:.3f}  Recall={choice['recall']:.3f}  "
            f"FlagRate={choice['flag_rate']:.3f}  "
            f"Savings@{('avg' if choice['chosen_c_m'] is None else '$'+str(int(choice['chosen_c_m'])))}="
            f"{est_pp_savings:.2f} per patient"
        )

        

        # 2) Apply chosen threshold to test
        _test_scores = get_scores(_model, self.X_test)
        if chosen_thresh is not None:
            y_pred = (_test_scores >= chosen_thresh).astype(int)
        else:
            y_pred = _model.predict(self.X_test)  # type: ignore

        # 3) Print/log test metrics
        from shared_util.metrics.printing import print_metrics
        print_metrics(
            y_true=self.y_test,
            y_pred=y_pred,
            y_proba=_test_scores,
            run_id=self.METRICS_DB_ID,
            metrics_notes=self.METRIC_NOTES,
            data='test',
            threshold_notes=threshold_notes,
            model=self.model,
            hyperparam_notes='tuned_best_roc_auc',
            pipeline_notes='under_sample',
            est_pp_savings=est_pp_savings,
            log=self.LOG
        )

        if show_duration_info:
            self._print_dur(t0)

    def run_full_suite(self,
                        run_baseline: bool = True,
                        run_hyperparam_search: bool = True,
                        baseline_undersample: bool = True,
                        baseline_smote: bool = True,
                        feature_importance: bool = True,
                        show_duration_info:bool = True,
                        feat_num: int = 15,
                        hyperparam_dist: Dict[str, Any] = {},
                        n_iter: int = 30,
                        n_jobs: int = -1,
                        refit: bool = True,
                        verbose: int = 1,
                        prefer_c_m: float | None = 330.0,
                        R: float = 16300.0,
                        e: float = 0.33,
                        c_m_list=(500.0, 1000.0, 1500.0),
                        **kwargs
                       ):
        
        if run_baseline:
            self.run_baseline(
                run_undersampler=baseline_undersample,
                run_smote=baseline_smote,
                show_duration_info=show_duration_info
            )
        
        if feature_importance:
            self.permute_importance(
                print_num=feat_num,
                plot=True,
                show_duration_info=show_duration_info
            )

        if run_hyperparam_search:
            self.run_hyperparam_search(
                hyperparam_dist=hyperparam_dist,
                n_iter=n_iter,
                n_jobs=n_jobs,
                refit=refit,
                show_duration_info=show_duration_info,
                verbose=verbose
            )
        self.validate(
            use_local_model=not run_hyperparam_search,
            show_duration_info=show_duration_info,
            **kwargs
        )

        self.test(
            prefer_c_m=prefer_c_m,
            R=R,
            e=e,
            c_m_list=c_m_list,
            use_local_model=not run_hyperparam_search,
            show_duration_info=show_duration_info,
            **kwargs
        )


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
        print(f"----------Running hyperparameter search for model: {self.model}----------")
        # Build a full pipeline so tuning evaluates preprocessing, sampling, and the estimator together.
        rs = RandomizedSearchCV(
            estimator=self._pipe_factory(self.model, random_state=self.random_state),
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

        

        def _run_model(pipeline_notes, sampler):
            # Local helper so each sampler shares the same preprocessing + logging logic.
            from shared_util.baseline import get_baseline_score

            # Build a single estimator that each sampling strategy will pair with.
            _pipe = self._pipe_factory(self.model, self.random_state, sampler=sampler)
        
            _roc_auc = get_baseline_score(
                pipe=_pipe,
                X_train=self.X_train,
                y_train=self.y_train,
                cv=self._make_group_cv()
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
            print('--------SMOTE Baseline--------')
            _run_model('smote', 'smote')        

        if run_undersampler:
            print('-----Undersample Baseline-----')
            _run_model('under_sample', 'under')
    

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
    
    def _model_factory(self, model:str,random_state=42, **kwargs):
        
        match model:
            case 'mlp':
                from sklearn.neural_network import MLPClassifier
                return MLPClassifier(random_state=random_state, **kwargs)
            case 'lr':
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(random_state=random_state, **kwargs)
            case 'nb':
                from sklearn.naive_bayes import GaussianNB
                return GaussianNB(**kwargs)
            case 'knn':
                from sklearn.neighbors import KNeighborsClassifier
                return KNeighborsClassifier(**kwargs)
            case 'rf':
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(random_state=random_state, **kwargs)
            case 'sgd':
                from sklearn.linear_model import SGDClassifier
                return SGDClassifier(random_state=random_state, **kwargs)
            case 'svc':
                from sklearn.svm import SVC
                return SVC(kernel='rbf', random_state=random_state, **kwargs)
            case 'xgb':
                from xgboost import XGBClassifier
                return XGBClassifier(objective='binary:logistic',
                                     tree_method='hist', 
                                     eval_metric='aucpr',
                                     device='cuda',
                                     random_state=random_state,
                                     **kwargs)
            case _:
                raise ValueError(f"Unknown model '{model}'")
            
    def _pipe_factory(self, model:str, random_state=42, sampler='under', **kwargs):
        from imblearn.pipeline import Pipeline
        from typing import Tuple

        _pre = self.preprocessor_settings.factory()
        steps:List[Tuple[str, Any]] = [("preprocessor", _pre)]

        if self.resample:
            if sampler == 'under':
                from imblearn.under_sampling import RandomUnderSampler
                _sampler = RandomUnderSampler(random_state=random_state)

            elif sampler == 'smote':
                from imblearn.over_sampling import SMOTE
                _sampler = SMOTE(random_state=random_state)

            steps.append(("sampler", _sampler)) #type: ignore
        

        if self.scaler is not None:
            steps.append(("scaler", self.scaler))

        if self.use_pca:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=0.95)
            steps.append(('PCA', pca))

        _model = self._model_factory(model=model, random_state=random_state, **kwargs)

        steps.append(("model", _model))

        return Pipeline(steps=steps)
