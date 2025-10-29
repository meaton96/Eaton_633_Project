from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, roc_auc_score
)
import numpy as np

def summary_by_auc(df, plot_id):
    from IPython.display import display
    from shared_util.metrics.util import filter_duplicates_by_id

    df_collapse = filter_duplicates_by_id(df, plot_id)

    df_collapse = df_collapse.sort_values(by='roc_auc', ascending=False)

    df_collapse = df_collapse = df_collapse[df_collapse['data'] == 'test']

    df_collapse['roc_auc'] = np.round(df_collapse['roc_auc'], 3)

    print('Models by ROC_AUC')
    display(df_collapse.head(5)[['model', 'roc_auc']])

def summary_by_recall(df):
    from IPython.display import display
    df_collapse = df.sort_values(by='recall', ascending=False)
    df_collapse = df_collapse = df_collapse[df_collapse['data'] == 'test']
    df_collapse['recall'] = np.round(df_collapse['recall'], 3)

    print('Models by Recall')
    display(df_collapse.head(5)[['model', 'recall']])

def print_metrics(y_true, 
                  y_pred,
                  y_proba,
                  run_id,
                  metrics_notes,
                  data,
                  threshold_notes='base', 
                  log=True, 
                  model=None,
                  pipeline_notes=None,
                  hyperparam_notes=None,
                  ):
    cnf = confusion_matrix(y_true, y_pred).ravel()
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    tn, fp, fn, tp = cnf[0], cnf[1], cnf[2], cnf[3]
    _roc_auc=float(roc_auc_score(y_true, y_proba))


    print(f"ROC_AUC:     {_roc_auc}")
    print("Accuracy:     ", acc)
    print("F1 score:     ", f1)
    print("Precision:    ", p)
    print("Recall:       ", r)
    print("\nConfusion matrix:\n")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print("\nClassification report:\n", classification_report(y_true, y_pred))

    if log:
        from shared_util.log import log_metric
        log_metric(
            run_id=run_id,
            notes=metrics_notes,
            model=model or "",
            data=data,
            threshold_notes=threshold_notes,
            hyperparam_notes=hyperparam_notes or "",
            pipeline_notes=pipeline_notes or "",
            accuracy=float(acc),
            f1=float(f1),
            precision=float(p),
            recall=float(r),
            TN=tn,
            TP=tp,
            FN=fn,
            FP=fp,
            roc_auc=_roc_auc
            )