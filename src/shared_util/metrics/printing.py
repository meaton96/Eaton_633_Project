from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, roc_auc_score
)

from shared_util.metrics import metrics_db


def print_metrics(y_true, 
                  y_pred,
                  y_proba,
                  metrics_db_id,
                  metrics_notes, 
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
        metrics_db.log_metric(
            metric_id=metrics_db_id,
            notes=metrics_notes,
            model=model or "",
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