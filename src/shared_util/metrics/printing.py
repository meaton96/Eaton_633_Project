from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

#from shared_util.metrics import metrics_db




def plot_y_dist(y, ax=None, title='Target Variable Distribution'):
    ax = ax or plt.gca()
    sns.countplot(x=y, hue=y, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Readmitted')
    ax.set_ylabel('count')

def check_y_dist(y_train, y_validate, y_test):
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    plot_y_dist(y_train, ax=axes[0], title='Train')
    plot_y_dist(y_validate, ax=axes[1], title='Validate')
    plot_y_dist(y_test, ax=axes[2], title='Test')

    plt.tight_layout()
    plt.show()

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
        from shared_util.metrics.log import log_metric
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