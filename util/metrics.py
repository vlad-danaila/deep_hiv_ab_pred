import numpy as np
from deep_hiv_ab_pred.training.constants import MATTHEWS_CORRELATION_COEFFICIENT, ACCURACY, AUC
import mlflow
import sklearn.metrics
import sklearn as sk
from deep_hiv_ab_pred.util.tools import to_numpy
from deep_hiv_ab_pred.training.constants import LOSS, ACCURACY, MATTHEWS_CORRELATION_COEFFICIENT

def log_metrics_per_cv_antibody(cv_metrics, antibody):
    cv_metrics = np.array(cv_metrics)
    cv_mean_acc = cv_metrics[:, ACCURACY].mean()
    cv_std_acc = cv_metrics[:, ACCURACY].std()
    cv_mean_mcc = cv_metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].mean()
    cv_std_mcc = cv_metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].std()
    print('CV Mean Acc', cv_mean_acc, 'CV Std Acc', cv_std_acc)
    print('CV Mean MCC', cv_mean_mcc, 'CV Std MCC', cv_std_mcc)
    mlflow.log_metrics({
        f'cv mean acc {antibody}': cv_mean_acc,
        f'cv std acc {antibody}': cv_std_acc,
        f'cv mean mcc {antibody}': cv_mean_mcc,
        f'cv std mcc {antibody}': cv_std_mcc
    })
    return cv_mean_acc, cv_mean_mcc

def compute_metrics(ground_truth, pred, loss):
    metrics = np.zeros(4)
    pred = to_numpy(pred)
    ground_truth = to_numpy(ground_truth)
    metrics[AUC] += sk.metrics.roc_auc_score(ground_truth, pred)
    pred_bin = pred > .5
    metrics[LOSS] += loss.item()
    metrics[ACCURACY] += sk.metrics.accuracy_score(ground_truth, pred_bin)
    metrics[MATTHEWS_CORRELATION_COEFFICIENT] += sk.metrics.matthews_corrcoef(ground_truth, pred_bin)
    return metrics