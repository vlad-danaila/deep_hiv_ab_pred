import numpy as np
from deep_hiv_ab_pred.training.constants import MATTHEWS_CORRELATION_COEFFICIENT, ACCURACY, AUC
import mlflow
import sklearn.metrics
import sklearn as sk
from deep_hiv_ab_pred.training.constants import ACCURACY, MATTHEWS_CORRELATION_COEFFICIENT
import logging
import statistics

def log_metrics_per_cv_antibody(cv_metrics, antibody):
    cv_metrics = np.array(cv_metrics)
    cv_mean_acc = cv_metrics[:, ACCURACY].mean()
    cv_std_acc = cv_metrics[:, ACCURACY].std()
    cv_mean_mcc = cv_metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].mean()
    cv_std_mcc = cv_metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].std()
    cv_mean_auc = cv_metrics[:, AUC].mean()
    cv_std_auc = cv_metrics[:, AUC].std()
    logging.info(f'CV Mean Acc {cv_mean_acc} CV Std Acc {cv_std_acc}')
    logging.info(f'CV Mean MCC {cv_mean_mcc} CV Std MCC {cv_std_mcc}')
    logging.info(f'CV Mean AUC {cv_mean_auc} CV Std AUC {cv_std_auc}')
    mlflow.log_metrics({
        f'cv mean acc {antibody}': cv_mean_acc,
        f'cv std acc {antibody}': cv_std_acc,
        f'cv mean mcc {antibody}': cv_mean_mcc,
        f'cv std mcc {antibody}': cv_std_mcc,
        f'cv mean auc {antibody}': cv_mean_auc,
        f'cv std auc {antibody}': cv_std_auc
    })
    return cv_mean_acc, cv_mean_mcc, cv_mean_auc

def log_test_metrics(test_metrics):
    logging.info(f'Test Acc {test_metrics[ACCURACY]}')
    logging.info(f'Test MCC {test_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}')
    mlflow.log_metrics({ 'test acc': test_metrics[ACCURACY], 'test mcc': test_metrics[MATTHEWS_CORRELATION_COEFFICIENT] })

def compute_metrics(ground_truth, pred, include_AUC = False):
    metrics = np.zeros(3)
    if include_AUC:
        metrics[AUC] = sk.metrics.roc_auc_score(ground_truth, pred)
    pred_bin = pred > .5
    metrics[ACCURACY] = sk.metrics.accuracy_score(ground_truth, pred_bin)
    metrics[MATTHEWS_CORRELATION_COEFFICIENT] = sk.metrics.matthews_corrcoef(ground_truth, pred_bin)
    return metrics

def log_metrics_from_lists(acc, mcc, auc):
    global_acc = statistics.mean(acc)
    global_mcc = statistics.mean(mcc)
    global_auc = statistics.mean(auc)
    logging.info(f'Global ACC {global_acc}')
    logging.info(f'Global MCC {global_mcc}')
    logging.info(f'Global AUC {global_auc}')
    mlflow.log_metrics({ 'global_acc': global_acc, 'global_mcc': global_mcc, 'global_auc': global_auc })