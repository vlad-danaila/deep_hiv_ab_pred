import numpy as np
from deep_hiv_ab_pred.training.constants import MATTHEWS_CORRELATION_COEFFICIENT, ACCURACY, AUC
import mlflow
import sklearn.metrics
import sklearn as sk
from deep_hiv_ab_pred.training.constants import ACCURACY, MATTHEWS_CORRELATION_COEFFICIENT
import logging
import statistics

def compute_cv_metrics(cv_metrics):
    cv_metrics = np.array(cv_metrics)
    cv_mean_acc = cv_metrics[:, ACCURACY].mean()
    cv_std_acc = cv_metrics[:, ACCURACY].std()
    cv_mean_mcc = cv_metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].mean()
    cv_std_mcc = cv_metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].std()
    cv_mean_auc = cv_metrics[:, AUC].mean()
    cv_std_auc = cv_metrics[:, AUC].std()
    return cv_mean_acc, cv_std_acc, cv_mean_mcc, cv_std_mcc, cv_mean_auc, cv_std_auc

def log_metrics_per_cv_antibody(cv_metrics, antibody):
    cv_mean_acc, cv_std_acc, cv_mean_mcc, cv_std_mcc, cv_mean_auc, cv_std_auc = compute_cv_metrics(cv_metrics)
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

def log_metrics_per_test_set_antibody(metrics, antibody):
    logging.info(f'Test Acc {metrics[ACCURACY]}')
    logging.info(f'Test MCC {metrics[MATTHEWS_CORRELATION_COEFFICIENT]}')
    logging.info(f'Test AUC {metrics[AUC]}')
    mlflow.log_metrics({
        f'test acc {antibody}': metrics[ACCURACY],
        f'test mcc {antibody}': metrics[MATTHEWS_CORRELATION_COEFFICIENT],
        f'test auc {antibody}': metrics[AUC],
    })
    return metrics[ACCURACY], metrics[MATTHEWS_CORRELATION_COEFFICIENT], metrics[AUC]

def log_test_metrics(test_metrics):
    logging.info(f'Test Acc {test_metrics[ACCURACY]}')
    logging.info(f'Test MCC {test_metrics[MATTHEWS_CORRELATION_COEFFICIENT]}')
    metrics_dict = { 'test acc': test_metrics[ACCURACY], 'test mcc': test_metrics[MATTHEWS_CORRELATION_COEFFICIENT] }
    if len(test_metrics) == 3:
        logging.info(f'Test AUC {test_metrics[AUC]}')
        metrics_dict['test auc'] = test_metrics[AUC]
    mlflow.log_metrics(metrics_dict)

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

def find_ideal_epoch(metrics_per_epochs):
    best = max(( m[MATTHEWS_CORRELATION_COEFFICIENT] for m in metrics_per_epochs ))
    epoch_index, metrics = [
        (i, m) for (i, m) in enumerate(metrics_per_epochs)
        if m[MATTHEWS_CORRELATION_COEFFICIENT] == best
    ][-1]
    return epoch_index, metrics