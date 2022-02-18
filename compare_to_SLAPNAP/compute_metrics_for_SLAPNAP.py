import numpy as np
from deep_hiv_ab_pred.util.tools import read_json_file
from deep_hiv_ab_pred.util.metrics import compute_cv_metrics
from deep_hiv_ab_pred.compare_to_SLAPNAP.constants import SLAPNAP_RESULTS
from os import listdir
from os.path import join
from deep_hiv_ab_pred.util.metrics import compute_metrics

def compute_metrics_for_SLAPNAP():
    metrics = {}
    for ab_results_file in listdir(SLAPNAP_RESULTS):
        if not ab_results_file.endswith('json'):
            continue
        ab_results = read_json_file(join(SLAPNAP_RESULTS, ab_results_file))
        antibody = '.'.join(ab_results_file.split('.')[:-1])
        pred = np.array(ab_results['pred'])
        ground_truth = np.array(ab_results['ground_truth'])
        cv_metrics = []
        for count, fold in ab_results['folds'].items():
            # folds were computed in R, and there the indexing starts with one instead of zero
            fold = np.array(fold) - 1
            fold_pred = pred[fold]
            fold_gt = ground_truth[fold]
            fold_metrics = compute_metrics(fold_gt, fold_pred, include_AUC = True)
            cv_metrics.append(fold_metrics.tolist())
        cv_mean_acc, cv_std_acc, cv_mean_mcc, cv_std_mcc, cv_mean_auc, cv_std_auc = compute_cv_metrics(cv_metrics)
        metrics[antibody] = {
            'cv_mean_acc': cv_mean_acc,
            'cv_std_acc': cv_std_acc,
            'cv_mean_mcc': cv_mean_mcc,
            'cv_std_mcc': cv_std_mcc,
            'cv_mean_auc': cv_mean_auc,
            'cv_std_auc': cv_std_auc
        }
    return metrics

if __name__ == '__main__':
    metrics = compute_metrics_for_SLAPNAP()
    print(metrics)