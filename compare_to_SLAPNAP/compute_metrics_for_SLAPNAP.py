import numpy as np
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.util.tools import read_json_file, dump_json
from deep_hiv_ab_pred.compare_to_SLAPNAP.constants import COMPARE_SPLITS_FOR_SLAPNAP, SLAPNAP_RESULTS
from os import listdir
from os.path import join
from deep_hiv_ab_pred.util.metrics import compute_metrics, ACCURACY, AUC, MATTHEWS_CORRELATION_COEFFICIENT

# 'pred'
# 'ground_truth'
# 'folds' are string keys and nb lists

def compute_metrics_for_SLAPNAP():
    metrics = {}
    for ab_results_file in listdir(SLAPNAP_RESULTS):
        if not ab_results_file.endswith('json'):
            continue
        ab_results = read_json_file(join(SLAPNAP_RESULTS, ab_results_file))
        antibody = '.'.join(ab_results_file.split('.')[:-1])
        pred = np.array(ab_results['pred'])
        ground_truth = np.array(ab_results['ground_truth'])
        for count, fold in ab_results['folds'].items():
            # folds were computed in R, and there the indexing starts with one instead of zero
            fold = np.array(fold) - 1
            fold_pred = pred[fold]
            fold_gt = ground_truth[fold]
            metrics = compute_metrics(fold_gt, fold_pred, include_AUC = True)
            print(antibody, 'fold', count, 'acc', metrics[ACCURACY], 'mcc', metrics[MATTHEWS_CORRELATION_COEFFICIENT], 'auc', metrics[AUC])
    return metrics

if __name__ == '__main__':
    metrics = compute_metrics_for_SLAPNAP()