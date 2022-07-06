import pandas as pd
import numpy as np
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import ANTIBODIES_LIST
from collections import defaultdict
from deep_hiv_ab_pred.compare_to_Rawi_gbm.create_latex_tables_Rawi import acc_mean, acc_std, auc_mean, auc_std, mcc_mean, mcc_std
import math

CV_FILES_DIR = 'deep_hiv_ab_pred/compare_to_Rawi_gbm/cross_valid_results_gbm'

def compute_metrics_from_cross_valid_files():
    results_rawi = defaultdict(dict)
    for ab in ANTIBODIES_LIST:
        print('Processing', ab)
        acc_list, mcc_list, auc_list = [], [], []
        for i in range(1, 11):
            cv_results_csv = f'{CV_FILES_DIR}/{ab}/retrain_{i}.csv'
            cv_results = pd.read_csv(cv_results_csv, index_col = 0)
            for i in range(1, 11):
                acc = cv_results.loc['accuracy'][f'cv_{i}_valid']
                mcc = cv_results.loc['mcc'][f'cv_{i}_valid']
                auc = cv_results.loc['auc'][f'cv_{i}_valid']
                if not math.isnan(acc):
                    acc_list.append(acc)
                if not math.isnan(mcc):
                    mcc_list.append(mcc)
                if not math.isnan(auc):
                    auc_list.append(auc)
        print(f'acc list size = {len(acc_list)}; mcc list size = {len(mcc_list)}; auc list size = {len(auc_list)}')
        acc_list_np = np.array(acc_list)
        mcc_list_np = np.array(mcc_list)
        auc_list_np = np.array(auc_list)
        results_rawi[ab][acc_mean] = acc_list_np.mean()
        results_rawi[ab][auc_mean] = auc_list_np.mean()
        results_rawi[ab][mcc_mean] = mcc_list_np.mean()
        results_rawi[ab][acc_std] = acc_list_np.std(ddof = 1)
        results_rawi[ab][auc_std] = auc_list_np.std(ddof = 1)
        results_rawi[ab][mcc_std] = mcc_list_np.std(ddof = 1)
    return results_rawi

if __name__ == '__main__':
    metrics = compute_metrics_from_cross_valid_files()
    for ab, metrics in metrics.items():
        print(ab, metrics)


