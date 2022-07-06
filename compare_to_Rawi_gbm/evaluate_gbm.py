import pandas as pd
import numpy as np
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import ANTIBODIES_LIST
from collections import defaultdict
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import acc_mean, acc_std, auc_mean, auc_std, mcc_mean, mcc_std
import math

CV_FILES_DIR = 'deep_hiv_ab_pred/compare_to_Rawi_gbm/cross_valid_results_gbm'

def compute_gbm_metrics_from_cross_valid_files():
    results_rawi = defaultdict(dict)
    for ab in ANTIBODIES_LIST:
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

'''
Results Rawi from supplementary table
'''
def get_GBM_results_from_supplementary_table():
    str_antibodies_rawi = '	b12	4E10	2F5	2G12	VRC01	PG9	PGT121	PGT128	PGT145	3BNC117	PG16	10-1074	PGDM1400	VRC26.08	VRC26.25	VRC13	VRC03	VRC-PG04	35O22	NIH45-46	VRC-CH31	8ANC195	HJ16	PGT151	VRC38.01	CH01	PGT135	DH270.1	DH270.5	DH270.6	VRC29.03	VRC34.01	VRC07'
    str_acc_rawi = '0.79 (0.01)	0.94 (0)	0.95 (0)	0.91 (0.01)	0.92 (0)	0.86 (0.01)	0.88 (0.01)	0.86 (0.01)	0.86 (0.02)	0.9 (0.01)	0.84 (0.01)	0.94 (0.01)	0.89 (0)	0.85 (0.01)	0.87 (0.01)	0.88 (0.01)	0.81 (0.02)	0.87 (0.01)	0.66 (0.02)	0.89 (0.01)	0.87 (0.01)	0.89 (0.02)	0.66 (0.02)	0.83 (0.01)	0.87 (0.03)	0.77 (0.03)	0.74 (0.02)	0.9 (0.02)	0.91 (0.01)	0.93 (0.01)	0.84 (0.01)	0.79 (0.03)	0.93 (0.01)'
    str_auc_rawi = '0.82 (0.01)	0.82 (0.02)	0.97 (0)	0.93 (0)	0.89 (0.01)	0.85 (0.01)	0.92 (0)	0.89 (0.01)	0.86 (0.02)	0.88 (0.02)	0.79 (0.02)	0.95 (0.01)	0.83 (0.02)	0.89 (0.01)	0.89 (0.01)	0.83 (0.01)	0.83 (0.02)	0.78 (0.05)	0.63 (0.02)	0.8 (0.02)	0.78 (0.03)	0.9 (0.03)	0.67 (0.02)	0.78 (0.02)	0.87 (0.02)	0.77 (0.02)	0.77 (0.02)	0.92 (0.02)	0.93 (0.02)	0.93 (0.01)	0.82 (0.02)	0.78 (0.03)	0.78 (0.05)'
    str_mcc_rawi = '0.56 (0.02)	0.63 (0.02)	0.89 (0.01)	0.75 (0.01)	0.7 (0.02)	0.61 (0.02)	0.75 (0.01)	0.72 (0.01)	0.67 (0.04)	0.69 (0.03)	0.57 (0.04)	0.86 (0.01)	0.66 (0.02)	0.7 (0.02)	0.71 (0.04)	0.63 (0.03)	0.61 (0.03)	0.57 (0.05)	0.38 (0.04)	0.59 (0.05)	0.6 (0.06)	0.77 (0.04)	0.42 (0.03)	0.58 (0.03)	0.7 (0.05)	0.56 (0.04)	0.54 (0.01)	0.82 (0.03)	0.83 (0.02)	0.85 (0.02)	0.64 (0.02)	0.61 (0.05)	0.66 (0.04)'

    antibodies_rawi = str_antibodies_rawi.split()
    acc_rawi = str_acc_rawi.split()
    auc_rawi = str_auc_rawi.split()
    mcc_rawi = str_mcc_rawi.split()

    results_rawi = defaultdict(dict)

    for i in range(len(antibodies_rawi)):
        ab = antibodies_rawi[i]
        results_rawi[ab][acc_mean] = float(acc_rawi[i * 2])
        results_rawi[ab][acc_std] = float(acc_rawi[i * 2 + 1][1:-1])
        results_rawi[ab][auc_mean] = float(auc_rawi[i * 2])
        results_rawi[ab][auc_std] = float(auc_rawi[i * 2 + 1][1:-1])
        results_rawi[ab][mcc_mean] = float(mcc_rawi[i * 2])
        results_rawi[ab][mcc_std] = float(mcc_rawi[i * 2 + 1][1:-1])

    return results_rawi

def get_GBM_results():
    results_from_supplement = get_GBM_results_from_supplementary_table()
    results_from_cv_files = compute_gbm_metrics_from_cross_valid_files()
    for ab in ANTIBODIES_LIST:
        results_from_supplement[ab][acc_std] = results_from_cv_files[ab][acc_std]
        results_from_supplement[ab][mcc_std] = results_from_cv_files[ab][mcc_std]
        results_from_supplement[ab][auc_std] = results_from_cv_files[ab][auc_std]
    return results_from_supplement

if __name__ == '__main__':
    gbm_results = get_GBM_results()


