import os
import pandas as pd
import numpy as np
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import ANTIBODIES_LIST

CV_FILES_DIR = 'C:/DOC/Articol HIV Antibodies/Comparing_with_other_work/Rawi/Results'

def compute_cv_metrics(ab):
    cv_paths = [f'{CV_FILES_DIR}/CV_{ab}/{p}' for p in os.listdir(f'{CV_FILES_DIR}/CV_{ab}')]
    all_results = []
    cv_means = []
    cv_std = []
    for i in range(len(cv_paths)):
        cv_results = pd.read_csv(cv_paths[i], index_col = 0).loc['mcc'][2:]
        cv_means.append(cv_results.mean())
        cv_std.append(cv_results.std())
        all_results += list(cv_results)
    all_results_np = np.array(all_results)
    cv_means_np = np.array(cv_means)
    cv_std_np = np.array(cv_std)
    assert len(all_results_np) == 100
    assert len(cv_means_np) == 10
    assert len(cv_std_np) == 10
    print(ab)
    print('All results mean', all_results_np.mean())
    print('All results std', all_results_np.std())
    print('Mean CV std', cv_std_np.mean())
    print('Std of CV means', cv_means_np.std())

if __name__ == '__main__':
    # for ab in ANTIBODIES_LIST:
    for ab in ['DH270.1']:
        compute_cv_metrics(ab)
