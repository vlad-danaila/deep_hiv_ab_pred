from collections import defaultdict
import numpy as np
from deep_hiv_ab_pred.catnap.dataset import find_ab_types
from deep_hiv_ab_pred.util.tools import read_json_file
from deep_hiv_ab_pred.compare_to_Rawi_gbm.evaluate_from_saved_hyperparameters import JSON_METRICS_FILE
from deep_hiv_ab_pred.util.metrics import compute_cv_metrics
from deep_hiv_ab_pred.compare_to_Rawi_gbm.evaluate_gbm import get_GBM_results
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import acc_mean, acc_std, auc_mean, auc_std, mcc_mean, mcc_std

'''
Result GBM
'''
results_rawi = get_GBM_results()

'''
Results FC-ATT-GRU
'''
results_fc_att_gru = defaultdict(dict)

all_metrics = read_json_file(JSON_METRICS_FILE)

for ab, metrics in all_metrics.items():
    cv_mean_acc, cv_std_acc, cv_mean_mcc, cv_std_mcc, cv_mean_auc, cv_std_auc = compute_cv_metrics(metrics)
    results_fc_att_gru[ab][acc_mean] = cv_mean_acc
    results_fc_att_gru[ab][auc_mean] = cv_mean_auc
    results_fc_att_gru[ab][mcc_mean] = cv_mean_mcc
    results_fc_att_gru[ab][acc_std] = cv_std_acc
    results_fc_att_gru[ab][auc_std] = cv_std_auc
    results_fc_att_gru[ab][mcc_std] = cv_std_mcc

def bold(text):
    return '\\textbf{' + text + '}'

def display_table_row(ab, metrics):
    rawi_mcc = f'{str(metrics[0])[:4]}({round(metrics[1], 2)})'
    rawi_auc = f'{str(metrics[2])[:4]}({round(metrics[3], 2)})'
    rawi_acc = f'{str(metrics[4])[:4]}({round(metrics[5], 2)})'
    net_mcc = f'{str(metrics[6])[:4]}({round(metrics[7], 2)})'
    net_auc = f'{str(metrics[8])[:4]}({round(metrics[9], 2)})'
    net_acc = f'{str(metrics[10])[:4]}({round(metrics[11], 2)})'

    if metrics[0] > metrics[6]:
        rawi_mcc = bold(rawi_mcc)
    elif metrics[0] < metrics[6]:
        net_mcc = bold(net_mcc)

    if metrics[2] > metrics[8]:
        rawi_auc = bold(rawi_auc)
    elif metrics[2] < metrics[8]:
        net_auc = bold(net_auc)

    if metrics[4] > metrics[10]:
        rawi_acc = bold(rawi_acc)
    elif metrics[4] < metrics[10]:
        net_acc = bold(net_acc)

    table_row = f'{ab} & {rawi_mcc} & {rawi_auc} & {rawi_acc} & {net_mcc} & {net_auc} & {net_acc}\\\\'
    return table_row

def group_antibodies_by_function_detailed(abs):
    ab_types = find_ab_types()
    types_ab = defaultdict(lambda : [])
    for ab in abs:
        types = ab_types[ab]
        types_ab[types].append(ab)
    for type in types_ab:
        types_ab[type] = sorted(types_ab[type], key = lambda x: x.lower())
    return types_ab

def group_antibodies_by_function_highlevel(abs):
    ab_types = find_ab_types()
    types_ab = defaultdict(lambda : [])
    for ab in abs:
        types = ab_types[ab]
        if 'gp120 V3 // V3 glycan (V3g)' in types \
                or 'gp120 V2 // V2 glycan(V2g) // V2 apex' in types \
                or 'gp120 V1-V2' in types:
            types_ab['gp120 other than CD4BS'].append(ab)
        elif 'gp41 MPER (membrane proximal external region)' in types \
                or 'gp41-gp120 interface' in types\
                or 'gp41-gp41 interface' in types\
                or 'fusion peptide // near gp41-gp120 interface' in types:
            types_ab['gp41 MPER, gp41-gp120 interface, and fusion peptide'].append(ab)
        elif 'gp120 CD4BS' in types:
            types_ab['gp120 CD4BS'].append(ab)
        else:
            raise 'Must map antibody category.'
    # Sort the antibodies
    for type in types_ab:
        types_ab[type] = sorted(types_ab[type], key = lambda x: x.lower())
    return types_ab

if __name__ == '__main__':
    totals = np.zeros(12)
    antibodies = list(results_fc_att_gru.keys())

    antibodies_grouped = group_antibodies_by_function_highlevel(antibodies)

    for ab_type in ['gp120 CD4BS', 'gp120 other than CD4BS', 'gp41 MPER, gp41-gp120 interface, and fusion peptide']:
        print('\midrule')
        print('\multicolumn{7}{c}{' + ab_type + '}\\\\')
        print('\midrule')
        totals_per_category = np.zeros(12)
        for ab in antibodies_grouped[ab_type]:
            metrics_us = results_fc_att_gru[ab]
            metrics_Rawi = results_rawi[ab]
            metrics = np.array([
                metrics_Rawi[mcc_mean], metrics_Rawi[mcc_std],
                metrics_Rawi[auc_mean], metrics_Rawi[auc_std],
                metrics_Rawi[acc_mean], metrics_Rawi[acc_std],
                metrics_us[mcc_mean], metrics_us[mcc_std],
                metrics_us[auc_mean], metrics_us[auc_std],
                metrics_us[acc_mean], metrics_us[acc_std]
            ])
            totals = totals + metrics
            totals_per_category = totals_per_category + metrics
            print(display_table_row(ab, metrics))
        totals_per_category = totals_per_category / len(antibodies_grouped[ab_type])
        print(display_table_row('Average', totals_per_category))
    totals = totals / len(results_fc_att_gru)
    print('\midrule')
    print(display_table_row('Global Average', totals))