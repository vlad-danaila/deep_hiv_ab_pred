from collections import defaultdict
import numpy as np
from deep_hiv_ab_pred.catnap.dataset import find_ab_types
from deep_hiv_ab_pred.util.tools import read_json_file
from deep_hiv_ab_pred.compare_to_Rawi_gbm.evaluate_from_saved_hyperparameters import JSON_METRICS_FILE
from deep_hiv_ab_pred.util.metrics import compute_cv_metrics

acc_mean, acc_std = 'acc_mean', 'acc_std'
auc_mean, auc_std = 'auc_mean', 'auc_std'
mcc_mean, mcc_std = 'mcc_mean', 'mcc_std'

'''
Results Rawi
'''
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