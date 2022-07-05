import logging
import numpy as np
from deep_hiv_ab_pred.util.logging import setup_simple_logging
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import COMPARE_SPLITS_FOR_RAWI, ANTIBODIES_LIST, MODELS_FOLDER
from deep_hiv_ab_pred.compare_to_Rawi_gbm.hyperparameter_optimisation import get_data, add_properties_from_base_config
from deep_hiv_ab_pred.util.tools import read_json_file, dump_json
from deep_hiv_ab_pred.compare_to_Rawi_gbm.train_evaluate import pretrain_net, cross_validate_antibody
import os
from deep_hiv_ab_pred.util.metrics import compute_cv_metrics

PRETRAIN_EPOCHS = 100
# MLFLOW_ARTIFACTS_FOLDER = 'C:/DOC/Articol HIV Antibodies/Experiments/Experiments/Compare Rawi/Compare Rawi FC ATT GRU 1 layer trial 252 props only - 1000 tpe trials/mlruns/1/708a375930dd4ff1bb2d1b9686a7ddde/artifacts'
MLFLOW_ARTIFACTS_FOLDER = 'artifacts'

JSON_METRICS_FILE = 'deep_hiv_ab_pred/compare_to_Rawi_gbm/fc_gru_att_cv_metrics.json'

def get_configuration_from_mlflow_artifacts_folder(antibody):
    conf_path = f'{MLFLOW_ARTIFACTS_FOLDER}/{antibody} conf.json/{antibody}.json'
    return read_json_file(conf_path)

def evaluate_trained_model():
    setup_simple_logging()
    all_splits, catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = get_data(COMPARE_SPLITS_FOR_RAWI)
    cv_metrics_dict = {}
    for i, antibody in enumerate(ANTIBODIES_LIST):
        logging.info(f'Processing antibody {i} -> {antibody}')
        if not os.path.isfile(os.path.join(MODELS_FOLDER, f'model_{antibody}_pretrain.tar')):
            pretrain_net(antibody, all_splits[antibody]['pretraining'], catnap, base_conf, virus_seq,
                virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, PRETRAIN_EPOCHS)
        conf = get_configuration_from_mlflow_artifacts_folder(antibody)
        conf = add_properties_from_base_config(conf, base_conf)
        cv_metrics = cross_validate_antibody(antibody, all_splits[antibody]['cross_validation'], catnap,
            conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq)
        cv_metrics_dict[antibody] = [list(m) for m in cv_metrics]
    dump_json(cv_metrics_dict, 'cv_metrics.json')

def display_metrics_from_json():
    all_metrics = read_json_file(JSON_METRICS_FILE)
    acc, mcc, auc = [], [], []
    for ab, metrics in all_metrics.items():
        cv_mean_acc, cv_std_acc, cv_mean_mcc, cv_std_mcc, cv_mean_auc, cv_std_auc = compute_cv_metrics(metrics)
        print(ab, 'cv_mean_mcc', cv_mean_mcc, 'cv_std_mcc', cv_std_mcc, 'cv_mean_auc', cv_mean_auc, 'cv_std_auc', cv_std_auc, 'cv_mean_acc', cv_mean_acc, 'cv_std_acc', cv_std_acc)
        acc.append(cv_mean_acc)
        mcc.append(cv_mean_mcc)
        auc.append(cv_mean_auc)
    global_acc = np.array(acc).mean()
    global_mcc = np.array(mcc).mean()
    global_auc = np.array(auc).mean()
    print(f'Global ACC {global_acc}')
    print(f'Global MCC {global_mcc}')
    print(f'Global AUC {global_auc}')

if __name__ == '__main__':
    # evaluate_trained_model()
    display_metrics_from_json()