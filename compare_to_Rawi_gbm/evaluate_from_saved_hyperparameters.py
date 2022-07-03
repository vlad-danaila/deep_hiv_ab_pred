from deep_hiv_ab_pred.util.logging import setup_simple_logging
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import COMPARE_SPLITS_FOR_RAWI, ANTIBODIES_LIST, MODELS_FOLDER
from deep_hiv_ab_pred.compare_to_Rawi_gbm.hyperparameter_optimisation import get_data, add_properties_from_base_config
from deep_hiv_ab_pred.util.tools import read_json_file, dump_json
from deep_hiv_ab_pred.compare_to_Rawi_gbm.train_evaluate import pretrain_net, cross_validate_antibody
import os

PRETRAIN_EPOCHS = 100
# MLFLOW_ARTIFACTS_FOLDER = 'C:/DOC/Articol HIV Antibodies/Experiments/Experiments/Compare Rawi/Compare Rawi FC ATT GRU 1 layer trial 252 props only - 1000 tpe trials/mlruns/1/708a375930dd4ff1bb2d1b9686a7ddde/artifacts'
MLFLOW_ARTIFACTS_FOLDER = 'artifacts'

def get_configuration_from_mlflow_artifacts_folder(antibody):
    conf_path = f'{MLFLOW_ARTIFACTS_FOLDER}/{antibody} conf.json/{antibody}.json'
    return read_json_file(conf_path)

def evaluate_trained_model():
    setup_simple_logging()
    all_splits, catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = get_data(COMPARE_SPLITS_FOR_RAWI)
    cv_metrics_dict = {}
    for antibody in ANTIBODIES_LIST:
        if not os.path.isfile(os.path.join(MODELS_FOLDER, f'model_{antibody}_pretrain.tar')):
            pretrain_net(antibody, all_splits[antibody]['pretraining'], catnap, base_conf, virus_seq,
                virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, PRETRAIN_EPOCHS)
        conf = get_configuration_from_mlflow_artifacts_folder(antibody)
        conf = add_properties_from_base_config(conf, base_conf)
        cv_metrics = cross_validate_antibody(antibody, all_splits[antibody]['cross_validation'], catnap,
            conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq)
        cv_metrics_dict[antibody] = cv_metrics
    dump_json(cv_metrics_dict)

if __name__ == '__main__':
    evaluate_trained_model()