import logging
import sys
from deep_hiv_ab_pred.util.logging import setup_simple_logging
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import COMPARE_SPLITS_FOR_RAWI, MODELS_FOLDER, \
    CV_FOLDS_TRIM, N_TRIALS, PRUNE_TREHOLD, ANTIBODIES_LIST, FREEZE_ANTIBODY_AND_EMBEDDINGS, FREEZE_ALL_BUT_LAST_LAYER
from deep_hiv_ab_pred.compare_to_Rawi_gbm.hyperparameter_optimisation import get_data, add_properties_from_base_config
from os import listdir
from os.path import join
from deep_hiv_ab_pred.util.tools import read_json_file
from deep_hiv_ab_pred.compare_to_Rawi_gbm.train_evaluate import pretrain_net, cross_validate_antibody

'''
FYI:
Conf base iti vine din get_data, si e configurata de constanta DEFAULT_CONF, cred ca imi tb 252
Conf specific il gasesti in artifactele mlflow
Refoloseste functia cross_validate_antibody din hyperparameter_optimisation
Uite-te in test_optimized_antibody din hyperparameter optimisation pt handling la conf si ca sa vezi cum chemi cross_validate_antibody

TODO:
Salveaza rezultatele cross validation in json
Testeaza totul pe o masina cu GPU

DONE:
Verifica daca ai luat in calcul in articol versiunea cu 1000 de iteratii sau pe cea mai restransa, tb sa verifici valorile din articol vs mlflow
Am folosit versiunea cu 1000 iteratii TPE

Tb sa extragi configurarile atat base config dar si config sepcific per antibody

Tb sa preantrenezi un model si sa il salvezi in destinatia lui, ca si in aplicatie, cred ca ai deja functie ca sa faci pretraining dar si salvare
'''

PRETRAIN_EPOCHS = 100
MLFLOW_ARTIFACTS_FOLDER = 'C:/DOC/Articol HIV Antibodies/Experiments/Experiments/Compare Rawi/Compare Rawi FC ATT GRU 1 layer trial 252 props only - 1000 tpe trials/mlruns/1/708a375930dd4ff1bb2d1b9686a7ddde/artifacts'

def get_configuration_from_mlflow_artifacts_folder(antibody):
    conf_path = f'{MLFLOW_ARTIFACTS_FOLDER}/{antibody} conf.json/{antibody}.json'
    return read_json_file(conf_path)

def evaluate_trained_model():
    setup_simple_logging()
    all_splits, catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = get_data(COMPARE_SPLITS_FOR_RAWI)
    for antibody in ANTIBODIES_LIST:
        pretrain_net(antibody, all_splits[antibody]['pretraining'], catnap, base_conf, virus_seq,
            virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, PRETRAIN_EPOCHS)
        conf = get_configuration_from_mlflow_artifacts_folder(antibody)
        conf = add_properties_from_base_config(conf, base_conf)
        cv_metrics = cross_validate_antibody(antibody, all_splits[antibody]['cross_validation'], catnap,
            conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq)

if __name__ == '__main__':
    evaluate_trained_model()