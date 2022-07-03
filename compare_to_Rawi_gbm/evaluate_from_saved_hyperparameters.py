import logging
import sys
from deep_hiv_ab_pred.util.logging import setup_simple_logging
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import COMPARE_SPLITS_FOR_RAWI, MODELS_FOLDER, \
    CV_FOLDS_TRIM, N_TRIALS, PRUNE_TREHOLD, ANTIBODIES_LIST, FREEZE_ANTIBODY_AND_EMBEDDINGS, FREEZE_ALL_BUT_LAST_LAYER
from deep_hiv_ab_pred.compare_to_Rawi_gbm.hyperparameter_optimisation import get_data

PRETRAIN_EPOCHS = 100
freeze_mode = FREEZE_ANTIBODY_AND_EMBEDDINGS

'''
FYI:
Conf base iti vine din get_data, si e configurata de constanta DEFAULT_CONF, cred ca imi tb 252
Refoloseste functia cross_validate_antibody din hyperparameter_optimisation
Uite-te in test_optimized_antibody din hyperparameter optimisation pt handling la conf si ca sa vezi cum chemi cross_validate_antibody

TODO:
Tb sa extragi configurarile atat base config dar si config sepcific per antibody
Tb sa preantrenezi un model si sa il salvezi in destinatia lui, ca si in aplicatie, cred ca ai deja functie ca sa faci pretraining dar si salvare
Verifica daca ai luat in calcul in articol versiunea cu 1000 de iteratii sau pe cea mai restransa, tb sa verifici valorile din articol vs mlflow
'''

def evaluate_trained_model():
    setup_simple_logging()
    all_splits, catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = get_data(COMPARE_SPLITS_FOR_RAWI)
    for antibody in ANTIBODIES_LIST:
        pass


if __name__ == '__main__':
    evaluate_trained_model()