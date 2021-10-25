import random

import numpy as np
import optuna
from deep_hiv_ab_pred.train_full_catnap.constants import SPLITS_HOLD_OUT_ONE_CLUSTER
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.util.tools import read_json_file
from deep_hiv_ab_pred.training.constants import MATTHEWS_CORRELATION_COEFFICIENT
# from deep_hiv_ab_pred.train_full_catnap.train_hold_out_one_cluster import train_hold_out_one_cluster

def propose(trial: optuna.trial.Trial):
    kmer_len_antb = trial.suggest_int('KMER_LEN_ANTB', 3, 110)
    kmer_len_virus = trial.suggest_int('KMER_LEN_VIRUS', 3, 110)
    return {
        'EMBEDDING_SIZE': trial.suggest_int('EMBEDDING_SIZE', 2, 128),
        'KMER_LEN_ANTB': kmer_len_antb,
        'KMER_STRIDE_ANTB': trial.suggest_int('KMER_STRIDE_ANTB', max(1, kmer_len_antb // 10), kmer_len_antb),
        'KMER_LEN_VIRUS': kmer_len_virus,
        'KMER_STRIDE_VIRUS': trial.suggest_int('KMER_STRIDE_VIRUS', max(1, kmer_len_virus // 10), kmer_len_virus),
        'BATCH_SIZE': trial.suggest_int('BATCH_SIZE', 50, 5000),
        'EPOCHS': trial.suggest_int('EPOCHS', 1, 100),
        'LEARNING_RATE': trial.suggest_loguniform('LEARNING_RATE', 1e-6, 1e-1),
        'GRAD_NORM_CLIP': trial.suggest_loguniform('GRAD_NORM_CLIP', 1e-2, 1000),
        'ANTIBODIES_LIGHT_RNN_HIDDEN_SIZE': trial.suggest_int('ANTIBODIES_LIGHT_RNN_HIDDEN_SIZE', 16, 1024),
        'ANTIBODIES_HEAVY_RNN_HIDDEN_SIZE': trial.suggest_int('ANTIBODIES_HEAVY_RNN_HIDDEN_SIZE', 16, 1024),
        'ANTIBODIES_LIGHT_RNN_NB_LAYERS': trial.suggest_int('ANTIBODIES_LIGHT_RNN_NB_LAYERS', 1, 10),
        'ANTIBODIES_HEAVY_RNN_NB_LAYERS': trial.suggest_int('ANTIBODIES_HEAVY_RNN_NB_LAYERS', 1, 10),
        'VIRUS_RNN_HIDDEN_NB_LAYERS': trial.suggest_int('VIRUS_RNN_HIDDEN_NB_LAYERS', 1, 10),
        'EMBEDDING_DROPOUT': trial.suggest_float('EMBEDDING_DROPOUT', 0, .5),
        'ANTIBODIES_LIGHT_RNN_DROPOUT': trial.suggest_float('ANTIBODIES_LIGHT_RNN_DROPOUT', 0, .5),
        'ANTIBODIES_HEAVY_RNN_DROPOUT': trial.suggest_float('ANTIBODIES_HEAVY_RNN_DROPOUT', 0, .5),
        'VIRUS_RNN_DROPOUT': trial.suggest_float('VIRUS_RNN_DROPOUT', 0, .5),
        'FULLY_CONNECTED_DROPOUT': trial.suggest_float('FULLY_CONNECTED_DROPOUT', 0, .5)
    }

def train_hold_out_one_cluster(splits, catnap, conf):
    return np.array([[random.random(), random.random(), random.random()]])

'''

TODO: Must maximize instead of minimize

'''
def get_objective_train_hold_out_one_cluster():
    splits = read_json_file(SPLITS_HOLD_OUT_ONE_CLUSTER)
    catnap = read_json_file(CATNAP_FLAT)
    def objective(trial):
        conf = propose(trial)
        try:
            cv_metrics = train_hold_out_one_cluster(splits, catnap, conf)
            cv_mean_mcc = cv_metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].mean()
        except Exception as e:
            print(e)
            return 0
        return cv_mean_mcc
    return objective

if __name__ == '__main__':
    study = optuna.create_study()
    objective = get_objective_train_hold_out_one_cluster()
    study.optimize(objective, n_trials=5)
    print(study.best_params)