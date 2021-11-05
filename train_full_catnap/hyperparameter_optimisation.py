import random
from deep_hiv_ab_pred.hyperparameters.constants import CONF_ICERI_V2
import numpy as np
import optuna
from optuna.pruners import BasePruner
from optuna.trial._state import TrialState
from deep_hiv_ab_pred.train_full_catnap.constants import SPLITS_HOLD_OUT_ONE_CLUSTER, BEST_PARAMS
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.training.constants import MATTHEWS_CORRELATION_COEFFICIENT
from deep_hiv_ab_pred.util.tools import read_json_file, read_yaml, dump_json
import torch as t
import logging
import sys
from deep_hiv_ab_pred.train_full_catnap.train_hold_out_one_cluster import train_hold_out_one_cluster
import os

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
        'RNN_HIDDEN_SIZE': trial.suggest_int('RNN_HIDDEN_SIZE', 16, 1024),
        'NB_LAYERS': trial.suggest_int('NB_LAYERS', 1, 10),
        'EMBEDDING_DROPOUT': trial.suggest_float('EMBEDDING_DROPOUT', 0, .5),
        'ANTIBODIES_RNN_DROPOUT': trial.suggest_float('ANTIBODIES_RNN_DROPOUT', 0, .5),
        'VIRUS_RNN_DROPOUT': trial.suggest_float('VIRUS_RNN_DROPOUT', 0, .5),
        'FULLY_CONNECTED_DROPOUT': trial.suggest_float('FULLY_CONNECTED_DROPOUT', 0, .5)
    }

# def train_hold_out_one_cluster(splits, catnap, conf, trial):
#     for i in range(5):
#         trial.report(random.random(), i)
#         if trial.should_prune():
#             raise optuna.TrialPruned()
#     return np.array([[random.random(), random.random(), random.random()]])

def get_objective_train_hold_out_one_cluster():
    splits = read_json_file(SPLITS_HOLD_OUT_ONE_CLUSTER)
    catnap = read_json_file(CATNAP_FLAT)
    def objective(trial):
        conf = propose(trial)
        try:
            cv_metrics = train_hold_out_one_cluster(splits, catnap, conf, trial)
            cv_metrics = np.array(cv_metrics)
            cv_mean_mcc = cv_metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].mean()
        except optuna.TrialPruned as pruneError:
            raise pruneError
        except Exception as e:
            if str(e).startswith('CUDA out of memory'):
                logging.error('CUDA out of memory', exc_info = True)
                # t.cuda.empty_cache()
                raise optuna.TrialPruned()
            elif 'CUDA error' in str(e):
                logging.error('CUDA error', exc_info = True)
                # t.cuda.empty_cache()
                raise optuna.TrialPruned()
            logging.exception(str(e), exc_info = True)
            logging.error('Configuration ' + conf)
            raise optuna.TrialPruned()
        return cv_mean_mcc
    return objective

class BestPruner(BasePruner):
    def __init__(self, treshold):
        self.treshold = treshold

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        step = trial.last_step
        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        if not completed_trials:
            return False
        score_matrix = np.array([
            [ t.intermediate_values[i] for i in range(len(t.intermediate_values)) ]
            for t in completed_trials
        ])
        global_average = np.zeros(len(score_matrix))
        trail_average = 0
        for i in range(step + 1):
            global_average = global_average + score_matrix[:, i]
            trail_average = trail_average + trial.intermediate_values[i]
        global_average = global_average / (step + 1)
        trail_average = trail_average / (step + 1)
        maximum = max(global_average)
        return trail_average < maximum - self.treshold

def optimize_hyperparameters():
    optuna.logging.get_logger("optuna").addHandler(logging.FileHandler('optuna log'))
    pruner = BestPruner(.05)
    study_name = 'ICERI2021_v2'
    study_exists = os.path.isfile(study_name + '.db')
    study = optuna.create_study(study_name = study_name, direction = 'maximize',
                            storage = f'sqlite:///{study_name}.db', load_if_exists = True, pruner = pruner)
    initial_conf = read_yaml(CONF_ICERI_V2)
    if not study_exists:
        study.enqueue_trial(initial_conf)
    objective = get_objective_train_hold_out_one_cluster()
    study.optimize(objective, n_trials=1000)
    logging.info(study.best_params)
    dump_json(study.best_params, BEST_PARAMS)

if __name__ == '__main__':
    optimize_hyperparameters()