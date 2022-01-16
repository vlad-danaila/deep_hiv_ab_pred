import mlflow

from deep_hiv_ab_pred.hyperparameters.constants import CONF_ICERI_V2, TRIAL_252
import numpy as np
import optuna
from optuna.pruners import BasePruner
from optuna.trial._state import TrialState
from deep_hiv_ab_pred.train_full_catnap.constants import SPLITS_HOLD_OUT_ONE_CLUSTER, SPLITS_UNIFORM, BEST_PARAMS
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.training.constants import MATTHEWS_CORRELATION_COEFFICIENT
from deep_hiv_ab_pred.util.tools import read_json_file, read_yaml, dump_json
from deep_hiv_ab_pred.util.logging import setup_logging
import logging
from deep_hiv_ab_pred.train_full_catnap.train_hold_out_one_cluster import train_hold_out_one_cluster
from deep_hiv_ab_pred.train_full_catnap.train_on_uniform_splits import train_on_uniform_splits
import os
import time
from deep_hiv_ab_pred.train_full_catnap.analyze_optuna_trials import get_best_trials_from_study
from deep_hiv_ab_pred.training.cv_pruner import CrossValidationPruner
import torch as t

def propose(trial: optuna.trial.Trial):
    kmer_len_virus = trial.suggest_int('KMER_LEN_VIRUS', 3, 110)
    return {
        'KMER_LEN_VIRUS': kmer_len_virus,
        'KMER_STRIDE_VIRUS': trial.suggest_int('KMER_STRIDE_VIRUS', max(1, kmer_len_virus // 10), kmer_len_virus),
        'BATCH_SIZE': trial.suggest_int('BATCH_SIZE', 50, 5000),
        'EPOCHS': 100,
        'LEARNING_RATE': trial.suggest_loguniform('LEARNING_RATE', 1e-6, 1e-1),
        'GRAD_NORM_CLIP': trial.suggest_loguniform('GRAD_NORM_CLIP', 1e-2, 1000),
        'RNN_HIDDEN_SIZE': trial.suggest_int('RNN_HIDDEN_SIZE', 16, 1024),
        'EMBEDDING_DROPOUT': trial.suggest_float('EMBEDDING_DROPOUT', 0, .5),
        'ANTIBODIES_DROPOUT': trial.suggest_float('ANTIBODIES_DROPOUT', 0, .5),
        'FULLY_CONNECTED_DROPOUT': trial.suggest_float('FULLY_CONNECTED_DROPOUT', 0, .5),
        'AB_TYPE_DROPOUT': trial.suggest_float('AB_TYPE_DROPOUT', 0, .5),
        'LOSS_AB_TO_VIRUS_RATIO': trial.suggest_float('LOSS_AB_TO_VIRUS_RATIO', 0, 1)
    }

# def train_hold_out_one_cluster(splits, catnap, conf, trial):
#     for i in range(5):
#         trial.report(random.random(), i)
#         if trial.should_prune():
#             raise optuna.TrialPruned()
#     return np.array([[random.random(), random.random(), random.random()]])

def empty_cuda_cahce():
    try:
        t.cuda.empty_cache()
    except Exception as e:
        logging.exception(str(e), exc_info = True)

def get_objective_train_hold_out_one_cluster():
    mlflow.set_tag('hyperparam opt', 'hold out one cluster')
    splits = read_json_file(SPLITS_HOLD_OUT_ONE_CLUSTER)
    catnap = read_json_file(CATNAP_FLAT)
    cvp = CrossValidationPruner(20, 3, 5, .1)
    def objective(trial):
        conf = propose(trial)
        try:
            start = time.time()
            cv_metrics = train_hold_out_one_cluster(splits, catnap, conf, cvp)
            end = time.time()
            cv_metrics = np.array(cv_metrics)
            cv_mean_mcc = cv_metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].mean()
        except optuna.TrialPruned as pruneError:
            raise pruneError
        except Exception as e:
            if str(e).startswith('CUDA out of memory'):
                logging.error('CUDA out of memory', exc_info = True)
                empty_cuda_cahce()
                raise optuna.TrialPruned()
            elif 'CUDA error' in str(e):
                logging.error('CUDA error', exc_info = True)
                empty_cuda_cahce()
                raise optuna.TrialPruned()
            logging.exception(str(e), exc_info = True)
            logging.error(f'Configuration {conf}')
            raise optuna.TrialPruned()
        return cv_mean_mcc
    return objective

def get_objective_train_on_uniform_splits():
    mlflow.set_tag('hyperparam opt', 'uniform splits')
    splits = read_json_file(SPLITS_UNIFORM)
    catnap = read_json_file(CATNAP_FLAT)
    cvp = CrossValidationPruner(20, 3, 1, .05)
    def objective(trial):
        conf = propose(trial)
        try:
            start = time.time()
            metrics = train_on_uniform_splits(splits, catnap, conf, None)
            end = time.time()
            metrics = np.array(metrics)
            return metrics[MATTHEWS_CORRELATION_COEFFICIENT]
        except optuna.TrialPruned as pruneError:
            raise pruneError
        except Exception as e:
            if str(e).startswith('CUDA out of memory'):
                logging.error('CUDA out of memory', exc_info = True)
                empty_cuda_cahce()
                raise optuna.TrialPruned()
            elif 'CUDA error' in str(e):
                logging.error('CUDA error', exc_info = True)
                empty_cuda_cahce()
                raise optuna.TrialPruned()
            logging.exception(str(e), exc_info = True)
            logging.error(f'Configuration {conf}')
            raise optuna.TrialPruned()
    return objective

# class CrossValidationPruner(BasePruner):
#     def __init__(self, treshold):
#         self.treshold = treshold
#
#     def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
#         step = trial.last_step
#         completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
#         if not completed_trials:
#             return False
#         score_matrix = np.array([
#             [ t.intermediate_values[i] for i in range(len(t.intermediate_values)) ]
#             for t in completed_trials
#         ])
#         global_average = np.zeros(len(score_matrix))
#         trail_average = 0
#         for i in range(step + 1):
#             global_average = global_average + score_matrix[:, i]
#             trail_average = trail_average + trial.intermediate_values[i]
#         global_average = global_average / (step + 1)
#         trail_average = trail_average / (step + 1)
#         maximum = max(global_average)
#         return trail_average < maximum - self.treshold

def optimize_hyperparameters():
    setup_logging()
    # trials = get_best_trials_from_study('ICERI_v2_previous', .48)
    # pruner = CrossValidationPruner(.05)
    study_name = 'ICERI2021_v2'
    study_exists = os.path.isfile(study_name + '.db')
    sampler = optuna.samplers.TPESampler(multivariate = True, warn_independent_sampling = True, n_startup_trials = 100)
    study = optuna.create_study(study_name = study_name, direction='maximize',
        storage = f'sqlite:///{study_name}.db', load_if_exists = True, sampler = sampler)
    initial_conf = read_yaml(CONF_ICERI_V2)
    trial_252 = read_json_file(TRIAL_252)
    if not study_exists:
        study.enqueue_trial(initial_conf)
        study.enqueue_trial(trial_252)
    #objective = get_objective_train_hold_out_one_cluster()
    objective = get_objective_train_on_uniform_splits()
    study.optimize(objective, n_trials=10000)
    logging.info(study.best_params)
    dump_json(study.best_params, BEST_PARAMS)

if __name__ == '__main__':
    optimize_hyperparameters()