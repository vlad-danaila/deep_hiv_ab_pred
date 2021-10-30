import numpy as np
import optuna
from deep_hiv_ab_pred.train_full_catnap.constants import BEST_PARAMS
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.training.constants import MATTHEWS_CORRELATION_COEFFICIENT
from deep_hiv_ab_pred.util.tools import read_json_file, read_yaml, dump_json
import logging
import os
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import COMPARE_SPLITS_FOR_RAWI, MODELS_FOLDER, HYPERPARAM_PRETRAIN, KMER_LEN, KMER_STRIDE
from deep_hiv_ab_pred.train_full_catnap.hyperparameter_optimisation import HoldOutOneClusterCVPruner
from deep_hiv_ab_pred.preprocessing.sequences import parse_catnap_sequences
from deep_hiv_ab_pred.compare_to_Rawi_gbm.train_evaluate import pretrain_net, cross_validate
import sys

def propose(trial: optuna.trial.Trial, base_conf: dict):
    return {
        'BATCH_SIZE': trial.suggest_int('BATCH_SIZE', 50, 5000),
        'EPOCHS': trial.suggest_int('EPOCHS', 1, 100),
        'LEARNING_RATE': trial.suggest_loguniform('LEARNING_RATE', 1e-6, 1e-1),
        'GRAD_NORM_CLIP': trial.suggest_loguniform('GRAD_NORM_CLIP', 1e-2, 1000),
        'EMBEDDING_DROPOUT': trial.suggest_float('EMBEDDING_DROPOUT', 0, .5),
        'VIRUS_RNN_DROPOUT': trial.suggest_float('VIRUS_RNN_DROPOUT', 0, .5),
        'FULLY_CONNECTED_DROPOUT': trial.suggest_float('FULLY_CONNECTED_DROPOUT', 0, .5),

        'EMBEDDING_SIZE': base_conf['EMBEDDING_SIZE'],
        'KMER_LEN_ANTB': base_conf['KMER_LEN_ANTB'],
        'KMER_LEN_VIRUS': base_conf['KMER_LEN_VIRUS'],
        'KMER_STRIDE_VIRUS': base_conf['KMER_STRIDE_VIRUS'],
        'KMER_STRIDE_ANTB': base_conf['KMER_STRIDE_ANTB'],
        'RNN_HIDDEN_SIZE': base_conf['RNN_HIDDEN_SIZE'],
        'NB_LAYERS': base_conf['NB_LAYERS'],
        'ANTIBODIES_RNN_DROPOUT': base_conf['ANTIBODIES_RNN_DROPOUT']
    }

def get_objective_cross_validation(antibody, cv_folds_trim):
    splits = read_json_file(COMPARE_SPLITS_FOR_RAWI)[antibody]
    catnap = read_json_file(CATNAP_FLAT)
    base_conf = read_json_file(HYPERPARAM_PRETRAIN)
    virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap_sequences(
        base_conf['KMER_LEN_VIRUS'], base_conf['KMER_STRIDE_VIRUS'], base_conf['KMER_LEN_ANTB'], base_conf['KMER_STRIDE_ANTB']
    )
    if not os.path.isfile(os.path.join(MODELS_FOLDER, f'model_{antibody}_pretrain.tar')):
        pretrain_net(antibody, splits['pretraining'], catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq)

    def objective(trial):
        conf = propose(trial, base_conf)
        try:
            cv_metrics = cross_validate(antibody, splits['cross_validation'], catnap, conf,
                virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, trial, cv_folds_trim)
            cv_metrics = np.array(cv_metrics)
            cv_mean_mcc = cv_metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].mean()
            return cv_mean_mcc
        except optuna.TrialPruned as pruneError:
            raise pruneError
        except Exception as e:
            if str(e).startswith('CUDA out of memory'):
                print('CUDA out of memory')
                # t.cuda.empty_cache()
                raise optuna.TrialPruned()
            elif 'CUDA error' in str(e):
                print('CUDA error')
                # t.cuda.empty_cache()
                raise optuna.TrialPruned()
            logging.exception(str(e))
            print('Configuration', conf)
            raise optuna.TrialPruned()
        return cv_mean_mcc
    return objective

def optimize_hyperparameters(antibody_name):
    optuna.logging.get_logger("optuna").addHandler(logging.FileHandler('optuna log'))
    pruner = HoldOutOneClusterCVPruner(.1)
    study_name = 'Compare_Rawi_ICERI2021_v2_' + antibody_name
    study = optuna.create_study(study_name = study_name, direction = 'maximize',
                                storage = f'sqlite:///{study_name}.db', load_if_exists = True, pruner = pruner)
    objective = get_objective_cross_validation(antibody_name, cv_folds_trim = 10)
    study.optimize(objective, n_trials=500)
    print(study.best_params)
    dump_json(study.best_params, BEST_PARAMS)

if __name__ == '__main__':
    optimize_hyperparameters('10-1074')