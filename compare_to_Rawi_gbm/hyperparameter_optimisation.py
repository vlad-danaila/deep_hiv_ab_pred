import numpy as np
import optuna
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import HYPERPARAM_FOLDER_ANTIBODIES
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.training.constants import MATTHEWS_CORRELATION_COEFFICIENT, ACCURACY
from deep_hiv_ab_pred.util.tools import read_json_file, dump_json
import logging
import os
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import COMPARE_SPLITS_FOR_RAWI, MODELS_FOLDER, \
    HYPERPARAM_PRETRAIN, CV_FOLDS_TRIM, N_TRIALS, PRUNE_TREHOLD, ANTIBODIES_LIST
from deep_hiv_ab_pred.train_full_catnap.hyperparameter_optimisation import HoldOutOneClusterCVPruner
from deep_hiv_ab_pred.preprocessing.sequences import parse_catnap_sequences
from deep_hiv_ab_pred.compare_to_Rawi_gbm.train_evaluate import pretrain_net, cross_validate
from os.path import join
import mlflow

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
    all_splits, catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = get_data()
    splits = all_splits[antibody]
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

def optimize_hyperparameters(antibody_name, cv_folds_trim = 10, n_trials = 1000, prune_trehold = .1):
    optuna.logging.get_logger("optuna").addHandler(logging.FileHandler('optuna log'))
    pruner = HoldOutOneClusterCVPruner(prune_trehold)
    study_name = 'Compare_Rawi_ICERI2021_v2_' + antibody_name
    study = optuna.create_study(study_name = study_name, direction = 'maximize',
                                storage = f'sqlite:///{study_name}.db', load_if_exists = True, pruner = pruner)
    objective = get_objective_cross_validation(antibody_name, cv_folds_trim = cv_folds_trim)
    study.optimize(objective, n_trials = n_trials)
    print(study.best_params)
    dump_json(study.best_params, join(HYPERPARAM_FOLDER_ANTIBODIES, f'{antibody_name}.json'))

def get_data():
    all_splits = read_json_file(COMPARE_SPLITS_FOR_RAWI)
    catnap = read_json_file(CATNAP_FLAT)
    base_conf = read_json_file(HYPERPARAM_PRETRAIN)
    virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap_sequences(
        base_conf['KMER_LEN_VIRUS'], base_conf['KMER_STRIDE_VIRUS'], base_conf['KMER_LEN_ANTB'], base_conf['KMER_STRIDE_ANTB']
    )
    return all_splits, catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq

def test_optimized_antibody(antibody):
    mlflow.log_params({ 'cv_folds_trim': CV_FOLDS_TRIM, 'n_trials': N_TRIALS, 'prune_trehold': PRUNE_TREHOLD })
    optimize_hyperparameters(antibody, cv_folds_trim = CV_FOLDS_TRIM, n_trials = N_TRIALS, prune_trehold = PRUNE_TREHOLD)
    all_splits, catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = get_data()
    mlflow.log_artifact(HYPERPARAM_PRETRAIN, 'base_conf.json')
    conf = read_json_file(join(HYPERPARAM_FOLDER_ANTIBODIES, f'{antibody}.json'))
    mlflow.log_params(conf)
    cv_metrics = cross_validate(antibody, all_splits[antibody]['cross_validation'], catnap, conf,
        virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq)
    log_metrics(cv_metrics, antibody)

def log_metrics(cv_metrics, antibody):
    cv_metrics = np.array(cv_metrics)
    cv_mean_acc = cv_metrics[:, ACCURACY].mean()
    cv_std_acc = cv_metrics[:, ACCURACY].std()
    cv_mean_mcc = cv_metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].mean()
    cv_std_mcc = cv_metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].std()
    print('CV Mean Acc', cv_mean_acc, 'CV Std Acc', cv_std_acc)
    print('CV Mean MCC', cv_mean_mcc, 'CV Std MCC', cv_std_mcc)
    mlflow.log_metrics({
        f'cv mean acc {antibody}': cv_mean_acc,
        f'cv std acc {antibody}': cv_std_acc,
        f'cv mean mcc {antibody}': cv_mean_mcc,
        f'cv std mcc {antibody}': cv_std_mcc
    })

def test_optimized_antibodies():
    for antibody in ANTIBODIES_LIST:
        test_optimized_antibody(antibody)

if __name__ == '__main__':
    test_optimized_antibodies()