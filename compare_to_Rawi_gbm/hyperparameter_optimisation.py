import numpy as np
import optuna
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import HYPERPARAM_FOLDER_ANTIBODIES
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.training.constants import MATTHEWS_CORRELATION_COEFFICIENT
from deep_hiv_ab_pred.util.tools import read_json_file, dump_json, get_experiment
import os
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import COMPARE_SPLITS_FOR_RAWI, MODELS_FOLDER, \
    CV_FOLDS_TRIM, N_TRIALS, PRUNE_TREHOLD, ANTIBODIES_LIST, FREEZE_ANTIBODY_AND_EMBEDDINGS, FREEZE_ALL_BUT_LAST_LAYER
from deep_hiv_ab_pred.global_constants import DEFAULT_CONF
from deep_hiv_ab_pred.preprocessing.seq_to_embed_for_transformer import parse_catnap_sequences_to_embeddings
from deep_hiv_ab_pred.compare_to_Rawi_gbm.train_evaluate import pretrain_net, cross_validate_antibody
from os.path import join
import mlflow
import statistics
from deep_hiv_ab_pred.util.metrics import log_metrics_per_cv_antibody
import copy
from deep_hiv_ab_pred.util.logging import setup_logging
import logging
from optuna.pruners import BasePruner
from optuna.trial._state import TrialState
from deep_hiv_ab_pred.train_full_catnap.hyperparameter_optimisation import handle_categorical_params

def propose_conf_for_frozen_antb_and_embeddings(trial: optuna.trial.Trial, base_conf: dict):
    return {
        "BATCH_SIZE": trial.suggest_int('BATCH_SIZE', 50, 5000),
        "LEARNING_RATE": trial.suggest_loguniform('LEARNING_RATE', 1e-4, 1),
        "WARMUP": trial.suggest_int("WARMUP", 1000, 100_000, log = True),
        "EMBEDDING_DROPOUT": base_conf['EMBEDDING_DROPOUT'],
        "FULLY_CONNECTED_DROPOUT": trial.suggest_float('FULLY_CONNECTED_DROPOUT', 0, .5),
        "N_HEADS_ENCODER": base_conf["N_HEADS_ENCODER"],
        "TRANS_HIDDEN_ENCODER": base_conf["TRANS_HIDDEN_ENCODER"],
        "TRANS_DROPOUT_ENCODER": base_conf["TRANS_DROPOUT_ENCODER"],
        "N_HEADS_DECODER": base_conf["N_HEADS_DECODER"],
        "TRANS_HIDDEN_DECODER": base_conf["TRANS_HIDDEN_DECODER"],
        "TRANS_DROPOUT_DECODER": trial.suggest_float('TRANS_DROPOUT_DECODER', 0, .5),
        "POS_EMBED": base_conf["POS_EMBED"],
        "TRANSF_ENCODER_LAYERS": base_conf["TRANSF_ENCODER_LAYERS"],
        "TRANSF_DECODER_LAYERS":  base_conf["TRANSF_DECODER_LAYERS"]
    }

class CrossValidationPruner(BasePruner):
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
        trial_average = 0
        for i in range(step + 1):
            global_average = global_average + score_matrix[:, i]
            trial_average = trial_average + trial.intermediate_values[i]
        global_average = global_average / (step + 1)
        trial_average = trial_average / (step + 1)
        maximum = max(global_average)
        return trial_average < maximum - self.treshold or trial_average <= 0

def propose_conf_for_frozen_net_without_last_layer(trial: optuna.trial.Trial, base_conf: dict):
    conf = copy.deepcopy(base_conf)
    conf['GRAD_NORM_CLIP'] = 1000
    conf['BATCH_SIZE'] = trial.suggest_int('BATCH_SIZE', 50, 5000)
    conf['EPOCHS'] = trial.suggest_int('EPOCHS', 1, 100)
    conf['LEARNING_RATE'] = trial.suggest_loguniform('LEARNING_RATE', 1e-6, 1e-1)
    conf['FULLY_CONNECTED_DROPOUT'] = trial.suggest_float('FULLY_CONNECTED_DROPOUT', 0, .5)
    return conf

def get_objective_cross_validation(antibody, cv_folds_trim, freeze_mode, pretrain_epochs):
    all_splits, catnap, base_conf, virus_seq, abs, virus_max_len, ab_max_len = get_data()
    splits = all_splits[antibody]
    if not os.path.isfile(os.path.join(MODELS_FOLDER, f'model_{antibody}_pretrain.tar')):
        pretrain_net(antibody, splits['pretraining'], catnap, base_conf, virus_seq, abs, virus_max_len, ab_max_len, pretrain_epochs)
    def objective(trial):
        if freeze_mode == FREEZE_ANTIBODY_AND_EMBEDDINGS:
            conf = propose_conf_for_frozen_antb_and_embeddings(trial, base_conf)
        # deprecated
        elif freeze_mode == FREEZE_ALL_BUT_LAST_LAYER:
            conf = propose_conf_for_frozen_net_without_last_layer(trial, base_conf)
        else:
            raise 'Must provide a proper freeze mode.'
        try:
            cv_metrics = cross_validate_antibody(antibody, splits['cross_validation'], catnap, conf, virus_seq,
                                                 abs, virus_max_len, ab_max_len, trial, cv_folds_trim, freeze_mode)
            cv_metrics = np.array(cv_metrics)
            cv_mean_mcc = cv_metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].mean()
            return cv_mean_mcc
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
            logging.error(f'Configuration {conf}')
            raise optuna.TrialPruned()
        return cv_mean_mcc
    return objective

def optimize_hyperparameters(antibody_name, cv_folds_trim = 10, n_trials = 1000, prune_trehold = .1, model_trial_name = '',
        freeze_mode = FREEZE_ANTIBODY_AND_EMBEDDINGS, pretrain_epochs=None):
    pruner = CrossValidationPruner(prune_trehold)
    sampler = optuna.samplers.TPESampler(multivariate = True, n_startup_trials = 100)
    study_name = f'Compare_Rawi_ICERI2021_v2_{model_trial_name}_{antibody_name}'
    study = optuna.create_study(study_name = study_name, direction = 'maximize',
                                storage = f'sqlite:///{study_name}.db', load_if_exists = True, pruner = pruner, sampler = sampler)
    objective = get_objective_cross_validation(antibody_name, cv_folds_trim, freeze_mode, pretrain_epochs)
    study.optimize(objective, n_trials = n_trials)
    logging.info(study.best_params)
    dump_json(study.best_params, join(HYPERPARAM_FOLDER_ANTIBODIES, f'{antibody_name}.json'))

def get_data():
    all_splits = read_json_file(COMPARE_SPLITS_FOR_RAWI)
    catnap = read_json_file(CATNAP_FLAT)
    base_conf = read_json_file(DEFAULT_CONF)
    base_conf = handle_categorical_params(base_conf)
    virus_seq, abs, virus_max_len, ab_max_len = parse_catnap_sequences_to_embeddings()
    return all_splits, catnap, base_conf, virus_seq, abs, virus_max_len, ab_max_len

def add_properties_from_base_config(conf, base_conf):
    for prop in base_conf:
        if prop not in conf:
            conf[prop] = base_conf[prop]
    return conf

def test_optimized_antibody(antibody, model_trial_name = '', freeze_mode = FREEZE_ANTIBODY_AND_EMBEDDINGS, pretrain_epochs = None):
    mlflow.log_params({ 'cv_folds_trim': CV_FOLDS_TRIM, 'n_trials': N_TRIALS, 'prune_trehold': PRUNE_TREHOLD })
    optimize_hyperparameters(antibody, cv_folds_trim = CV_FOLDS_TRIM, n_trials = N_TRIALS,
        prune_trehold = PRUNE_TREHOLD, model_trial_name = model_trial_name, freeze_mode = freeze_mode, pretrain_epochs = pretrain_epochs)
    all_splits, catnap, base_conf, virus_seq, abs, virus_max_len, ab_max_len = get_data()
    mlflow.log_artifact(DEFAULT_CONF, 'base_conf.json')
    conf = read_json_file(join(HYPERPARAM_FOLDER_ANTIBODIES, f'{antibody}.json'))
    mlflow.log_artifact(join(HYPERPARAM_FOLDER_ANTIBODIES, f'{antibody}.json'), f'{antibody} conf.json')
    conf = add_properties_from_base_config(conf, base_conf)
    cv_metrics = cross_validate_antibody(antibody, all_splits[antibody]['cross_validation'], catnap, conf,
        virus_seq, abs, virus_max_len, ab_max_len, freeze_mode = freeze_mode)
    cv_mean_acc, cv_mean_mcc, cv_mean_auc = log_metrics_per_cv_antibody(cv_metrics, antibody)
    return cv_mean_acc, cv_mean_mcc, cv_mean_auc

def test_optimized_antibodies(experiment_name, tags = None, model_trial_name = '', freeze_mode = FREEZE_ANTIBODY_AND_EMBEDDINGS, pretrain_epochs = None):
    setup_logging()
    experiment_name += f' {model_trial_name}'
    experiment_id = get_experiment(experiment_name)
    with mlflow.start_run(experiment_id = experiment_id, tags = tags):
        acc, mcc, auc = [], [], []
        for antibody in ANTIBODIES_LIST:
            cv_mean_acc, cv_mean_mcc, cv_mean_auc = test_optimized_antibody(antibody, model_trial_name, freeze_mode, pretrain_epochs)
            acc.append(cv_mean_acc)
            mcc.append(cv_mean_mcc)
            auc.append(cv_mean_auc)
        global_acc = statistics.mean(acc)
        global_mcc = statistics.mean(mcc)
        global_auc = statistics.mean(auc)
        logging.info(f'Global ACC {global_acc}')
        logging.info(f'Global MCC {global_mcc}')
        logging.info(f'Global AUC {global_auc}')
        mlflow.log_metrics({ 'global_acc': global_acc, 'global_mcc': global_mcc, 'global_auc': global_auc })
    dump_json({'finished': 'true'}, 'finished.json')

if __name__ == '__main__':
    tags = {
        'freeze': 'antb and embed',
        'trial': '330',
        'validation': 'uniform',
        'prune': 'treshold 0.05',
        'pretrain_epochs': '10'
    }
    test_optimized_antibodies('ICERI V2', tags = tags, model_trial_name = 'uniform_330', pretrain_epochs = 10)