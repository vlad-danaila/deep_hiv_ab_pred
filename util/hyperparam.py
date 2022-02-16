import optuna
from os import listdir, sep
from os.path import join
from deep_hiv_ab_pred.util.tools import read_json_file, dump_json
from deep_hiv_ab_pred.global_constants import DEFAULT_CONF
from copy import deepcopy
from deep_hiv_ab_pred.compare_to_Rawi_gbm.hyperparameter_optimisation import add_properties_from_base_config

def read_hyperparameters_for_antibodies(abntibodies_hyperparam_folder, pretraining_hyperparameters_file):
    pretrainig_hyperparams = read_json_file(pretraining_hyperparameters_file)
    hyperparameters = { 'pretrain': pretrainig_hyperparams }
    for file in listdir(abntibodies_hyperparam_folder):
        if not file.endswith('.db'):
            continue
        study_name = '.'.join(file.split('.')[:-1])
        sotrage_file = f'sqlite:///{join(abntibodies_hyperparam_folder, file)}'
        study = optuna.create_study(study_name = study_name, direction = 'maximize', storage = sotrage_file, load_if_exists = True)
        trials = [t for t in study.get_trials() if t.state == optuna.trial.TrialState.COMPLETE]
        best_trial = max(trials, key = lambda t: t.value)
        ab_hyperparams = add_properties_from_base_config(best_trial.params, deepcopy(pretrainig_hyperparams))
        antibody = study_name.split('_')[-1]
        hyperparameters[antibody] = ab_hyperparams
    return hyperparameters

if __name__ == '__main__':
    abntibodies_hyperparam_folder = 'C:/DOC/Articol HIV Antibodies/Experiments/Experiments/Compare Rawi/Compare Rawi FC ATT GRU 1 layer trial 252 props only'
    pretraining_hyperparameters_file = DEFAULT_CONF
    hyperparams = read_hyperparameters_for_antibodies(abntibodies_hyperparam_folder, pretraining_hyperparameters_file)
    dump_path = r'C:\DOC\Workspace\HIV Article\deep_hiv_ab_pred\hyperparameters\specific_antibodies\complete\\' + DEFAULT_CONF.split(sep)[-1]
    dump_json(hyperparams, dump_path)