import optuna
from os import listdir
from os.path import join

def read_hyperparameters_for_antibodies(abntibodies_hyperparam_folder, pretraining_hyperparameters_file):
    hyperparameters = {}
    for file in listdir(abntibodies_hyperparam_folder):
        if not file.endswith('.db'):
            continue
        study_name = '.'.join(file.split('.')[:-1])
        study = optuna.create_study(study_name = study_name, direction = 'maximize', storage = f'sqlite:///{join(abntibodies_hyperparam_folder, file)}', load_if_exists = True)
        trials = [t for t in study.get_trials() ]
        print(trials)
    return hyperparameters

if __name__ == '__main__':
    abntibodies_hyperparam_folder = 'C:/DOC/Articol HIV Antibodies/Experiments/Experiments/Compare Rawi/Compare Rawi FC ATT GRU 1 layer trial 252 props only'
    hyperparams = read_hyperparameters_for_antibodies(abntibodies_hyperparam_folder, None)