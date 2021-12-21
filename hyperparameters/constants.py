from os.path import join

HYPERPARAMETERS_FOLDER = join('deep_hiv_ab_pred','hyperparameters')

CONF_ICERI = join(HYPERPARAMETERS_FOLDER, 'hyperparameters_iceri_2021.yml')
CONF_ICERI_V2 = join(HYPERPARAMETERS_FOLDER, 'hyperparameters_iceri_2021_v2.yml')

# INITIAL_CONF_TRANS = join(HYPERPARAMETERS_FOLDER, 'hyperparam_transformers_example.json')
INITIAL_CONF_TRANS = join(HYPERPARAMETERS_FOLDER, 'hyperparam_transf_6cdr_noam_100_epochs_uniform_props_only_multivariate_tpe_v2_trial_1169.json')