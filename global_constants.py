from os.path import join

#EMBEDDING = 'LEARNED'
EMBEDDING = 'ONE-HOT'
# EMBEDDING = 'ONE-HOT-AND-PROPS'
# EMBEDDING = 'PROPS-ONLY'

HYPERPARAM_FOLDER = join('deep_hiv_ab_pred', 'hyperparameters')

DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_iceri_v2_uniform_prune_treshold_05_trial_330.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_iceri_v2_hold_out_prune_treshold_01_trial_162.json')

# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_gru_gru_props_onlu_trial_367.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_gru_gru_props_onlu_trial_151.json')

#DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_gru_gru_one_hot_trial_356.json')
