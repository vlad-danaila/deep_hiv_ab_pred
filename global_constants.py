from os.path import join

#EMBEDDING = 'LEARNED'
# EMBEDDING = 'ONE-HOT'
# EMBEDDING = 'ONE-HOT-AND-PROPS'
EMBEDDING = 'PROPS-ONLY'

HYPERPARAM_FOLDER = join('deep_hiv_ab_pred', 'hyperparameters')

# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_fc_att_gru_uniform_props_only_1_layer_multivariate_tpe_no_prunner_trial_319.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_fc_att_gru_uniform_one_hot_1_layer_trial_159.json')
DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_fc_att_gru_uniform_props_only_1_layer_trial_252.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_fc_att_gru_uniform_one_hot_and_props_1_layer_trial_213.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_iceri_v2_uniform_prune_treshold_05_trial_330.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_iceri_v2_hold_out_prune_treshold_01_trial_162.json')