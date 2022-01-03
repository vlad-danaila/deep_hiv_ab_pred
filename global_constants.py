from os.path import join

#EMBEDDING = 'LEARNED'
# EMBEDDING = 'ONE-HOT'
# EMBEDDING = 'ONE-HOT-AND-PROPS'
EMBEDDING = 'PROPS-ONLY'

HYPERPARAM_FOLDER = join('deep_hiv_ab_pred', 'hyperparameters')

DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_transf_6cdr_noam_100_epochs_uniform_props_only_multivariate_tpe_no_prunner_trial_229.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_transf_6cdr_noam_100_epohs_uniform_props_only_multivariate_tpe_no_prunner_trial_382.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_transf_6cdr_noam_100_epochs_uniform_props_only_multivariate_tpe_v2_trial_1169.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_transf_6cdr_noam_100_epochs_unifrom_props_only_multivariate_tpe_trial_16492.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_transf_6cdr_noam_100_epochs_uniform_props_only_multivariate_tpe_trial_606.json')

# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_transf_6cdr_adam_100_epochs_uniform_props_only_trial_415.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_transformers_example.json')

# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_6cdr_seq_only_fc_gru_uniform_props_only_1_layer_trial_95.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_6cdr_seq_only_fc_gru_uniform_props_only_1_layer_trial_402.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_6cdr_seq_only_fc_gru_uniform_props_only_1_layer_trial_294.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_6cdr_mask_all_feat_fc_gru_uniform_props_only_1_layer_trial_457.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_6cdr_mask_all_feat_fc_gru_uniform_props_only_1_layer_trial_503.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_6cdr_fc_gru_uniform_props_only_1_layer_trial_415.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_6cdr_fc_gru_uniform_props_only_1_layer_trial_207.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_cdr_fc_gru_uniform_props_only_1_layer_trial_253.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_cdr_fc_gru_uniform_props_only_1_layer_trial_417.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_fc_att_gru_uniform_one_hot_1_layer_trial_159.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_fc_att_gru_uniform_props_only_1_layer_trial_252.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_fc_att_gru_uniform_one_hot_and_props_1_layer_trial_213.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_iceri_v2_uniform_prune_treshold_05_trial_330.json')
# DEFAULT_CONF = join(HYPERPARAM_FOLDER, 'hyperparam_iceri_v2_hold_out_prune_treshold_01_trial_162.json')

INCLUDE_CDR_MASK_FEATURES = False
INCLUDE_CDR_POSITION_FEATURES = False

OUTPUT_AGGREGATE_MODE = 'LAST'
# OUTPUT_AGGREGATE_MODE = 'SUM'