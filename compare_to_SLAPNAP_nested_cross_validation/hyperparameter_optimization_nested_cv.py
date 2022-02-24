from deep_hiv_ab_pred.compare_to_Rawi_gbm.hyperparameter_optimisation import test_optimized_antibodies
from deep_hiv_ab_pred.compare_to_SLAPNAP_nested_cross_validation.constants import COMPARE_SPLITS_FOR_SLAPNAP_NESTED_CV

if __name__ == '__main__':
    tags = {
        'freeze': 'antb and embed',
        'model': 'fc_att_gru_1_layer',
        'splits': 'uniform',
        'input': 'props-only',
        'trial': '252',
        'prune': 'treshold 0.05',
        'pretrain_epochs': 100
    }
    test_optimized_antibodies('Trial 252', tags = tags, model_trial_name = 'fc_att_gru_trial_252', pretrain_epochs = 100, splits_file = COMPARE_SPLITS_FOR_SLAPNAP_NESTED_CV)