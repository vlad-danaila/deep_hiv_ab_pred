from os.path import join

COMPARE_RAWI_FOLDER = join('deep_hiv_ab_pred', 'compare_to_Rawi_gbm')
RAWI_DATA = join(COMPARE_RAWI_FOLDER, 'Rawi_data.json')
COMPARE_SPLITS_FOR_RAWI = join(COMPARE_RAWI_FOLDER, 'splits_Rawi_comparison.json')
CATNAP_DATA = join(COMPARE_RAWI_FOLDER, 'catnap_classification.json')
MODELS_FOLDER = join(COMPARE_RAWI_FOLDER, 'models')
HYPERPARAM_FOLDER = join('deep_hiv_ab_pred', 'hyperparameters')

# Need to clone https://github.com/RedaRawi/bNAb-ReP.git
SEQ_FOLDER = r'C:\DOC\Workspace\bNAb-ReP\alignments'

HYPERPARAM_PRETRAIN = join(HYPERPARAM_FOLDER, 'hyperparameters_iceri_2021_v2_29_10_2021.json')