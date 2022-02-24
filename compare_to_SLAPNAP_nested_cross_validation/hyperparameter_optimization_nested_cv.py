from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import HYPERPARAM_FOLDER_ANTIBODIES
from deep_hiv_ab_pred.util.tools import read_json_file, dump_json, get_experiment
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import COMPARE_SPLITS_FOR_RAWI, MODELS_FOLDER, \
    CV_FOLDS_TRIM, N_TRIALS, PRUNE_TREHOLD, ANTIBODIES_LIST, FREEZE_ANTIBODY_AND_EMBEDDINGS, FREEZE_ALL_BUT_LAST_LAYER
from deep_hiv_ab_pred.global_constants import DEFAULT_CONF
from os.path import join
import mlflow
import statistics
from deep_hiv_ab_pred.util.metrics import log_metrics_per_test_set_antibody
from deep_hiv_ab_pred.util.logging import setup_logging
import logging
from deep_hiv_ab_pred.compare_to_SLAPNAP_nested_cross_validation.constants import COMPARE_SPLITS_FOR_SLAPNAP_NESTED_CV_1
from deep_hiv_ab_pred.compare_to_Rawi_gbm.hyperparameter_optimisation import optimize_hyperparameters, get_data, add_properties_from_base_config
from deep_hiv_ab_pred.compare_to_Rawi_gbm.train_evaluate import TRAIN, TEST, FREEZE_ALL
from deep_hiv_ab_pred.preprocessing.pytorch_dataset import AssayDataset, zero_padding
import torch as t
from deep_hiv_ab_pred.model.FC_GRU_ATT import get_FC_GRU_ATT_model
from deep_hiv_ab_pred.training.training import train_with_frozen_antibody_and_embedding, eval_network

def evaluate_on_test_data(antibody, splits, catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq,
                          antibody_heavy_seq, freeze_mode = FREEZE_ANTIBODY_AND_EMBEDDINGS):
    train_ids, test_ids = splits[TRAIN], splits[TEST]
    train_assays = [a for a in catnap if a[0] in train_ids]
    test_assays = [a for a in catnap if a[0] in test_ids]
    assert len(train_assays) == len(train_ids) and len(test_assays) == len(test_ids)
    train_set = AssayDataset(train_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
    test_set = AssayDataset(test_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
    loader_train = t.utils.data.DataLoader(train_set, conf['BATCH_SIZE'], shuffle = True, collate_fn = zero_padding, num_workers = 0)
    loader_test = t.utils.data.DataLoader(test_set, len(test_set), shuffle = False, collate_fn = zero_padding, num_workers = 0)
    model = get_FC_GRU_ATT_model(conf)
    checkpoint = t.load(join(MODELS_FOLDER, f'model_{antibody}_pretrain.tar'))
    model.load_state_dict(checkpoint['model'])
    if freeze_mode == FREEZE_ANTIBODY_AND_EMBEDDINGS:
        _, _, metrics = train_with_frozen_antibody_and_embedding(
            model, conf, loader_train, loader_test, 0, 100, f'model_{antibody}', MODELS_FOLDER, False, log_every_epoch = False
        )
    elif freeze_mode == FREEZE_ALL:
        metrics = eval_network(model, loader_test)
    else:
        raise 'Must provide a freeze mode.'
    return metrics


def test_optimized_antibody(antibody, splits_file, model_trial_name = '', freeze_mode = FREEZE_ANTIBODY_AND_EMBEDDINGS, pretrain_epochs = None):
    mlflow.log_params({ 'n_trials': N_TRIALS, 'prune_trehold': PRUNE_TREHOLD })
    optimize_hyperparameters(antibody, splits_file, cv_folds_trim = CV_FOLDS_TRIM, n_trials = N_TRIALS, prune_trehold = PRUNE_TREHOLD,
                             model_trial_name = model_trial_name, freeze_mode = freeze_mode, pretrain_epochs = pretrain_epochs)
    all_splits, catnap, base_conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = get_data(splits_file)
    mlflow.log_artifact(DEFAULT_CONF, 'base_conf.json')
    conf = read_json_file(join(HYPERPARAM_FOLDER_ANTIBODIES, f'{antibody}.json'))
    mlflow.log_artifact(join(HYPERPARAM_FOLDER_ANTIBODIES, f'{antibody}.json'), f'{antibody} conf.json')
    conf = add_properties_from_base_config(conf, base_conf)
    metrics = evaluate_on_test_data(antibody, all_splits[antibody]['test'], catnap, conf,
                                         virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, freeze_mode = freeze_mode)
    acc, mcc, auc = log_metrics_per_test_set_antibody(metrics, antibody)
    return acc, mcc, auc

def test_optimized_antibodies(experiment_name, tags = None, model_trial_name = '',
                              freeze_mode = FREEZE_ANTIBODY_AND_EMBEDDINGS, pretrain_epochs = None, splits_file = COMPARE_SPLITS_FOR_RAWI):
    setup_logging()
    experiment_name += f' {model_trial_name}'
    experiment_id = get_experiment(experiment_name)
    with mlflow.start_run(experiment_id = experiment_id, tags = tags):
        acc, mcc, auc = [], [], []
        for antibody in ANTIBODIES_LIST:
            _acc, _mcc, _auc = test_optimized_antibody(antibody, splits_file, model_trial_name, freeze_mode, pretrain_epochs)
            acc.append(_acc)
            mcc.append(_mcc)
            auc.append(_auc)
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
        'model': 'fc_att_gru_1_layer',
        'splits': 'uniform',
        'input': 'props-only',
        'trial': '252',
        'prune': 'treshold 0.05',
        'pretrain_epochs': 100
    }
    test_optimized_antibodies('Trial 252', tags = tags, model_trial_name = 'fc_att_gru_trial_252',
                              pretrain_epochs = 100, splits_file = COMPARE_SPLITS_FOR_SLAPNAP_NESTED_CV_1)