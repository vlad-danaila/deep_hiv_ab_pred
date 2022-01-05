import statistics
from deep_hiv_ab_pred.util.tools import read_json_file, read_yaml, device, get_experiment
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import COMPARE_SPLITS_FOR_RAWI, MODELS_FOLDER, \
    FREEZE_ANTIBODY_AND_EMBEDDINGS, FREEZE_ALL_BUT_LAST_LAYER, FREEZE_ALL
from deep_hiv_ab_pred.global_constants import DEFAULT_CONF
import torch as t
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.preprocessing.pytorch_dataset import AssayDataset, zero_padding
from deep_hiv_ab_pred.preprocessing.sequences_to_embedding import parse_catnap_sequences_to_embeddings
from deep_hiv_ab_pred.model.FC_GRU_ATT import get_FC_GRU_ATT_model
from deep_hiv_ab_pred.training.training import train_network, eval_network, train_with_frozen_antibody_and_embedding, train_with_fozen_net_except_of_last_layer
from os.path import join
from deep_hiv_ab_pred.training.constants import MATTHEWS_CORRELATION_COEFFICIENT
import mlflow
import optuna
from deep_hiv_ab_pred.util.metrics import log_metrics_per_cv_antibody
import logging

PRETRAINING = 'pretraining'
CV = 'cross_validation'
TRAIN = 'train'
TEST = 'test'

def pretrain_net(antibody, splits_pretraining, catnap, conf, virus_seq, virus_pngs_mask,
    antibody_light_seq, antibody_heavy_seq, pretrain_epochs, ab_to_types):

    pretraining_assays = [a for a in catnap if a[0] in splits_pretraining]
    rest_assays = [a for a in catnap if a[0] not in splits_pretraining]
    assert len(pretraining_assays) == len(splits_pretraining)
    pretrain_set = AssayDataset(pretraining_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
    val_set = AssayDataset(rest_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
    loader_pretrain = t.utils.data.DataLoader(pretrain_set, conf['BATCH_SIZE'], shuffle = True, collate_fn = zero_padding, num_workers = 0)
    loader_val = t.utils.data.DataLoader(val_set, conf['BATCH_SIZE'], shuffle = True, collate_fn = zero_padding, num_workers = 0)
    model = get_FC_GRU_ATT_model(conf)
    assert pretrain_epochs
    metrics_train_per_epochs, metrics_test_per_epochs, best = train_network(
        model, conf, loader_pretrain, loader_val, None, pretrain_epochs, f'model_{antibody}_pretrain', MODELS_FOLDER
    )
    return metrics_train_per_epochs, metrics_test_per_epochs, best

def cross_validate_antibody(antibody, splits_cv, catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq,
    antibody_heavy_seq, ab_to_types, trial = None, cv_folds_trim = 100, freeze_mode = FREEZE_ANTIBODY_AND_EMBEDDINGS):

    cv_metrics = []
    for (i, cv_fold) in enumerate(splits_cv[:cv_folds_trim]):
        train_ids, test_ids = cv_fold[TRAIN], cv_fold[TEST]
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
                model, conf, loader_train, loader_test, i, 100, f'model_{antibody}', MODELS_FOLDER, False, log_every_epoch = False
            )
        # deprecated
        elif freeze_mode == FREEZE_ALL_BUT_LAST_LAYER:
            _, _, metrics = train_with_fozen_net_except_of_last_layer(
                model, conf, loader_train, loader_test, i, conf['EPOCHS'], f'model_{antibody}', MODELS_FOLDER, False, log_every_epoch = False
            )
        elif freeze_mode == FREEZE_ALL:
            metrics = eval_network(model, loader_test)
        else:
            raise 'Must provide a freeze mode.'
        cv_metrics.append(metrics)
        if trial:
            trial.report(metrics[MATTHEWS_CORRELATION_COEFFICIENT], i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return cv_metrics

def train_net(experiment_name, tags = None, freeze_mode = FREEZE_ANTIBODY_AND_EMBEDDINGS):
    experiment_id = get_experiment(experiment_name)
    with mlflow.start_run(experiment_id = experiment_id, tags = tags):
        conf = read_json_file(DEFAULT_CONF)
        mlflow.log_artifact(DEFAULT_CONF, 'base_conf.json')
        all_splits = read_json_file(COMPARE_SPLITS_FOR_RAWI)
        # mlflow.log_artifact(COMPARE_SPLITS_FOR_RAWI)
        catnap = read_json_file(CATNAP_FLAT)
        # mlflow.log_artifact(CATNAP_FLAT)
        virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, ab_to_types = parse_catnap_sequences_to_embeddings(
            conf['KMER_LEN_VIRUS'], conf['KMER_STRIDE_VIRUS']
        )
        acc, mcc = [], []
        for i, (antibody, splits) in enumerate(all_splits.items()):
            logging.info(f'{i}. Antibody {antibody}')
            pretrain_net(antibody, splits[PRETRAINING], catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, 100, ab_to_types)
            cv_metrics = cross_validate_antibody(antibody, splits[CV], catnap, conf, virus_seq, virus_pngs_mask,
                antibody_light_seq, antibody_heavy_seq, ab_to_types, freeze_mode = freeze_mode)
            cv_mean_acc, cv_mean_mcc = log_metrics_per_cv_antibody(cv_metrics, antibody)
            acc.append(cv_mean_acc)
            mcc.append(cv_mean_mcc)
        global_acc = statistics.mean(acc)
        global_mcc = statistics.mean(mcc)
        logging.info('Global ACC ' + global_acc)
        logging.info('Global MCC ' + global_mcc)
        mlflow.log_metrics({ 'global_acc': global_acc, 'global_mcc': global_mcc })

if __name__ == '__main__':
    tags = {
        'note1': 'virus seq aligned unlike in ICERI2021',
        'note2': 'no parameters are freezed'
    }
    train_net('ICERI2021', tags, freeze_mode = FREEZE_ANTIBODY_AND_EMBEDDINGS)