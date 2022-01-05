import random
from deep_hiv_ab_pred.training.cv_pruner import CrossValidationPruner
from deep_hiv_ab_pred.train_full_catnap.constants import SPLITS_HOLD_OUT_ONE_CLUSTER, MODELS_FOLDER
from deep_hiv_ab_pred.training.training import train_network_n_times, eval_network
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.preprocessing.pytorch_dataset import AssayDataset, zero_padding
from deep_hiv_ab_pred.preprocessing.sequences_to_embedding import parse_catnap_sequences_to_embeddings
from deep_hiv_ab_pred.util.tools import read_json_file
from deep_hiv_ab_pred.model.FC_GRU_ATT import get_FC_GRU_ATT_model
import torch as t
import numpy as np
from deep_hiv_ab_pred.training.constants import ACCURACY, MATTHEWS_CORRELATION_COEFFICIENT
import mlflow
from os.path import join
from deep_hiv_ab_pred.global_constants import DEFAULT_CONF
import logging
from deep_hiv_ab_pred.util.logging import setup_logging
from deep_hiv_ab_pred.util.metrics import log_test_metrics

def log_cv_metrics(cv_metrics):
    cv_metrics = np.array(cv_metrics)
    for cv_fold in range(len(cv_metrics)):
        mlflow.log_metrics({
            f'cv{cv_fold} acc': cv_metrics[cv_fold][ACCURACY],
            f'cv{cv_fold} mcc': cv_metrics[cv_fold][MATTHEWS_CORRELATION_COEFFICIENT]
        })
    cv_mean_acc = cv_metrics[:, ACCURACY].mean()
    cv_std_acc = cv_metrics[:, ACCURACY].std()
    cv_mean_mcc = cv_metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].mean()
    cv_std_mcc = cv_metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].std()
    logging.info(f'CV Mean Acc {cv_mean_acc} CV Std Acc {cv_std_acc}')
    logging.info(f'CV Mean MCC {cv_mean_mcc} CV Std MCC {cv_std_mcc}')
    mlflow.log_metrics({
        f'cv mean acc': cv_mean_acc,
        f'cv std acc': cv_std_acc,
        f'cv mean mcc': cv_mean_mcc,
        f'cv std mcc': cv_std_mcc
    })

# DEPRECATED
def train_hold_out_one_cluster(splits, catnap, conf, pruner: CrossValidationPruner = None):
    virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, ab_to_types = parse_catnap_sequences_to_embeddings(
        conf['KMER_LEN_VIRUS'], conf['KMER_STRIDE_VIRUS']
    )
    cv_metrics = []
    cv_folds = list(range(len(splits['cv'])))
    random.shuffle(cv_folds)
    for i in cv_folds:
        cv_fold = splits['cv'][i]
        train_ids, val_ids = cv_fold['train'], cv_fold['val']
        train_assays = [a for a in catnap if a[0] in train_ids]
        val_assays = [a for a in catnap if a[0] in val_ids]
        train_set = AssayDataset(train_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        val_set = AssayDataset(val_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        loader_train = t.utils.data.DataLoader(train_set, conf['BATCH_SIZE'], shuffle = True, collate_fn = zero_padding, num_workers = 0)
        loader_val = t.utils.data.DataLoader(val_set, conf['BATCH_SIZE'], shuffle = False, collate_fn = zero_padding, num_workers = 0)
        model = get_FC_GRU_ATT_model(conf)
        _, _, best = train_network_n_times(model, conf, loader_train, loader_val, i, conf['EPOCHS'], f'model_cv_{i}', MODELS_FOLDER, pruner)
        cv_metrics.append(best)
    log_cv_metrics(cv_metrics)
    return cv_metrics

# DEPRECATED
def test(splits, catnap, conf):
    virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, ab_to_types = parse_catnap_sequences_to_embeddings(
        conf['KMER_LEN_VIRUS'], conf['KMER_STRIDE_VIRUS']
    )
    test_ids = splits['test']
    train_assays = [a for a in catnap if a[0] not in test_ids]
    test_assays = [a for a in catnap if a[0] in test_ids]
    train_set = AssayDataset(train_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
    test_set = AssayDataset(test_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
    loader_train = t.utils.data.DataLoader(train_set, conf['BATCH_SIZE'], shuffle = True, collate_fn = zero_padding, num_workers = 0)
    loader_test = t.utils.data.DataLoader(test_set, conf['BATCH_SIZE'], shuffle = False, collate_fn = zero_padding, num_workers = 0)
    model = get_FC_GRU_ATT_model(conf)
    model_name = 'model_test'
    _, _, best = train_network_n_times(model, conf, loader_train, None, None, conf['EPOCHS'], model_name, MODELS_FOLDER)
    checkpoint = t.load(join(MODELS_FOLDER, f'{model_name}.tar'))
    model.load_state_dict(checkpoint['model'])
    test_metrics = eval_network(model, loader_test)
    log_test_metrics(test_metrics)
    return test_metrics

def main_train():
    setup_logging()
    # conf = read_yaml(CONF_ICERI_V2)
    conf = read_json_file(DEFAULT_CONF)
    splits = read_json_file(SPLITS_HOLD_OUT_ONE_CLUSTER)
    catnap = read_json_file(CATNAP_FLAT)
    metrics = train_hold_out_one_cluster(splits, catnap, conf)

def main_test():
    setup_logging()
    # conf = read_yaml(CONF_ICERI_V2)
    conf = read_json_file(DEFAULT_CONF)
    splits = read_json_file(SPLITS_HOLD_OUT_ONE_CLUSTER)
    catnap = read_json_file(CATNAP_FLAT)
    metrics = test(splits, catnap, conf)

if __name__ == '__main__':
    main_train()
    main_test()