from deep_hiv_ab_pred.train_full_catnap.constants import SPLITS_UNIFORM, MODELS_FOLDER
from deep_hiv_ab_pred.training.training import train_network_n_times, eval_network
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.preprocessing.pytorch_dataset import AssayDataset, zero_padding
from deep_hiv_ab_pred.preprocessing.sequences import parse_catnap_sequences
from deep_hiv_ab_pred.util.tools import read_json_file, device
from deep_hiv_ab_pred.model.ICERI2021_v2 import ICERI2021Net_V2
import torch as t
import numpy as np
from deep_hiv_ab_pred.training.constants import ACCURACY, MATTHEWS_CORRELATION_COEFFICIENT, AUC
import mlflow
from os.path import join
import optuna
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import HYPERPARAM_PRETRAIN
from deep_hiv_ab_pred.train_full_catnap.train_hold_out_one_cluster import test

def log_metrics(metrics):
    print('Acc', metrics[ACCURACY])
    print('MCC', metrics[MATTHEWS_CORRELATION_COEFFICIENT])
    print('AUC', metrics[AUC])
    mlflow.log_metrics({
        f'acc': metrics[ACCURACY],
        f'mcc': metrics[MATTHEWS_CORRELATION_COEFFICIENT],
        f'auc': metrics[AUC]
    })

def train_on_uniform_splits(splits, catnap, conf, trial = None):
    virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap_sequences(
        conf['KMER_LEN_VIRUS'], conf['KMER_STRIDE_VIRUS'], conf['KMER_LEN_ANTB'], conf['KMER_STRIDE_ANTB']
    )
    metrics = []
    train_ids, val_ids = splits['train'], splits['val']
    train_assays = [a for a in catnap if a[0] in train_ids]
    val_assays = [a for a in catnap if a[0] in val_ids]
    train_set = AssayDataset(train_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
    val_set = AssayDataset(val_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
    loader_train = t.utils.data.DataLoader(train_set, conf['BATCH_SIZE'], shuffle = True, collate_fn = zero_padding, num_workers = 0)
    loader_val = t.utils.data.DataLoader(val_set, conf['BATCH_SIZE'], shuffle = False, collate_fn = zero_padding, num_workers = 0)


    log_metrics(metrics)
    return metrics

def main_train():
    # conf = read_yaml(CONF_ICERI_V2)
    conf = read_json_file(HYPERPARAM_PRETRAIN)
    splits = read_json_file(SPLITS_UNIFORM)
    catnap = read_json_file(CATNAP_FLAT)
    metrics = train_on_uniform_splits(splits, catnap, conf)

def main_test():
    # conf = read_yaml(CONF_ICERI_V2)
    conf = read_json_file(HYPERPARAM_PRETRAIN)
    splits = read_json_file(SPLITS_UNIFORM)
    catnap = read_json_file(CATNAP_FLAT)
    metrics = test(splits, catnap, conf)

if __name__ == '__main__':
    main_train()
    main_test()