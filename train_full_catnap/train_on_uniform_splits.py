from deep_hiv_ab_pred.train_full_catnap.constants import SPLITS_UNIFORM, MODELS_FOLDER
from deep_hiv_ab_pred.training.training import train_network_n_times, eval_network
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.preprocessing.pytorch_dataset import AssayDataset, zero_padding
from deep_hiv_ab_pred.preprocessing.sequences_to_embedding import parse_catnap_sequences_to_embeddings
from deep_hiv_ab_pred.util.tools import read_json_file
import torch as t
from deep_hiv_ab_pred.training.constants import ACCURACY, MATTHEWS_CORRELATION_COEFFICIENT, AUC
import mlflow
from deep_hiv_ab_pred.global_constants import DEFAULT_CONF, HYPERPARAM_FOLDER
from deep_hiv_ab_pred.train_full_catnap.train_hold_out_one_cluster import test
from deep_hiv_ab_pred.model.GRU_GRU import get_GRU_GRU_model
import logging
from deep_hiv_ab_pred.training.cv_pruner import CrossValidationPruner
from deep_hiv_ab_pred.util.logging import setup_logging
from deep_hiv_ab_pred.util.plotting import plot_epochs
from os.path import join

def log_metrics(metrics):
    logging.info(f'Acc {metrics[ACCURACY]}')
    logging.info(f'MCC {metrics[MATTHEWS_CORRELATION_COEFFICIENT]}')
    logging.info(f'AUC {metrics[AUC]}')
    mlflow.log_metrics({
        f'acc': metrics[ACCURACY],
        f'mcc': metrics[MATTHEWS_CORRELATION_COEFFICIENT],
        f'auc': metrics[AUC]
    })

def train_on_uniform_splits(splits, catnap, conf, pruner: CrossValidationPruner = None):
    virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap_sequences_to_embeddings(
        conf['KMER_LEN_VIRUS'], conf['KMER_STRIDE_VIRUS'], conf['KMER_LEN_ANTB'], conf['KMER_STRIDE_ANTB']
    )
    train_ids, val_ids = splits['train'], splits['val']
    train_assays = [a for a in catnap if a[0] in train_ids]
    val_assays = [a for a in catnap if a[0] in val_ids]
    train_set = AssayDataset(train_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
    val_set = AssayDataset(val_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
    loader_train = t.utils.data.DataLoader(train_set, conf['BATCH_SIZE'], shuffle = True, collate_fn = zero_padding, num_workers = 0)
    loader_val = t.utils.data.DataLoader(val_set, conf['BATCH_SIZE'], shuffle = False, collate_fn = zero_padding, num_workers = 0)
    model = get_GRU_GRU_model(conf)
    _, _, metrics = train_network_n_times(model, conf, loader_train, loader_val, None, conf['EPOCHS'], f'model', MODELS_FOLDER, pruner)
    log_metrics(metrics)
    return metrics

def inspect_performance_per_epocs(hyperparam_file, nb_epochs = None):
    setup_logging()
    conf = read_json_file(join(HYPERPARAM_FOLDER, hyperparam_file))
    splits = read_json_file(SPLITS_UNIFORM)
    catnap = read_json_file(CATNAP_FLAT)
    virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap_sequences_to_embeddings(
        conf['KMER_LEN_VIRUS'], conf['KMER_STRIDE_VIRUS'], conf['KMER_LEN_ANTB'], conf['KMER_STRIDE_ANTB']
    )
    test_ids = splits['test']
    train_assays = [a for a in catnap if a[0] not in test_ids]
    test_assays = [a for a in catnap if a[0] in test_ids]
    train_set = AssayDataset(train_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
    test_set = AssayDataset(test_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
    loader_train = t.utils.data.DataLoader(train_set, conf['BATCH_SIZE'], shuffle = True, collate_fn = zero_padding, num_workers = 0)
    loader_test = t.utils.data.DataLoader(test_set, conf['BATCH_SIZE'], shuffle = False, collate_fn = zero_padding, num_workers = 0)
    model = get_GRU_GRU_model(conf)
    model_name = 'model_test'
    epochs = nb_epochs if nb_epochs else conf['EPOCHS']
    train_metrics_list, test_metrics_list, last = train_network_n_times(
        model, conf, loader_train, loader_test, None, epochs, model_name, MODELS_FOLDER)
    mccs = [m[MATTHEWS_CORRELATION_COEFFICIENT] for m in test_metrics_list]
    best_mcc = max(mccs)
    ideal_epoch = mccs.index(best_mcc) + 1
    logging.info(f'Best MCC {best_mcc} at epoch {ideal_epoch}')
    plot_epochs(train_metrics_list, test_metrics_list)

def main_train():
    setup_logging()
    # conf = read_yaml(CONF_ICERI_V2)
    conf = read_json_file(DEFAULT_CONF)
    splits = read_json_file(SPLITS_UNIFORM)
    catnap = read_json_file(CATNAP_FLAT)
    metrics = train_on_uniform_splits(splits, catnap, conf)

def main_test():
    setup_logging()
    # conf = read_yaml(CONF_ICERI_V2)
    conf = read_json_file(DEFAULT_CONF)
    splits = read_json_file(SPLITS_UNIFORM)
    catnap = read_json_file(CATNAP_FLAT)
    metrics = test(splits, catnap, conf)

if __name__ == '__main__':
    main_train()
    main_test()