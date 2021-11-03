from deep_hiv_ab_pred.train_full_catnap.constants import SPLITS_HOLD_OUT_ONE_CLUSTER, MODELS_FOLDER
from deep_hiv_ab_pred.training.training import train_network_n_times, eval_network
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.preprocessing.pytorch_dataset import AssayDataset, zero_padding
from deep_hiv_ab_pred.preprocessing.sequences import parse_catnap_sequences
from deep_hiv_ab_pred.util.tools import read_json_file, device
from deep_hiv_ab_pred.model.ICERI2021_v2 import ICERI2021Net_V2
import torch as t
import numpy as np
from deep_hiv_ab_pred.training.constants import ACCURACY, MATTHEWS_CORRELATION_COEFFICIENT
import mlflow
from os.path import join
import optuna
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import HYPERPARAM_PRETRAIN
from deep_hiv_ab_pred.train_full_catnap.train_hold_out_one_cluster import test

def train_on_uniform_splits(splits, catnap, conf):
    # TODO
    pass

def main_train():
    # conf = read_yaml(CONF_ICERI_V2)
    conf = read_json_file(HYPERPARAM_PRETRAIN)
    splits = read_json_file(SPLITS_HOLD_OUT_ONE_CLUSTER)
    catnap = read_json_file(CATNAP_FLAT)
    metrics = train_on_uniform_splits(splits, catnap, conf)

def main_test():
    # conf = read_yaml(CONF_ICERI_V2)
    conf = read_json_file(HYPERPARAM_PRETRAIN)
    splits = read_json_file(SPLITS_HOLD_OUT_ONE_CLUSTER)
    catnap = read_json_file(CATNAP_FLAT)
    metrics = test(splits, catnap, conf)

if __name__ == '__main__':
    main_train()
    main_test()