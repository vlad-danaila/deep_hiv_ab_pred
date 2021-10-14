from util.tools import read_json_file, read_yaml, device
from compare_to_Rawi_gbm.constants import COMPARE_SPLITS_FOR_RAWI
import torch as t
from catnap.constants import CATNAP_FLAT
from preprocessing.pytorch_dataset import AssayDataset, zero_padding
from preprocessing.sequences import parse_catnap
from hyperparameters.constants import CONF_ICERI
from model.ICERI2021 import ICERI2021Net
from training.training import train_network

PRETRAINING = 'pretraining'
CV = 'cross_validation'
KMER_LEN = 'KMER_LEN'
KMER_STRIDE = 'KMER_STRIDE'

def pretrain_net(antibody, splits):
    global catnap, conf
    pretraining_assays = [a for a in catnap if a[0] in splits[PRETRAINING]]
    assert len(pretraining_assays) == len(splits[PRETRAINING])
    virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap(
        conf[KMER_LEN], conf[KMER_STRIDE], conf[KMER_LEN], conf[KMER_STRIDE]
    )
    pretrain_set = AssayDataset(
        pretraining_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask
    )
    loader_pretrain = t.utils.data.DataLoader(
        pretrain_set, conf['BATCH_SIZE'], shuffle = True, collate_fn = zero_padding, num_workers = 0
    )
    model = ICERI2021Net(conf).to(device)
    _, _, best = train_network(model, conf, loader_pretrain, None, None, conf['EPOCHS'], 'model_pretrain')

if __name__ == '__main__':
    conf = read_yaml(CONF_ICERI)
    all_splits = read_json_file(COMPARE_SPLITS_FOR_RAWI)
    catnap = read_json_file(CATNAP_FLAT)
    for antibody, splits in all_splits.items():
        pretrain_net(antibody, splits)