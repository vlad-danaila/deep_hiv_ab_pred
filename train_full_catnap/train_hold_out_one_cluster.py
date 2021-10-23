from deep_hiv_ab_pred.util.tools import read_json_file, read_yaml
from deep_hiv_ab_pred.train_full_catnap.constants import SPLITS_HOLD_OUT_ONE_CLUSTER
from deep_hiv_ab_pred.training.training import train_network, eval_network
from deep_hiv_ab_pred.hyperparameters.constants import CONF_ICERI
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.preprocessing.pytorch_dataset import AssayDataset, zero_padding
from deep_hiv_ab_pred.preprocessing.sequences import parse_catnap_sequences
from deep_hiv_ab_pred.util.tools import read_json_file, read_yaml, device, get_experiment
from deep_hiv_ab_pred.model.ICERI2021 import ICERI2021Net

KMER_LEN = 'KMER_LEN'
KMER_STRIDE = 'KMER_STRIDE'

def train(splits, catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq):

    for cv_fold in splits['cv']:
        train_ids, val_ids = cv_fold['train'], cv_fold['val']
        train_assays = [a for a in catnap if a[0] in train_ids]
        val_assays = [a for a in catnap if a[0] in val_ids]
        train_set = AssayDataset(train_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        val_set = AssayDataset(val_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        model = ICERI2021Net(conf).to(device)
        print(len(train_set), len(val_set))

if __name__ == '__main__':
    conf = read_yaml(CONF_ICERI)
    splits = read_json_file(SPLITS_HOLD_OUT_ONE_CLUSTER)
    catnap = read_json_file(CATNAP_FLAT)
    virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap_sequences(
        conf[KMER_LEN], conf[KMER_STRIDE], conf[KMER_LEN], conf[KMER_STRIDE]
    )
    metrics = train(splits, catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq)