from deep_hiv_ab_pred.train_full_catnap.constants import *
from deep_hiv_ab_pred.util.tools import read_yaml, read_json_file, dump_json
from deep_hiv_ab_pred.hyperparameters.constants import CONF_ICERI
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.train_full_catnap.constants import SPLITS_HOLD_OUT_ONE_CLUSTER, SPLITS_UNIFORM
from deep_hiv_ab_pred.preprocessing.sequences_to_embedding import parse_catnap_sequences_to_embeddings
from deep_hiv_ab_pred.train_full_catnap.create_splits_hold_out_one_cluster import print_phylogenetic_tree, \
    create_virus_to_cluster_mapping, train_val_test_splits

KMER_LEN = 'KMER_LEN'
KMER_STRIDE = 'KMER_STRIDE'
TRAIN = 'train'
VAL = 'val'
TEST = 'test'

if __name__ == '__main__':
    # For making sure the phylogenetic tree was created
    # print_phylogenetic_tree()
    conf = read_yaml(CONF_ICERI)
    virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap_sequences_to_embeddings(
        conf[KMER_LEN], conf[KMER_STRIDE]
    )
    assays = read_json_file(CATNAP_FLAT)
    splits_hold_out_one_cluster = read_json_file(SPLITS_HOLD_OUT_ONE_CLUSTER)
    test_set = splits_hold_out_one_cluster['test']
    assays_rest = [a for a in assays if a[0] not in test_set]
    virus_to_cluster, cluster_to_virus, clusters = create_virus_to_cluster_mapping(virus_seq, nclusters = 5)
    _, val_set = train_val_test_splits(assays_rest, virus_to_cluster, clusters, TRAIN_VAL_SPLIT)
    train_set = [a[0] for a in assays_rest if a[0] not in val_set]
    dump_json({'train': train_set, 'val': val_set, 'test': test_set}, SPLITS_UNIFORM)