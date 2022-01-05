from deep_hiv_ab_pred.train_full_catnap.constants import *
from Bio import Phylo
from Bio.Cluster import kmedoids
import numpy as np
import collections
from deep_hiv_ab_pred.util.tools import read_yaml, read_json_file, dump_json
from deep_hiv_ab_pred.hyperparameters.constants import CONF_ICERI
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT, ANTIBODIES_DETAILS_FILE
from deep_hiv_ab_pred.train_full_catnap.constants import TEST_SPLIT, SPLITS_HOLD_OUT_ONE_CLUSTER
from deep_hiv_ab_pred.preprocessing.sequences_to_embedding import parse_catnap_sequences_to_embeddings
import pandas as pd
import math
import random
import itertools

PRETRAINING = 'pretraining'
CV = 'cross_validation'
KMER_LEN = 'KMER_LEN'
KMER_STRIDE = 'KMER_STRIDE'
TRAIN = 'train'
TEST = 'test'

def print_phylogenetic_tree():
    tree = Phylo.read(VIRUS_TREE, "newick")
    tree.ladderize()
    Phylo.draw_ascii(tree)

def read_virus_distance_matrix(virus_seq):
    distance_matrix = []
    index_to_virus_id = {}
    count = 0
    skip_first_line = True
    with open(VIRUS_DISTANCE_MATRIX) as file:
        for line in file:
            if skip_first_line:
                skip_first_line = False
                continue
            line_split = line.split()
            # if True => the start of a new sequence in the distance matrix
            if not line.startswith(' '):
                virus_id = line_split[0].split('.')[-2]
                assert virus_id in virus_seq
                index_to_virus_id[count] = virus_id
                distance_matrix.append([float(s) for s in line_split[1:]])
                count += 1
            else:
                distance_matrix[-1] += [float(s) for s in line_split]
        assert count == len(virus_seq)
    return np.array(distance_matrix), index_to_virus_id

def create_virus_to_cluster_mapping(virus_seq, nclusters = NB_VIRUS_CLUSTERS):
    distance_matrix, index_to_virus_id = read_virus_distance_matrix(virus_seq)
    # Create k-medoids clustering from distance matrix
    clusterid, error, nfound = kmedoids(distance_matrix, nclusters = nclusters, npass = 1000)
    medoids, counts = np.unique(clusterid, return_counts=True)
    print(f'Cluster medoids: {medoids}, cluster sizes: {counts}')
    virus_to_cluster = dict()
    cluster_to_virus = collections.defaultdict(lambda: set())
    for i, cluster in enumerate(clusterid):
        virus = index_to_virus_id[i]
        virus_to_cluster[virus] = cluster
        cluster_to_virus[cluster].add(virus)
    return virus_to_cluster, cluster_to_virus, set(medoids)

def group_assays_by_visrus_clusters(assays, virus_to_cluster):
    assays_by_virus = collections.defaultdict(lambda: set())
    for assay in assays:
        id, antibdy, virus, outcome = assay
        virus_cluster = virus_to_cluster[virus]
        assays_by_virus[virus_cluster].add(tuple(assay))
    return assays_by_virus

def group_assays_by_antibody_type(assays):
    antibody_details_df = pd.read_csv(ANTIBODIES_DETAILS_FILE, sep = '\t')
    assays_by_antibody_type = collections.defaultdict(lambda: set())
    for assay in assays:
        id, antibody, virus, outcome = assay
        is_antibody_name = antibody_details_df['Name'] == antibody
        antibody_details = antibody_details_df[is_antibody_name]
        antibody_type = antibody_details['Type'].values[0]
        assays_by_antibody_type[antibody_type].add(tuple(assay))
    return assays_by_antibody_type

def train_val_test_splits(assays, virus_to_cluster, clusters, test_split = TEST_SPLIT):
    assays_by_antibody_type = group_assays_by_antibody_type(assays)
    test_set = []
    train_validation_sets = {cluster:[] for cluster in clusters}

    for antibody_type, assays_per_antibody in assays_by_antibody_type.items():
        assays_by_clusters = group_assays_by_visrus_clusters(assays_per_antibody, virus_to_cluster)
        # Gather test set
        for virus_cluster, assays_per_cluster in assays_by_clusters.items():
            nb_test_sampled = math.ceil(test_split * len(assays_per_cluster))
            test_sampled = set(random.sample(assays_per_cluster, nb_test_sampled))
            test_set.extend([s[0] for s in test_sampled])
            # assays_by_clusters[virus_cluster] = assays_per_cluster - test_sampled
            assays_per_cluster = assays_per_cluster - test_sampled
            # Gather cross validation splits (for training + validation)
            train_validation_sets[virus_cluster].extend([a[0] for a in assays_per_cluster])

    splits_lenghts = [len(train_validation_sets[c]) for c in train_validation_sets]
    assert len(assays) == sum(splits_lenghts) + len(test_set)
    return train_validation_sets, test_set

if __name__ == '__main__':
    # For making sure the phylogenetic tree was created
    # print_phylogenetic_tree()
    conf = read_yaml(CONF_ICERI)
    virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, ab_to_types = parse_catnap_sequences_to_embeddings(
        conf[KMER_LEN], conf[KMER_STRIDE]
    )
    assays = read_json_file(CATNAP_FLAT)
    virus_to_cluster, cluster_to_virus, clusters = create_virus_to_cluster_mapping(virus_seq, nclusters = 5)
    train_validation_sets, test_set = train_val_test_splits(assays, virus_to_cluster, clusters)
    splits = { 'test': test_set, 'cv': [] }
    train_and_val_assays = set(itertools.chain(*train_validation_sets.values()))
    for cluster_id, val_assays in train_validation_sets.items():
        train_assays = train_and_val_assays - set(val_assays)
        assert len(train_and_val_assays) == len(train_assays) + len(val_assays)
        splits['cv'].append({'train': list(train_assays), 'val': val_assays})
    dump_json(splits, SPLITS_HOLD_OUT_ONE_CLUSTER)
