from deep_hiv_ab_pred.train_full_catnap.constants import *
from Bio import Phylo
from Bio.Cluster import kmedoids
import numpy as np
import collections
from deep_hiv_ab_pred.util.tools import read_yaml, read_json_file
from deep_hiv_ab_pred.hyperparameters.constants import CONF_ICERI
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.preprocessing.sequences import parse_catnap_sequences

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

if __name__ == '__main__':
    # For making sure the phylogenetic tree was created
    # print_phylogenetic_tree()
    conf = read_yaml(CONF_ICERI)
    virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap_sequences(
        conf[KMER_LEN], conf[KMER_STRIDE], conf[KMER_LEN], conf[KMER_STRIDE]
    )
    assays = read_json_file(CATNAP_FLAT)
    virus_to_cluster, cluster_to_virus, clusters = create_virus_to_cluster_mapping(virus_seq, nclusters = 5)
    assays_by_virus_cluster = group_assays_by_visrus_clusters(assays, virus_to_cluster)
