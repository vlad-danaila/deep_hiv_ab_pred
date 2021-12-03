import math
from deep_hiv_ab_pred.util.tools import normalize, to_torch
import numpy as np
from deep_hiv_ab_pred.preprocessing.aminoacids import amino_to_index
from deep_hiv_ab_pred.util.tools import device, read_json_file
from Bio import SeqIO
from deep_hiv_ab_pred.catnap.constants import VIRUS_FILE, VIRUS_WITH_PNGS_FILE, ANTIBODIES_LIGHT_FASTA_FILE, ANTIBODIES_HEAVY_FASTA_FILE
from deep_hiv_ab_pred.preprocessing.constants import LIGHT_ANTIBODY_TRIM, HEAVY_ANTIBODY_TRIM
from deep_hiv_ab_pred.preprocessing.sequences_to_embedding import read_virus_fasta_sequences, read_virus_pngs_mask
from deep_hiv_ab_pred.catnap.constants import AB_CDRS
from itertools import chain

AB_LIGHT = 'ab_light'
AB_HEAVY = 'ab_heavy'

def read_cdrs(include_position_features = True):
    cdr_dict = read_json_file(AB_CDRS)
    cdr_arrays = {}
    tensor_sizes = find_cdr_lengths()
    cdr_positions = find_cdr_centers()
    cdr_positions_std = find_cdr_position_std()
    for ab, cdr_data in cdr_dict.items():
        abs = cdr_data[AB_LIGHT] + cdr_data[AB_HEAVY] # concatenate
        cdr_arrays[ab] = ab_cdrs_to_tensor(abs, tensor_sizes, cdr_positions, cdr_positions_std, include_position_features)
    return cdr_arrays

def ab_cdrs_to_tensor(abs, tensor_sizes, cdr_positions, cdr_positions_std, include_position_features):
    sequences = [c[0] for c in abs]
    sequences_index = [ [amino_to_index[s] for s in seq] for seq in sequences ]
    sequences_index = [
        np.pad(
            s,
            ((tensor_sizes[i] - len(s)) // 2, math.ceil((tensor_sizes[i] - len(s)) / 2)),
            'constant',
            constant_values = (amino_to_index['X'], amino_to_index['X'])
        )
        for (i, s) in enumerate(sequences_index)
    ]
    sequences_index = np.concatenate(sequences_index)
    if include_position_features:
        assert cdr_positions and cdr_positions_std
        positions = [
            normalize(c[1][0] + (c[1][1] - c[1][0])/2, cdr_positions[i], cdr_positions_std[i])
            for (i, c) in enumerate(abs)
        ]
        return sequences_index, np.array(positions)
    return sequences_index

def cdr_indexes():
    cdr_dict = read_json_file(AB_CDRS)
    cdrs = []
    for ab, cdr_data in cdr_dict.items():
        ab_light_cdrs = cdr_data[AB_LIGHT]
        ab_heavy_cdrs = cdr_data[AB_HEAVY]
        cdrs_light_indexes = [cdr[1] for cdr in ab_light_cdrs]
        cdrs_heavy_indexes = [cdr[1] for cdr in ab_heavy_cdrs]
        all_cdr_indexes = cdrs_light_indexes + cdrs_heavy_indexes
        all_cdr_indexes = list(chain(*all_cdr_indexes))
        cdrs.append(all_cdr_indexes)
    cdrs = np.array(cdrs)
    return cdrs

def find_cdr_lengths():
    cdrs = cdr_indexes()
    len_light_cdr1 = cdrs[:, 1] - cdrs[:, 0]
    len_light_cdr2 = cdrs[:, 3] - cdrs[:, 2]
    len_light_cdr3 = cdrs[:, 5] - cdrs[:, 4]
    len_heavy_cdr1 = cdrs[:, 7] - cdrs[:, 6]
    len_heavy_cdr2 = cdrs[:, 9] - cdrs[:, 8]
    len_heavy_cdr3 = cdrs[:, 11] - cdrs[:, 10]
    return max(len_light_cdr1) + 1, max(len_light_cdr2) + 1, max(len_light_cdr3) + 1,\
           max(len_heavy_cdr1) + 1, max(len_heavy_cdr2) + 1, max(len_heavy_cdr3) + 1

def find_cdr_centers():
    cdrs = cdr_indexes()
    center_light_cdr1 = np.mean(cdrs[:, 0] + (cdrs[:, 1] - cdrs[:, 0]) / 2)
    center_light_cdr2 = np.mean(cdrs[:, 2] + (cdrs[:, 3] - cdrs[:, 2]) / 2)
    center_light_cdr3 = np.mean(cdrs[:, 4] + (cdrs[:, 5] - cdrs[:, 4]) / 2)
    center_heavy_cdr1 = np.mean(cdrs[:, 6] + (cdrs[:, 7] - cdrs[:, 6]) / 2)
    center_heavy_cdr2 = np.mean(cdrs[:, 8] + (cdrs[:, 9] - cdrs[:, 8]) / 2)
    center_heavy_cdr3 = np.mean(cdrs[:, 10] + (cdrs[:, 11] - cdrs[:, 10]) / 2)
    return center_light_cdr1, center_light_cdr2, center_light_cdr3, center_heavy_cdr1, center_heavy_cdr2, center_heavy_cdr3

def find_cdr_position_std():
    cdrs = cdr_indexes()
    std_light_cdr1 = np.std(cdrs[:, 0] + (cdrs[:, 1] - cdrs[:, 0]) / 2)
    std_light_cdr2 = np.std(cdrs[:, 2] + (cdrs[:, 3] - cdrs[:, 2]) / 2)
    std_light_cdr3 = np.std(cdrs[:, 4] + (cdrs[:, 5] - cdrs[:, 4]) / 2)
    std_heavy_cdr1 = np.std(cdrs[:, 6] + (cdrs[:, 7] - cdrs[:, 6]) / 2)
    std_heavy_cdr2 = np.std(cdrs[:, 8] + (cdrs[:, 9] - cdrs[:, 8]) / 2)
    std_heavy_cdr3 = np.std(cdrs[:, 10] + (cdrs[:, 11] - cdrs[:, 10]) / 2)
    return std_light_cdr1, std_light_cdr2, std_light_cdr3, std_heavy_cdr1, std_heavy_cdr2, std_heavy_cdr3

def parse_catnap_sequences_to_embeddings(virus_kmer_len, virus_kmer_stride):
    virus_seq = read_virus_fasta_sequences(VIRUS_FILE, virus_kmer_len, virus_kmer_stride)
    virus_pngs_mask = read_virus_pngs_mask(VIRUS_WITH_PNGS_FILE, virus_kmer_len, virus_kmer_stride)
    antibody_cdrs = read_cdrs()
    return virus_seq, virus_pngs_mask, antibody_cdrs

if __name__ == '__main__':
    cdr_dict = read_cdrs()
    print('done')

    # virus_seq, virus_pngs_mask, antibody_cdrs = parse_catnap_sequences_to_embeddings(51, 25)
