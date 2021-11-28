import numpy as np
import torch as t
from deep_hiv_ab_pred.preprocessing.aminoacids import amino_to_index
from deep_hiv_ab_pred.util.tools import device, read_json_file
from Bio import SeqIO
from deep_hiv_ab_pred.catnap.constants import VIRUS_FILE, VIRUS_WITH_PNGS_FILE, ANTIBODIES_LIGHT_FASTA_FILE, ANTIBODIES_HEAVY_FASTA_FILE
from deep_hiv_ab_pred.preprocessing.constants import LIGHT_ANTIBODY_TRIM, HEAVY_ANTIBODY_TRIM
from deep_hiv_ab_pred.preprocessing.sequences_to_embedding import read_virus_fasta_sequences, read_virus_pngs_mask
from deep_hiv_ab_pred.catnap.constants import AB_CDRS
from itertools import chain
import numpy as np

AB_LIGHT = 'ab_light'
AB_HEAVY = 'ab_heavy'

def read_cdrs():
    cdr_dict = read_json_file(AB_CDRS)
    cdr_tensors = {}
    tensor_sizes = find_cdr_tensor_sizes()
    for ab, cdr_data in cdr_dict.items():
        ab_light_cdrs = cdr_data[AB_LIGHT]
        ab_heavy_cdrs = cdr_data[AB_HEAVY]
        ab_cdrs_to_tensor(ab_light_cdrs, tensor_sizes[:3])
        ab_cdrs_to_tensor(ab_heavy_cdrs, tensor_sizes[3:])

def ab_cdrs_to_tensor(cdrs, tensor_sizes):
    sequences = [c[0] for c in cdrs]
    sequences_index = [
        t.tensor([amino_to_index[s] for s in seq], dtype=t.long, device = device)
        for seq in sequences
    ]
    pass

def find_cdr_tensor_sizes():
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
    len_light_cdr1 = cdrs[:, 1] - cdrs[:, 0]
    len_light_cdr2 = cdrs[:, 3] - cdrs[:, 2]
    len_light_cdr3 = cdrs[:, 5] - cdrs[:, 4]
    len_heavy_cdr1 = cdrs[:, 7] - cdrs[:, 6]
    len_heavy_cdr2 = cdrs[:, 9] - cdrs[:, 8]
    len_heavy_cdr3 = cdrs[:, 11] - cdrs[:, 10]
    return max(len_light_cdr1), max(len_light_cdr2), max(len_light_cdr3), max(len_heavy_cdr1), max(len_heavy_cdr2), max(len_heavy_cdr3)

def parse_catnap_sequences_to_embeddings(virus_kmer_len, virus_kmer_stride):
    virus_seq = read_virus_fasta_sequences(VIRUS_FILE, virus_kmer_len, virus_kmer_stride)
    virus_pngs_mask = read_virus_pngs_mask(VIRUS_WITH_PNGS_FILE, virus_kmer_len, virus_kmer_stride)
    antibody_cdrs = read_cdrs()
    return virus_seq, virus_pngs_mask, antibody_cdrs

if __name__ == '__main__':
    find_cdr_tensor_sizes()

    # virus_seq, virus_pngs_mask, antibody_cdrs = parse_catnap_sequences_to_embeddings(51, 25)