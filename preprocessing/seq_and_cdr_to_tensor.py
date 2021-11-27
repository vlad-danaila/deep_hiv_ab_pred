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
    for ab, cdr_data in cdr_dict.items():
        ab_light_cdrs = cdr_data[AB_LIGHT]
        ab_heavy_cdrs = cdr_data[AB_HEAVY]
        # TODO call ab_cdrs_to_tensor

def ab_cdrs_to_tensor(cdr_1_size, cdr_2_size, cdr_3_size):
    # TODO
    pass

# TODO rezolva bug, ai uitat sa pui indexi la corecti
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

def parse_catnap_sequences_to_embeddings(virus_kmer_len, virus_kmer_stride):
    virus_seq = read_virus_fasta_sequences(VIRUS_FILE, virus_kmer_len, virus_kmer_stride)
    virus_pngs_mask = read_virus_pngs_mask(VIRUS_WITH_PNGS_FILE, virus_kmer_len, virus_kmer_stride)
    antibody_cdrs = read_cdrs()
    return virus_seq, virus_pngs_mask, antibody_cdrs

if __name__ == '__main__':
    find_cdr_tensor_sizes()

    # virus_seq, virus_pngs_mask, antibody_cdrs = parse_catnap_sequences_to_embeddings(51, 25)
