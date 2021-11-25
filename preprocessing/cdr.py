from deep_hiv_ab_pred.catnap.constants import PARATOME_AB_LIGHT_CDR, PARATOME_AB_HEAVY_CDR, VIRUS_FILE, VIRUS_WITH_PNGS_FILE
from deep_hiv_ab_pred.preprocessing.aminoacids import amino_to_index
import torch as t
from deep_hiv_ab_pred.util.tools import device
from deep_hiv_ab_pred.preprocessing.sequences_to_embedding import read_virus_fasta_sequences, read_virus_pngs_mask

def read_cdr_data(file_path):
    antibody_seq_dict = {}
    with open(file_path) as f:
        for line in f:
            line_split = line.split()
            if len(line_split) == 5:
                antb, chain, cdr_1, cdr_2, cdr_3 = tuple(line.split())
                antibody_id = antb.split('_')[0]
                cdr_1 = t.tensor([amino_to_index[a] for a in cdr_1], dtype=t.long, device = device)
                cdr_2 = t.tensor([amino_to_index[a] for a in cdr_2], dtype=t.long, device = device)
                cdr_3 = t.tensor([amino_to_index[a] for a in cdr_3], dtype=t.long, device = device)
                antibody_seq_dict[antibody_id] = cdr_1, cdr_2, cdr_3
    return antibody_seq_dict

def parse_catnap_sequences_to_embeddings(virus_kmer_len, virus_kmer_stride):
    virus_seq = read_virus_fasta_sequences(VIRUS_FILE, virus_kmer_len, virus_kmer_stride)
    virus_pngs_mask = read_virus_pngs_mask(VIRUS_WITH_PNGS_FILE, virus_kmer_len, virus_kmer_stride)
    antibody_light_seq = read_cdr_data(PARATOME_AB_LIGHT_CDR)
    antibody_heavy_seq = read_cdr_data(PARATOME_AB_HEAVY_CDR)
    return virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq

if __name__ == '__main__':
    virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap_sequences_to_embeddings(51, 25)