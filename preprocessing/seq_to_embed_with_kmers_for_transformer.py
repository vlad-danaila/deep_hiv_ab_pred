import numpy as np
from deep_hiv_ab_pred.preprocessing.aminoacids import amino_to_index
from Bio import SeqIO
from deep_hiv_ab_pred.catnap.constants import VIRUS_FILE, VIRUS_WITH_PNGS_FILE, ANTIBODIES_LIGHT_FASTA_FILE, ANTIBODIES_HEAVY_FASTA_FILE
from deep_hiv_ab_pred.preprocessing.sequences_to_embedding import read_virus_fasta_sequences, read_virus_pngs_mask
from deep_hiv_ab_pred.catnap.parse_cdr_reports import get_id_to_seq_mapping_from_fasta_file as ab_to_seq
from deep_hiv_ab_pred.catnap.parse_cdr_reports import assays_abs

def antibody_relevant_sites():
    light_ab_cdr_possible_sites = ((17, 77), (84, 133))
    heavy_ab_cdr_possible_sites = ((13, 79), (83, 135))
    return light_ab_cdr_possible_sites, heavy_ab_cdr_possible_sites

def parse_ab_sites():
    abs_in_assays = assays_abs()
    selected_ab_seq = {}
    ab_seq_light = ab_to_seq(ANTIBODIES_LIGHT_FASTA_FILE)
    ab_seq_heavy = ab_to_seq(ANTIBODIES_HEAVY_FASTA_FILE)
    light_cdr_possible_sites, heavy_cdr_possible_sites = antibody_relevant_sites()
    (light_cdr_1_2_begin, light_cdr_1_2_end), (light_cdr_3_begin, light_cdr_3_end) = light_cdr_possible_sites
    (heavy_cdr_1_2_begin, heavy_cdr_1_2_end), (heavy_cdr_3_begin, heavy_cdr_3_end) = heavy_cdr_possible_sites
    for ab in abs_in_assays:
        seq_light = ab_seq_light[ab]
        seq_heavy = ab_seq_heavy[ab]
        concatenated = seq_light[light_cdr_1_2_begin : light_cdr_1_2_end] \
                              + seq_light[light_cdr_3_begin : light_cdr_3_end] \
                              + seq_heavy[heavy_cdr_1_2_begin : heavy_cdr_1_2_end] \
                              + seq_heavy[heavy_cdr_3_begin : heavy_cdr_3_end]
        selected_ab_seq[ab] = np.array([ amino_to_index[a] for a in concatenated ])
    return selected_ab_seq

def parse_catnap_sequences_to_embeddings():
    virus_seq = read_virus_fasta_sequences(VIRUS_FILE)
    virus_pngs_mask = read_virus_pngs_mask(VIRUS_WITH_PNGS_FILE)
    abs = parse_ab_sites()

    ab_max_len = max((len(seq) for seq in abs.values()))
    virus_seq_max_len = max((len(seq) for seq in virus_seq.values()))
    pngs_max_len = max((len(seq) for seq in virus_pngs_mask.values()))
    virus_max_len = max(virus_seq_max_len, pngs_max_len)

    for ab_id, seq in abs.items():
        pad_amnt = ab_max_len - len(seq)
        seq = np.pad(seq, (0, pad_amnt), 'constant', constant_values = amino_to_index['X'])
        abs[ab_id] = seq

    for v in virus_seq:
        pad_amnt = virus_max_len - len(virus_seq[v])
        seq = np.pad(virus_seq[v], (0, pad_amnt), 'constant', constant_values = amino_to_index['X'])
        pngs = np.pad(virus_pngs_mask[v], (0, virus_max_len - len(virus_pngs_mask[v])), 'constant', constant_values = 0)
        virus_seq[v] = (seq, pngs)

    return virus_seq, abs, virus_max_len, ab_max_len

if __name__ == '__main__':
    virus_seq, abs, virus_max_len, ab_max_len = parse_catnap_sequences_to_embeddings()

