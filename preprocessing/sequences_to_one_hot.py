from deep_hiv_ab_pred.preprocessing.sequences_to_embedding import read_virus_pngs_mask
from deep_hiv_ab_pred.preprocessing.sequences_to_embedding import aminoacids_len, amino_to_index, aminoacids
import torch as t
from deep_hiv_ab_pred.util.tools import device
from Bio import SeqIO
from deep_hiv_ab_pred.catnap.constants import VIRUS_FILE, VIRUS_WITH_PNGS_FILE, ANTIBODIES_LIGHT_FILE, ANTIBODIES_HEAVY_FILE
from deep_hiv_ab_pred.preprocessing.constants import LIGHT_ANTIBODY_TRIM, HEAVY_ANTIBODY_TRIM

def calculate_tensor_coordinates_from_sequence(seq, kmer_len, kmer_stride):
    kmer_size = kmer_len * aminoacids_len
    kmer_count = int((len(seq) - kmer_len) / kmer_stride) + 1
    indexes = []
    for i, j in enumerate(range(0, len(seq) - kmer_len + 1, kmer_stride)):
        for k in range(kmer_len):
            indexes.append(k * aminoacids_len + amino_to_index[seq[j + k]] + i * kmer_size)
    return kmer_count, indexes

def kmers_tensor_from_coordinates(kmer_count, indexes, kmer_len):
    kmer_size = kmer_len * aminoacids_len
    kmer_tensor = t.zeros(kmer_count * kmer_size, dtype=t.float32, device = device)
    kmer_tensor[indexes] = 1
    return kmer_tensor.reshape(int(kmer_count), kmer_size)

def read_virus_fasta_sequences(fasta_file_path, kmer_len, kmer_stride):
    virus_seq_dict = {}
    for seq_record in SeqIO.parse(fasta_file_path, "fasta"):
        id_split = seq_record.id.split('.')
        virus_id = id_split[-2]
        seq = str(seq_record.seq)
        # seq = seq.replace('-', '')
        virus_seq_dict[virus_id] = calculate_tensor_coordinates_from_sequence(seq, kmer_len, kmer_stride)
    return virus_seq_dict

def read_antibody_fasta_sequences(fasta_file_path, antibody_trim, kmer_len, kmer_stride):
    antibody_seq_dict = {}
    for seq_record in SeqIO.parse(fasta_file_path, "fasta"):
        id_split = seq_record.id.split('_')
        antibody_id = id_split[0]
        seq = str(seq_record.seq)
        if len(seq) > antibody_trim:
            seq = seq[:antibody_trim]
        antibody_seq_dict[antibody_id] = calculate_tensor_coordinates_from_sequence(seq, kmer_len, kmer_stride)
    return antibody_seq_dict

def parse_catnap_sequences_to_one_hot(virus_kmer_len, virus_kmer_stride, antibody_kmer_len, antibody_kmer_stride):
    virus_seq = read_virus_fasta_sequences(VIRUS_FILE, virus_kmer_len, virus_kmer_stride)
    virus_pngs_mask = read_virus_pngs_mask(VIRUS_WITH_PNGS_FILE, virus_kmer_len, virus_kmer_stride)
    antibody_light_seq = read_antibody_fasta_sequences(ANTIBODIES_LIGHT_FILE, LIGHT_ANTIBODY_TRIM, antibody_kmer_len, antibody_kmer_stride)
    antibody_heavy_seq = read_antibody_fasta_sequences(ANTIBODIES_HEAVY_FILE, HEAVY_ANTIBODY_TRIM, antibody_kmer_len, antibody_kmer_stride)
    return virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq

if __name__ == '__main__':
    virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap_sequences_to_one_hot(51, 25, 15, 7)