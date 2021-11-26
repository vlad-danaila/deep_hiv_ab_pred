import torch as t
from deep_hiv_ab_pred.preprocessing.aminoacids import amino_to_index
from deep_hiv_ab_pred.util.tools import device
from Bio import SeqIO
from deep_hiv_ab_pred.catnap.constants import VIRUS_FILE, VIRUS_WITH_PNGS_FILE, ANTIBODIES_LIGHT_FASTA_FILE, ANTIBODIES_HEAVY_FASTA_FILE
from deep_hiv_ab_pred.preprocessing.constants import LIGHT_ANTIBODY_TRIM, HEAVY_ANTIBODY_TRIM

def sequence_to_indexes(seq, kmer_len, kmer_stride):
    return t.tensor([
        [amino_to_index[seq[i + j]] for j in range(kmer_len)]
        for i in range(0, len(seq) - kmer_len + 1, kmer_stride)
    ], dtype=t.long, device = device)

def kmers(seq, kmer_len, stride):
    return [
        seq[i : i + kmer_len]
        for i in range(0, len(seq) - kmer_len + 1, stride)
    ]

def pngs_mask_to_kemr_tensor(pngs_mask, kmer_len, kmer_stride):
    kmer_list = kmers(pngs_mask, kmer_len, kmer_stride)
    return t.tensor(kmer_list, dtype=t.float32, device=device)

def read_virus_fasta_sequences(fasta_file_path, kmer_len, kmer_stride):
    virus_seq_dict = {}
    for seq_record in SeqIO.parse(fasta_file_path, "fasta"):
        id_split = seq_record.id.split('.')
        virus_id = id_split[-2]
        seq = str(seq_record.seq)
        # seq = seq.replace('-', '')
        virus_seq_dict[virus_id] = sequence_to_indexes(seq, kmer_len, kmer_stride)
    return virus_seq_dict

def read_antibody_fasta_sequences(fasta_file_path, antibody_trim):
    antibody_seq_dict = {}
    for seq_record in SeqIO.parse(fasta_file_path, "fasta"):
        id_split = seq_record.id.split('_')
        antibody_id = id_split[0]
        seq = str(seq_record.seq)
        if len(seq) > antibody_trim:
            # print('Triming', seq_record.id)
            # print(f'>{seq_record.id}')
            # print(seq_record.seq)
            seq = seq[:antibody_trim]
        antibody_seq_dict[antibody_id] = t.tensor([amino_to_index[s] for s in seq], dtype=t.long, device = device)
    return antibody_seq_dict

def read_virus_pngs_mask(fasta_file_path, kmer_len, kmer_stride):
    virus_seq_dict = {}

    for seq_record in SeqIO.parse(fasta_file_path, "fasta"):
        id_split = seq_record.id.split('.')
        virus_id = id_split[-2]
        # seq = str(seq_record.seq).replace('-', '')
        seq = str(seq_record.seq)
        virus_seq_dict[virus_id] = seq

    for virus, seq in virus_seq_dict.items():
        binary_mask = [1. if c == 'O' else 0. for c in seq]
        virus_seq_dict[virus] = pngs_mask_to_kemr_tensor(binary_mask, kmer_len, kmer_stride)

    return virus_seq_dict

def parse_catnap_sequences_to_embeddings(virus_kmer_len, virus_kmer_stride):
    virus_seq = read_virus_fasta_sequences(VIRUS_FILE, virus_kmer_len, virus_kmer_stride)
    virus_pngs_mask = read_virus_pngs_mask(VIRUS_WITH_PNGS_FILE, virus_kmer_len, virus_kmer_stride)
    antibody_light_seq = read_antibody_fasta_sequences(ANTIBODIES_LIGHT_FASTA_FILE, LIGHT_ANTIBODY_TRIM)
    antibody_heavy_seq = read_antibody_fasta_sequences(ANTIBODIES_HEAVY_FASTA_FILE, HEAVY_ANTIBODY_TRIM)
    return virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq

if __name__ == '__main__':
    virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap_sequences_to_embeddings(51, 25)