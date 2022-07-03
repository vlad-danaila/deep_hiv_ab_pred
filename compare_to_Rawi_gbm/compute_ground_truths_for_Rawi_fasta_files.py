from Bio import SeqIO
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.util.tools import read_json_file
from typing import List
from os import listdir

def read_virus_fasta_file(fasta_file_path):
    viruses_ids = []
    for seq_record in SeqIO.parse(fasta_file_path, "fasta"):
        id_split = seq_record.id.split('.')
        viruses_ids.append(id_split[-2])
    return viruses_ids

def find_ground_truths(catnap: List[List], viruses_ids: List[str], antibody_id: str):
    ground_truths = []
    for virus_id in viruses_ids:
        ground_truth = [a for a in catnap if a[1] == antibody_id and a[2] == virus_id][0][3]
        ground_truths.append(ground_truth)
    return ground_truths

def find_antibody_id_from_file_name(fasta_file: str):
    return fasta_file.split('_')[0]

if __name__ == '__main__':
    fasta_files_dir = 'C:/DOC/Workspace/bNAb-ReP/alignments'
    catnap = read_json_file(CATNAP_FLAT)

    for fasta_file in listdir(fasta_files_dir):
        absolute_fasta_file = fasta_files_dir + '/' + fasta_file
        ab = find_antibody_id_from_file_name(fasta_file)
        if ab in ['VRC13', 'VRC29.03']:
            continue
        print(ab)
        viruses_ids = read_virus_fasta_file(absolute_fasta_file)
        ground_truths = find_ground_truths(catnap, viruses_ids, ab)
        ground_truths = [('1\n' if g else '0\n') for g in ground_truths]
        assert len(viruses_ids) == len(ground_truths)
        ground_truths_file_name = fasta_files_dir + '/' + fasta_file.replace('fasta', 'txt').replace('alignment', 'neutralization')
        with open(ground_truths_file_name, 'w') as ground_truths_file:
            ground_truths_file.writelines(ground_truths)
