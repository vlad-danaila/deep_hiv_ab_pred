from Bio import SeqIO
from deep_hiv_ab_pred.catnap.constants import ANTIBODIES_LIGHT_FASTA_FILE, ANTIBODIES_HEAVY_FASTA_FILE, VIRUS_FILE
from collections import Counter
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.util.tools import read_json_file

def get_ab_id_from_fasta_file(ab_fasta_file_path):
    ab_ids = []
    for seq_record in SeqIO.parse(ab_fasta_file_path, "fasta"):
        id_split = seq_record.id.split('_')
        antibody_id = id_split[0]
        ab_ids.append(antibody_id)
    return ab_ids

def get_virus_ids_from_fasta_file(fasta_file_path):
    virus_ids = []
    for seq_record in SeqIO.parse(fasta_file_path, "fasta"):
        id_split = seq_record.id.split('.')
        virus_id = id_split[-2]
        virus_ids.append(virus_id)
    return virus_ids

if __name__ == '__main__':
    ab_light_ids = get_ab_id_from_fasta_file(ANTIBODIES_LIGHT_FASTA_FILE)
    ab_heavy_ids = get_ab_id_from_fasta_file(ANTIBODIES_HEAVY_FASTA_FILE)

    ab_light_count = Counter(ab_light_ids)
    ab_heavy_count = Counter(ab_heavy_ids)

    catnap_assays = read_json_file(CATNAP_FLAT)
    assays_abs = { a[1] for a in catnap_assays }

    print({ ab_id : count for ab_id, count in ab_light_count.items() if count > 1 and ab_id in assays_abs })
    print({ ab_id : count for ab_id, count in ab_heavy_count.items() if count > 1 and ab_id in assays_abs })

    virus_ids = get_virus_ids_from_fasta_file(VIRUS_FILE)
    virus_ids_count = Counter(virus_ids)

    print({ vir_id : count for vir_id, count in virus_ids_count.items() if count > 1 })