from deep_hiv_ab_pred.catnap.constants import AB_LIGHT_CDR, AB_HEAVY_CDR
from Bio import SeqIO
from deep_hiv_ab_pred.catnap.constants import ANTIBODIES_LIGHT_FILE, ANTIBODIES_HEAVY_FILE
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.util.tools import read_json_file

SEQ_PARSE = 'SEQ_PARSE'
CDR_PARSE = 'CDR_PARSE'

def extract_cdr_data(line):
    cdr_num, cdr_seq, indexes = tuple(line.split(' '))
    cdr_num = int(cdr_num[3]) # It has the form "ABR1:" "ABR2:" "ABR3:"
    indexes = indexes[1:-2] # remove parantheses and new line
    indexes = tuple((int(s) for s in indexes.split('-')))
    return cdr_num, cdr_seq, indexes

def parse_paratome_report(report_file):
    cdrs_in_seq = {}
    state = None
    with open(report_file) as file:
        for line in file:
            if line.startswith('>paratome'):
                if state == CDR_PARSE:
                    cdrs_in_seq[seq] = cdrs
                cdrs = []
                seq = ''
                state = SEQ_PARSE
                continue
            if state == SEQ_PARSE:
                if line == '\n':
                    state = CDR_PARSE
                else:
                    seq += line[:-1]
                continue
            if state == CDR_PARSE:
                cdrs.append(extract_cdr_data(line))
    cdrs_in_seq[seq] = cdrs # corresponding to the last line of the report
    return cdrs_in_seq

def get_id_to_seq_mapping_from_fasta_file(ab_fasta_file_path):
    antibody_seq_dict = {}
    for seq_record in SeqIO.parse(ab_fasta_file_path, "fasta"):
        id_split = seq_record.id.split('_')
        antibody_id = id_split[0]
        seq = str(seq_record.seq)
        antibody_seq_dict[antibody_id] = seq
    return antibody_seq_dict

def get_ab_ids_to_cdrs_from_paratome(paratome_report_file, antibodies_fasta_file):
    catnap = read_json_file(CATNAP_FLAT)
    antibodies_in_assayes = { c[1] for c in catnap }
    ids_to_seq = get_id_to_seq_mapping_from_fasta_file(antibodies_fasta_file)
    seq_to_cdr = parse_paratome_report(paratome_report_file)
    ab_ids_to_cdrs = { id : seq_to_cdr[seq] for (id, seq) in ids_to_seq.items() if id in antibodies_in_assayes }
    return ab_ids_to_cdrs



if __name__ == '__main__':
    # ab_light_cdrs = get_ab_ids_to_cdrs_from_paratome(AB_LIGHT_CDR, ANTIBODIES_LIGHT_FILE)
    ab_heavy_cdrs = get_ab_ids_to_cdrs_from_paratome(AB_HEAVY_CDR, ANTIBODIES_HEAVY_FILE)

    # TODO handle missing sequences from paratome report
    # TODO vezi unde nu sunt toate 3 cdr si ia-le de la celalat tool sau pune un default