from deep_hiv_ab_pred.catnap.constants import PARATOME_AB_LIGHT_CDR, PARATOME_AB_HEAVY_CDR, ABRSA_AB_LIGHT_CDR, ABRSA_AB_HEAVY_CDR
from Bio import SeqIO
from deep_hiv_ab_pred.catnap.constants import ANTIBODIES_LIGHT_FILE, ANTIBODIES_HEAVY_FILE
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.util.tools import read_json_file
import numpy as np

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
    ab_ids_to_cdrs = {}
    for id, seq in ids_to_seq.items():
        if id not in antibodies_in_assayes:
            continue
        if seq not in seq_to_cdr:
            ab_ids_to_cdrs[id] = []
            continue
        ab_ids_to_cdrs[id] = seq_to_cdr[seq]
    return ab_ids_to_cdrs

def ab_light_cdrs_from_paratome():
    return get_ab_ids_to_cdrs_from_paratome(PARATOME_AB_LIGHT_CDR, ANTIBODIES_LIGHT_FILE)

def ab_heavy_cdrs_from_paratome():
    return get_ab_ids_to_cdrs_from_paratome(PARATOME_AB_HEAVY_CDR, ANTIBODIES_HEAVY_FILE)

def ab_light_cdrs_from_AbRSA():
    return parse_AbRSA_report(ABRSA_AB_LIGHT_CDR)

def ab_heavy_cdrs_from_AbRSA():
    return parse_AbRSA_report(ABRSA_AB_HEAVY_CDR)

def parse_AbRSA_report(report_file):
    id_to_cdr = {}
    with open(report_file) as file:
        for line in file:
            line_split = line.split()
            antibody_id = line_split[0].split('_')[0]
            id_to_cdr[antibody_id] = line_split[2:]
    return id_to_cdr

def combine_paratome_and_abrsa(paratome_cdrs: dict, abrsa_cdrs: dict):
    combined = {}
    for ab_id, cdrs_from_paratome in paratome_cdrs.items():
        if len(cdrs_from_paratome) < 3:
            cdrs_from_abrsa = abrsa_cdrs[ab_id.upper()]
            if len(cdrs_from_abrsa) == 3:
                combined[ab_id] = cdrs_from_abrsa
            elif ab_id == 'F105':
                # Verified (only one amino acid differs between the sequence from CATNAP and PDB)
                # Reparsed with Paratome using antibody fragment with larger sequence from https://www.rcsb.org/structure/1U6A
                # QVQLQESGPGLVKPSETLSLTCTVSGGSISSHYWSWIRQSPGKGLQWIGYIYYSGSTNYSPSLKSRVTISVETAKNQFSLKLTSMTAADTAVYYCARGPVPAVFYGDYRLDPWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPK
                combined[ab_id] = ['GSISSHYWS', 'WIGYIYYSGSTNY', 'RGPVPAVFYGDYRLDP']
            elif ab_id == '1F7':
                # Verified (no differences)
                # Reparsed with Paratome using antibody fragment with larger sequence from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3753353/
                # QVQLQESGPGLVKPSETLSLTCSVSGGSLSNFYWSWIRQFPGKRLEWIAYINFNNEKSNQNPSLKGRLTVSGDPSKNHLSMRLTSVTAADTAVYFCARGRFDYFRGGHRLIFDSWGRGTLVAVSS
                combined[ab_id] = ['GSLSNFYWS', 'WIAYINFNNEKSNQ', 'RGRFDYFRGGHRLIFDS']
            else:
                print(ab_id, 'abrsa', cdrs_from_abrsa, 'paratome', cdrs_from_paratome )
                # combined[ab_id] = []
        else:
            combined[ab_id] = [cdr[1] for cdr in cdrs_from_paratome]
    return combined

def get_cdr_indexes(ab_cdrs_paratome):
    id_to_cdr_indexes = {}
    for id, cdrs in ab_cdrs_paratome.items():
        if len(cdrs) == 3:
            cdr_indexes = np.stack([  np.array(cdr[2]) for cdr in cdrs ]).reshape(-1)
            id_to_cdr_indexes[id] = cdr_indexes
    return id_to_cdr_indexes



if __name__ == '__main__':
    ab_light_cdrs_paratome = ab_light_cdrs_from_paratome()
    ab_heavy_cdrs_paratome = ab_heavy_cdrs_from_paratome()

    ab_light_cdrs_abrsa = ab_light_cdrs_from_AbRSA()
    ab_heavy_cdrs_abrsa = ab_heavy_cdrs_from_AbRSA()

    ab_light_combined = combine_paratome_and_abrsa(ab_light_cdrs_paratome, ab_light_cdrs_abrsa)
    ab_heavy_combined = combine_paratome_and_abrsa(ab_heavy_cdrs_paratome, ab_heavy_cdrs_abrsa)

    ab_heavy_id_to_indexes = get_cdr_indexes(ab_heavy_cdrs_paratome)

    all_indexes = np.stack(list(ab_heavy_id_to_indexes.values()))

    # TODO handle missing sequences from paratome report
    # TODO vezi unde nu sunt toate 3 cdr si ia-le de la celalat tool sau pune un default