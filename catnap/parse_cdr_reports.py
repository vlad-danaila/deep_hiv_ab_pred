from deep_hiv_ab_pred.catnap.constants import PARATOME_AB_LIGHT_CDR, PARATOME_AB_HEAVY_CDR, ABRSA_AB_LIGHT_CDR, ABRSA_AB_HEAVY_CDR
from Bio import SeqIO
from deep_hiv_ab_pred.catnap.constants import ANTIBODIES_LIGHT_FASTA_FILE, ANTIBODIES_HEAVY_FASTA_FILE
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT, AB_CDRS
from deep_hiv_ab_pred.util.tools import read_json_file, dump_json
import re

SEQ_PARSE = 'SEQ_PARSE'
CDR_PARSE = 'CDR_PARSE'
AB_TYPE_LIGHT = 'AB_TYPE_LIGHT'
AB_TYPE_HEAVY = 'AB_TYPE_HEAVY'

def assays_abs():
    catnap_assays = read_json_file(CATNAP_FLAT)
    return { a[1] for a in catnap_assays }

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
    return get_ab_ids_to_cdrs_from_paratome(PARATOME_AB_LIGHT_CDR, ANTIBODIES_LIGHT_FASTA_FILE)

def ab_heavy_cdrs_from_paratome():
    return get_ab_ids_to_cdrs_from_paratome(PARATOME_AB_HEAVY_CDR, ANTIBODIES_HEAVY_FASTA_FILE)

def ab_light_cdrs_from_AbRSA():
    return parse_AbRSA_report(ABRSA_AB_LIGHT_CDR, ANTIBODIES_LIGHT_FASTA_FILE)

def ab_heavy_cdrs_from_AbRSA():
    return parse_AbRSA_report(ABRSA_AB_HEAVY_CDR, ANTIBODIES_HEAVY_FASTA_FILE)

def parse_AbRSA_report(report_file, antibodies_fasta_file):
    abs_in_assays = [a.upper() for a in assays_abs()]
    id_to_cdr = {}
    ids_to_seq = get_id_to_seq_mapping_from_fasta_file(antibodies_fasta_file)
    ids_to_seq = { k.upper() : v for k, v in ids_to_seq.items() }
    with open(report_file) as file:
        for line in file:
            line_split = line.split()
            antibody_id = line_split[0].split('_')[0]
            if antibody_id not in abs_in_assays:
                continue
            cdrs = line_split[2:]
            seq = ids_to_seq[antibody_id]
            id_to_cdr[antibody_id] = [(c, find_indexes_of_subsequence(c, seq)) for c in cdrs]
    return id_to_cdr

def combine_paratome_and_abrsa(paratome_cdrs: dict, abrsa_cdrs: dict, ab_type):
    combined = {}
    for ab_id, cdrs_from_paratome in paratome_cdrs.items():
        if len(cdrs_from_paratome) < 3:
            cdrs_from_abrsa = abrsa_cdrs[ab_id.upper()]
            if len(cdrs_from_abrsa) == 3:
                # cdrs_from_abrsa = [ (cdr, ) for cdr in cdrs_from_abrsa ]
                combined[ab_id] = cdrs_from_abrsa
            # CORRECTIONS:
            elif ab_type == AB_TYPE_HEAVY and ab_id == 'F105':
                # Verified (only one amino acid differs between the sequence from CATNAP and PDB)
                # Reparsed with Paratome using antibody fragment with larger sequence from https://www.rcsb.org/structure/1U6A
                # QVQLQESGPGLVKPSETLSLTCTVSGGSISSHYWSWIRQSPGKGLQWIGYIYYSGSTNYSPSLKSRVTISVETAKNQFSLKLTSMTAADTAVYYCARGPVPAVFYGDYRLDPWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPK
                combined[ab_id] = [('GSISSHYWS', (27, 35)), ('WIGYIYYSGSTNY', (47, 59)), ('RGPVPAVFYGDYRLDP', (97, 112))]
            elif ab_type == AB_TYPE_HEAVY and ab_id == '1F7':
                # Verified (no differences)
                # Reparsed with Paratome using antibody fragment with larger sequence from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3753353/
                # QVQLQESGPGLVKPSETLSLTCSVSGGSLSNFYWSWIRQFPGKRLEWIAYINFNNEKSNQNPSLKGRLTVSGDPSKNHLSMRLTSVTAADTAVYFCARGRFDYFRGGHRLIFDSWGRGTLVAVSS
                combined[ab_id] = [('GSLSNFYWS', (27, 35)), ('WIAYINFNNEKSNQ', (47, 60)), ('RGRFDYFRGGHRLIFDS', (98, 114))]
            elif ab_type == AB_TYPE_HEAVY and ab_id == '2F5':
                # Verified (no differences)
                # Reparsed with Paratome using antibody fragment with larger sequence from https://www.rcsb.org/structure/3LEV
                # RITLKESGPPLVKPTQTLTLTCSFSGFSLSDFGVGVGWIRQPPGKALEWLAIIYSDDDKRYSPSLNTRLTITKDTSKNQVVLVMTRVSPVDTATYFCAHRRGPTTLFGVPIARGPVNAMDVWGQGITVTISSTSTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDK
                combined[ab_id] = [('FSLSDFGVGVG', (27, 37)), ('WLAIIYSDDDKRY', (49, 61)), ('HRRGPTTLFGVPIARGPVNAMDV', (99, 121))]
            elif ab_type == AB_TYPE_HEAVY and ab_id == '2G12':
                # Verified (no differences)
                # Reparsed with Paratome using antibody fragment with larger sequence from https://www.rcsb.org/structure/2OQJ
                # EVQLVESGGGLVKAGGSLILSCGVSNFRISAHTMNWVRRVPGGGLEWVASISTSSTYRDYADAVKGRFTVSRDDLEDFVYLQMHKMRVEDTAIYYCARKGSDRLSDNDPFDAWGPGTVVTVSPASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPK
                combined[ab_id] = [('FRISAHTMN', (27, 35)), ('WVASISTSSTYRDY', (47, 60)), ('RKGSDRLSDNDPFDA', (98, 112))]
            else:
                print(ab_id, 'abrsa', cdrs_from_abrsa, 'paratome', cdrs_from_paratome )
        else:
            combined[ab_id] = [(cdr[1], cdr[2]) for cdr in cdrs_from_paratome]
    return combined

def find_indexes_of_subsequence(subseq, seq):
    return re.search(subseq, seq).span()

def verify_cdr_data(id_to_cds):
    for ab, cdrs in ab_light_combined.items():
        assert len(cdrs) == 3
        for cdr in cdrs:
            assert len(cdr) == 2

if __name__ == '__main__':
    ab_light_cdrs_paratome = ab_light_cdrs_from_paratome()
    ab_heavy_cdrs_paratome = ab_heavy_cdrs_from_paratome()

    ab_light_cdrs_abrsa = ab_light_cdrs_from_AbRSA()
    ab_heavy_cdrs_abrsa = ab_heavy_cdrs_from_AbRSA()

    ab_light_combined = combine_paratome_and_abrsa(ab_light_cdrs_paratome, ab_light_cdrs_abrsa, AB_TYPE_LIGHT)
    ab_heavy_combined = combine_paratome_and_abrsa(ab_heavy_cdrs_paratome, ab_heavy_cdrs_abrsa, AB_TYPE_HEAVY)

    verify_cdr_data(ab_light_combined)
    verify_cdr_data(ab_heavy_combined)

    cdr_data = {}
    for ab in ab_light_combined:
        ab_light = ab_light_combined[ab]
        ab_heavy = ab_heavy_combined[ab]
        cdr_data[ab] = { 'ab_light': ab_light, 'ab_heavy': ab_heavy }

    dump_json(cdr_data, AB_CDRS)
