from os.path import join
from deep_hiv_ab_pred.catnap.constants import AB_LIGHT_CDR, AB_HEAVY_CDR
from statistics import mean, median

def extract_cdr_indexes(text):
    text_between_parantheses = text[ text.find('(') + 1 : text.find(')') ]
    begin, end = tuple(text_between_parantheses.split('-'))
    return int(begin), int(end)

def gather_cdr_indexes(cdr_file):
    cdr_1_begin, cdr_2_begin, cdr_3_begin, cdr_1_end, cdr_2_end, cdr_3_end  = [], [], [], [], [], [],
    with open(cdr_file) as file:
        for line in file:
            if not line.startswith('ABR'):
                continue
            begin, end = extract_cdr_indexes(line)
            if line.startswith('ABR L1') or line.startswith('ABR H1'):
                cdr_1_begin.append(begin)
                cdr_1_end.append(end)
            elif line.startswith('ABR L2') or line.startswith('ABR H2'):
                cdr_2_begin.append(begin)
                cdr_2_end.append(end)
            elif line.startswith('ABR L3')  or line.startswith('ABR H3'):
                cdr_3_begin.append(begin)
                cdr_3_end.append(end)
    return cdr_1_begin, cdr_2_begin, cdr_3_begin, cdr_1_end, cdr_2_end, cdr_3_end

def parse_paratome_report(report_file):
    cdrs_in_seq = {}
    seq_parse = False
    cdr_parse = False
    seq = ''
    with open(report_file) as file:
        for line in file:
            if line.startswith('>paratome'):
                seq_parse = True
                continue
            if seq_parse:
                seq += line[:-1]
                if line == '\n':
                    seq_parse = False
                    cdr_parse = True
            if cdr_parse:
                #TODO

if __name__ == '__main__':
    parse_paratome_report(AB_LIGHT_CDR)