from deep_hiv_ab_pred.catnap.constants import AB_LIGHT_CDR, AB_HEAVY_CDR

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
    return cdrs_in_seq

if __name__ == '__main__':
    print('Ab Light')
    ab_light_cdrs_by_seq = parse_paratome_report(AB_LIGHT_CDR)
    for seq, cdrs in ab_light_cdrs_by_seq.items():
        if len(cdrs) != 3:
            print(seq, len(cdrs))
    print('-' * 50)

    print('Ab Heavy')
    ab_heavy_cdrs_by_seq = parse_paratome_report(AB_HEAVY_CDR)
    for seq, cdrs in ab_heavy_cdrs_by_seq.items():
        if len(cdrs) != 3:
            print(seq, len(cdrs))
    print('-' * 50)