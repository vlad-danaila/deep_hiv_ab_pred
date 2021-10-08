from util.tools import read_json_file
from constants import RAWI_DATA, CATNAP_DATA

def print_antibody_virus_stats_for_rawi():
    rawi_data = read_json_file(RAWI_DATA)
    catnap_data = read_json_file(CATNAP_DATA)
    total_viruses, total_sensitive, total_resistant = 0, 0, 0
    for antibody, viruses in rawi_data.items():
        virus_ids = [v.split('.')[-2] for v in viruses]
        neutralizations = [catnap_data[antibody][v_id] for v_id in virus_ids]
        nb_viruses = len(viruses)
        sensitive = sum(neutralizations)
        resistant = nb_viruses - sensitive
        total_viruses += nb_viruses
        total_sensitive += sensitive
        total_resistant += resistant
        print(f'antibody={antibody}, viruses={nb_viruses}, sesnsitive={sensitive}, resistant={resistant}')
    print('-' * 30, 'total', '-' * 30)
    print(f'antibody-virus combinations={total_viruses}, sesnsitive={total_sensitive}, resistant={total_resistant}')

    total_catnap_antibody_virus_combinations = sum([len(viruses) for viruses in catnap_data.values()])
    total_catnap_antibodies = len(catnap_data)
    print('-' * 30, 'total', '-' * 30)
    print(f'Remaining catnap antibody-virus combinations {total_catnap_antibody_virus_combinations - total_viruses}, remaining catnap antibodies {total_catnap_antibodies - 33}')

if __name__ == '__main__':
    print_antibody_virus_stats_for_rawi()
