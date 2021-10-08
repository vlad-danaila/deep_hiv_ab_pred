from util.tools import read_json_file, dump_json
from compare_to_Rawi_gbm.constants import RAWI_DATA, CATNAP_DATA

'''Exclude Rawi data from CATNAP data, that will be the pretraining.'''
def catnap_without_rawi_antibodies(rawi_data):
    catnap_data = read_json_file(CATNAP_DATA)
    initial_catnap_len = len(catnap_data)
    for antibody in rawi_data.keys():
        del catnap_data[antibody]
    assert initial_catnap_len == len(catnap_data) + 33
    return catnap_data

def create_splits_to_compare_with_rawi():
    rawi_data = read_json_file(RAWI_DATA)
    catnap_remaining = catnap_without_rawi_antibodies(rawi_data)

    for antibody, viruses in rawi_data.items():
        # Tb sa faci oversamling la minority class
        print(antibody, viruses)

if __name__ == '__main__':
    splits = create_splits_to_compare_with_rawi()