import numpy as np
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.util.tools import read_json_file, dump_json
from deep_hiv_ab_pred.compare_to_SLAPNAP.constants import SPLITS_COMPARE_SLAPNAP, SLAPNAP_RESULTS
import random
from sklearn.model_selection import RepeatedStratifiedKFold
from os import listdir
from os.path import join

def get_slpnap_ab_virus_pairs():
    ab_viruses = {}
    for ab_results_file in listdir(SLAPNAP_RESULTS):
        if not ab_results_file.endswith('json'):
            continue
        ab_results = read_json_file(join(SLAPNAP_RESULTS, ab_results_file))
        antibody = '.'.join(ab_results_file.split('.')[:-1])
        ab_viruses[antibody] = ab_results['virus_ids']
    return ab_viruses

def create_splits_to_compare_with_slapnap(catnap):
    slapnap_data = get_slpnap_ab_virus_pairs()
    splits = {}
    for antibody, virus_ids in slapnap_data.items():
        pretrain_data = [ data[0] for data in catnap if data[1] != antibody ]
        cv_tuples = [(antibody, virus) for virus in virus_ids]
        cv_data = [data[0] for data in catnap if (data[1], data[2]) in cv_tuples]
        ground_truths = [data[3] for data in catnap if (data[1], data[2]) in cv_tuples]
        if len(cv_data) != len(cv_tuples):
            print('Skipping', antibody)
            continue
        print(antibody)
        # cv_splits = cross_validation_splits(cv_data, ground_truths)
        splits[antibody] = { 'pretraining': pretrain_data, 'cross_validation': [] }
    return splits

if __name__ == '__main__':
    catnap = read_json_file(CATNAP_FLAT)
    splits = create_splits_to_compare_with_slapnap(catnap)
    dump_json(splits, SPLITS_COMPARE_SLAPNAP)