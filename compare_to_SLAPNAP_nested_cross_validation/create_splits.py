import numpy as np
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.util.tools import read_json_file, dump_json
from deep_hiv_ab_pred.compare_to_SLAPNAP_nested_cross_validation.constants import SLAPNAP_RESULTS_NESTED_CV, COMPARE_SPLITS_FOR_SLAPNAP_NESTED_CV
from os import listdir
from os.path import join

def get_slpnap_ab_data():
    ab_viruses = {}
    for ab_results_file in listdir(SLAPNAP_RESULTS_NESTED_CV):
        if not ab_results_file.endswith('json'):
            continue
        ab_results = read_json_file(join(SLAPNAP_RESULTS_NESTED_CV, ab_results_file))
        antibody = '.'.join(ab_results_file.split('.')[:-1])
        ab_viruses[antibody] = (ab_results['virus_ids'], ab_results['folds'], ab_results['nested_folds'])
    return ab_viruses

def map_to_ids(antibody, viruses, catnap):
    cv_data = [data[0] for data in catnap if data[1] == antibody and data[2] in viruses]
    assert len(cv_data) == len(viruses)
    return cv_data

def cross_validation_splits(virus_ids, folds, antibody, catnap):
    virus_ids_np = np.array(virus_ids)
    cv_splits = []
    for fold, indexes in folds.items():
        # indexes are coming from R, where indices start with 1
        indexes = np.array(indexes) - 1
        test_viruses = virus_ids_np[indexes]
        train_splits_indexes = [i for i in range(len(virus_ids_np)) if i not in indexes]
        train_viruses = virus_ids_np[train_splits_indexes]
        assert len(train_viruses) + len(test_viruses) == len(virus_ids)
        train_ids = map_to_ids(antibody, train_viruses, catnap)
        test_ids = map_to_ids(antibody, test_viruses, catnap)
        cv_splits.append({ 'train': train_ids, 'test': test_ids })
    return cv_splits

def create_splits_to_compare_with_slapnap(catnap, fold):
    fold = str(fold)
    slapnap_data = get_slpnap_ab_data()
    splits = {}
    # The elements from fold i are not found in the elements of nested folds with the same index i
    # Therefore the nested fold is for training/validation and the fold is for testing
    for antibody, (virus_ids, folds, nested_folds) in slapnap_data.items():
        pretrain_data = [ data[0] for data in catnap if data[1] != antibody ]
        print(antibody)
        cv_splits = cross_validation_splits(virus_ids, nested_folds[fold], antibody, catnap)
        splits[antibody] = { 'pretraining': pretrain_data, 'test': folds[fold], 'cross_validation': cv_splits }
    return splits

if __name__ == '__main__':
    catnap = read_json_file(CATNAP_FLAT)
    for i in range(1, 6):
        splits = create_splits_to_compare_with_slapnap(catnap)
        dump_json(splits, COMPARE_SPLITS_FOR_SLAPNAP_NESTED_CV.replace('.json', f'_{i}.json'))