import numpy as np
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.util.tools import read_json_file, dump_json
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import RAWI_DATA, COMPARE_SPLITS_FOR_RAWI
import random
from sklearn.model_selection import RepeatedStratifiedKFold

def cross_validation_splits(cv_data, ground_truths, folds = 10, repeats = 11):
    random.shuffle(cv_data)
    cv_data = np.array(cv_data)
    rkf = RepeatedStratifiedKFold(n_splits = folds, n_repeats = repeats)
    return [
        { 'train': cv_data[train].tolist(), 'test': cv_data[test].tolist() }
        for train, test in rkf.split(cv_data, ground_truths)
    ]

def is_cv_splits_valid(splits, catnap):
    for split in splits:
        train, test = split['train'], split['test']
        train_ground_truths = [ int(data[3]) for data in catnap if data[0] in train ]
        test_ground_truths = [ int(data[3]) for data in catnap if data[0] in test ]
        # If there are both positive and negative outcomes return True (valid)
        if sum(train_ground_truths) == len(train_ground_truths) or sum(test_ground_truths) == len(test_ground_truths):
            return False
    return True

def create_splits_to_compare_with_rawi(catnap):
    rawi_data = read_json_file(RAWI_DATA)
    splits = {}
    for antibody, viruses in rawi_data.items():
        virus_ids = [v.split('.')[-2] for v in viruses]
        pretrain_data = [ data[0] for data in catnap if data[1] != antibody ]
        cv_tuples = [(antibody, virus) for virus in virus_ids]
        cv_data = [data[0] for data in catnap if (data[1], data[2]) in cv_tuples]
        ground_truths = [data[3] for data in catnap if (data[1], data[2]) in cv_tuples]
        if len(cv_data) != len(cv_tuples):
            print('Skipping', antibody)
            continue
        print(antibody)
        cv_splits = cross_validation_splits(cv_data, ground_truths)
        while not is_cv_splits_valid(cv_splits, catnap):
            print('Retrying ', antibody)
            cv_splits = cross_validation_splits(cv_data, ground_truths) # Retry
        splits[antibody] = { 'pretraining': pretrain_data, 'cross_validation': cv_splits }
    return splits

if __name__ == '__main__':
    catnap = read_json_file(CATNAP_FLAT)
    splits = create_splits_to_compare_with_rawi(catnap)
    dump_json(splits, COMPARE_SPLITS_FOR_RAWI)