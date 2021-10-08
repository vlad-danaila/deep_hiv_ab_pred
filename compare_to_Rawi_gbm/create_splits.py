import numpy as np
from util.tools import read_json_file, dump_json
from compare_to_Rawi_gbm.constants import RAWI_DATA, COMPARE_SPLITS_FOR_RAWI, CATNAP_DATA
import random
from sklearn.model_selection import RepeatedKFold

def flatten_catnap_data(catnap_data):
    flat = []
    for antibody, viruses in catnap_data.items():
        flat.extend([(antibody, virus, catnap_data[antibody][virus]) for virus in viruses])
    return flat

def cross_validation_splits(cv_data, folds = 10, repeats = 10):
    random.shuffle(cv_data)
    cv_data = np.array(cv_data)
    rkf = RepeatedKFold(n_splits = folds, n_repeats = repeats)
    return [
        { 'train': cv_data[train].tolist(), 'test': cv_data[test].tolist() }
        for train, test in rkf.split(range(len(cv_data)))
    ]

def create_splits_to_compare_with_rawi():
    rawi_data = read_json_file(RAWI_DATA)
    catnap_data = read_json_file(CATNAP_DATA)
    flat_catnap = flatten_catnap_data(catnap_data)
    splits = {}
    for antibody, viruses in rawi_data.items():
        virus_ids = [v.split('.')[-2] for v in viruses]
        pretrain_data = [ data for data in flat_catnap if data[0] != antibody ]
        cross_validation_data = [ (antibody, virus, catnap_data[antibody][virus]) for virus in virus_ids ]
        cv_splits = cross_validation_splits(cross_validation_data)
        splits[antibody] = {
            'pretraining': pretrain_data,
            'cross_validation': cv_splits
        }
    return splits

if __name__ == '__main__':
    splits = create_splits_to_compare_with_rawi()
    dump_json(splits, COMPARE_SPLITS_FOR_RAWI)