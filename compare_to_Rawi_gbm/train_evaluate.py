import numpy as np
import statistics
from deep_hiv_ab_pred.util.tools import read_json_file, read_yaml, device, get_experiment
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import COMPARE_SPLITS_FOR_RAWI, MODELS_FOLDER, KMER_LEN, KMER_STRIDE
import torch as t
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.preprocessing.pytorch_dataset import AssayDataset, zero_padding
from deep_hiv_ab_pred.preprocessing.sequences import parse_catnap_sequences
from deep_hiv_ab_pred.hyperparameters.constants import CONF_ICERI
from deep_hiv_ab_pred.model.ICERI2021_v2 import ICERI2021Net_V2
from deep_hiv_ab_pred.training.training import train_network, eval_network, train_with_frozen_antibody_and_embedding
from os.path import join
from deep_hiv_ab_pred.training.constants import LOSS, ACCURACY, MATTHEWS_CORRELATION_COEFFICIENT
import mlflow
import optuna

PRETRAINING = 'pretraining'
CV = 'cross_validation'
TRAIN = 'train'
TEST = 'test'

def pretrain_net(antibody, splits_pretraining, catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq):
    pretraining_assays = [a for a in catnap if a[0] in splits_pretraining]
    assert len(pretraining_assays) == len(splits_pretraining)
    pretrain_set = AssayDataset(
        pretraining_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask
    )
    loader_pretrain = t.utils.data.DataLoader(
        pretrain_set, conf['BATCH_SIZE'], shuffle = True, collate_fn = zero_padding, num_workers = 0
    )
    model = ICERI2021Net_V2(conf).to(device)
    _, _, best = train_network(
        model, conf, loader_pretrain, None, None, conf['EPOCHS'], f'model_{antibody}_pretrain', MODELS_FOLDER
    )

def cross_validate_antibody(antibody, splits_cv, catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, trial = None, cv_folds_trim = 100):
    cv_metrics = []
    for (i, cv_fold) in enumerate(splits_cv[:cv_folds_trim]):
        train_ids, test_ids = cv_fold[TRAIN], cv_fold[TEST]
        train_assays = [a for a in catnap if a[0] in train_ids]
        test_assays = [a for a in catnap if a[0] in test_ids]
        assert len(train_assays) == len(train_ids) and len(test_assays) == len(test_ids)
        train_set = AssayDataset(train_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        test_set = AssayDataset(test_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        loader_train = t.utils.data.DataLoader(train_set, conf['BATCH_SIZE'], shuffle = True, collate_fn = zero_padding, num_workers = 0)
        loader_test = t.utils.data.DataLoader(test_set, len(test_set), shuffle = False, collate_fn = zero_padding, num_workers = 0)
        model = ICERI2021Net_V2(conf).to(device)
        checkpoint = t.load(join(MODELS_FOLDER, f'model_{antibody}_pretrain.tar'))
        model.load_state_dict(checkpoint['model'])
        _, _, best = train_with_frozen_antibody_and_embedding(
            model, conf, loader_train, loader_test, i, conf['EPOCHS'], f'model_{antibody}', MODELS_FOLDER, False, log_every_epoch = False
        )
        cv_metrics.append(best)
        if trial:
            trial.report(best[MATTHEWS_CORRELATION_COEFFICIENT], i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return cv_metrics

def train_net(experiment_name, tags = None):
    experiment_id = get_experiment(experiment_name)
    with mlflow.start_run(experiment_id = experiment_id, tags = tags):
        conf = read_yaml(CONF_ICERI)
        mlflow.log_params(conf)
        all_splits = read_json_file(COMPARE_SPLITS_FOR_RAWI)
        # mlflow.log_artifact(COMPARE_SPLITS_FOR_RAWI)
        catnap = read_json_file(CATNAP_FLAT)
        # mlflow.log_artifact(CATNAP_FLAT)
        virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap_sequences(
            conf[KMER_LEN], conf[KMER_STRIDE], conf[KMER_LEN], conf[KMER_STRIDE]
        )
        acc, mcc = [], []
        for i, (antibody, splits) in enumerate(all_splits.items()):
            print(f'{i}. Antibody', antibody)
            pretrain_net(antibody, splits[PRETRAINING], catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq)
            cv_metrics = cross_validate_antibody(antibody, splits[CV], catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq)
            cv_metrics = np.array(cv_metrics)
            cv_mean_acc = cv_metrics[:, ACCURACY].mean()
            cv_std_acc = cv_metrics[:, ACCURACY].std()
            cv_mean_mcc = cv_metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].mean()
            cv_std_mcc = cv_metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].std()
            print(f'{i}. Antibody', antibody)
            print('CV Mean Acc', cv_mean_acc, 'CV Std Acc', cv_std_acc)
            print('CV Mean MCC', cv_mean_mcc, 'CV Std MCC', cv_std_mcc)
            mlflow.log_metrics({
                f'cv mean acc {antibody}': cv_mean_acc,
                f'cv std acc {antibody}': cv_std_acc,
                f'cv mean mcc {antibody}': cv_mean_mcc,
                f'cv std mcc {antibody}': cv_std_mcc
            })
            acc.append(cv_mean_acc)
            mcc.append(cv_mean_mcc)
        global_acc = statistics.mean(acc)
        global_mcc = statistics.mean(mcc)
        print('Global ACC', global_acc)
        print('Global MCC', global_mcc)
        mlflow.log_metrics({ 'global_acc': global_acc, 'global_mcc': global_mcc })

if __name__ == '__main__':
    tags = {
        'note1': 'virus seq aligned unlike in ICERI2021',
        'note2': 'no parameters are freezed'
    }
    train_net('ICERI2021', tags)