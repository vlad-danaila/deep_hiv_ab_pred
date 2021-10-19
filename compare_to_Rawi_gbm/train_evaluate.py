from deep_hiv_ab_pred.util.tools import read_json_file, read_yaml, device
from deep_hiv_ab_pred.compare_to_Rawi_gbm.constants import COMPARE_SPLITS_FOR_RAWI, MODELS_FOLDER
import torch as t
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.preprocessing.pytorch_dataset import AssayDataset, zero_padding
from deep_hiv_ab_pred.preprocessing.sequences import parse_catnap_sequences
from deep_hiv_ab_pred.hyperparameters.constants import CONF_ICERI
from deep_hiv_ab_pred.model.ICERI2021 import ICERI2021Net
from deep_hiv_ab_pred.training.training import train_network, eval_network
from os.path import join
from deep_hiv_ab_pred.training.constants import LOSS, ACCURACY, MATTHEWS_CORRELATION_COEFFICIENT
import mlflow

PRETRAINING = 'pretraining'
CV = 'cross_validation'
KMER_LEN = 'KMER_LEN'
KMER_STRIDE = 'KMER_STRIDE'
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
    model = ICERI2021Net(conf).to(device)
    _, _, best = train_network(
        model, conf, loader_pretrain, None, None, conf['EPOCHS_PRETRAIN'], f'model_{antibody}_pretrain', MODELS_FOLDER, 'pretrain'
    )

def cross_validate(antibody, splits_cv, catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq):
    for (i, cv_fold) in enumerate(splits_cv):
        train_ids, test_ids = cv_fold[TRAIN], cv_fold[TEST]
        train_assays = [a for a in catnap if a[0] in train_ids]
        test_assays = [a for a in catnap if a[0] in test_ids]
        assert len(train_assays) == len(train_ids) and len(test_assays) == len(test_ids)
        train_set = AssayDataset(train_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        test_set = AssayDataset(test_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        loader_train = t.utils.data.DataLoader(train_set, conf['BATCH_SIZE'], shuffle = True, collate_fn = zero_padding, num_workers = 0)
        loader_test = t.utils.data.DataLoader(test_set, len(test_set), shuffle = False, collate_fn = zero_padding, num_workers = 0)
        model = ICERI2021Net(conf).to(device)
        checkpoint = t.load(join(MODELS_FOLDER, f'model_{antibody}_pretrain.tar'))
        model.load_state_dict(checkpoint['model'])
        # Try freezing part of net
        # for param in model.parameters():
        #     param.requires_grad = False
        # for param in model.fully_connected.parameters():
        #     param.requires_grad = True
        _, _, best = train_network(
            model, conf, loader_train, loader_test, i, conf['EPOCHS_CV'], f'model_{antibody}', MODELS_FOLDER, f'cv{i+1}'
        )

def train_net(experiment, tags = None):
    with mlflow.start_run(experiment_id = experiment, tags = tags):
        conf = read_yaml(CONF_ICERI)
        mlflow.log_params(conf)
        all_splits = read_json_file(COMPARE_SPLITS_FOR_RAWI)
        mlflow.log_artifact(COMPARE_SPLITS_FOR_RAWI)
        catnap = read_json_file(CATNAP_FLAT)
        mlflow.log_artifact(CATNAP_FLAT)
        virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap_sequences(
            conf[KMER_LEN], conf[KMER_STRIDE], conf[KMER_LEN], conf[KMER_STRIDE]
        )
        for antibody, splits in all_splits.items():
            print('Antibody', antibody)
            pretrain_net(antibody, splits[PRETRAINING], catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq)
            cross_validate(antibody, splits[CV], catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq)

if __name__ == '__main__':
    tags = {
        'note1': 'virus seq aligned unlike in ICERI2021',
        'note2': 'no parameters are freezed'
    }
    train_net('ICERI2021', tags)