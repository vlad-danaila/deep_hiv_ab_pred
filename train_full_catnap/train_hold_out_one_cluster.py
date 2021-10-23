from deep_hiv_ab_pred.util.tools import read_json_file, read_yaml
from deep_hiv_ab_pred.train_full_catnap.constants import SPLITS_HOLD_OUT_ONE_CLUSTER, MODELS_FOLDER
from deep_hiv_ab_pred.training.training import train_network, eval_network
from deep_hiv_ab_pred.hyperparameters.constants import CONF_ICERI_V2
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.preprocessing.pytorch_dataset import AssayDataset, zero_padding
from deep_hiv_ab_pred.preprocessing.sequences import parse_catnap_sequences
from deep_hiv_ab_pred.util.tools import read_json_file, read_yaml, device, get_experiment
from deep_hiv_ab_pred.model.ICERI2021_v2 import ICERI2021Net_V2
import torch as t
import numpy as np
from deep_hiv_ab_pred.training.constants import LOSS, ACCURACY, MATTHEWS_CORRELATION_COEFFICIENT
import mlflow

def log_metrics(metrics):
    metrics = np.array(metrics)
    for cv_fold in range(len(metrics)):
        mlflow.log_metrics({
            f'cv{cv_fold} acc': metrics[cv_fold][ACCURACY],
            f'cv{cv_fold} mcc': metrics[cv_fold][MATTHEWS_CORRELATION_COEFFICIENT]
        })
    cv_mean_acc = metrics[:, ACCURACY].mean()
    cv_std_acc = metrics[:, ACCURACY].std()
    cv_mean_mcc = metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].mean()
    cv_std_mcc = metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].std()
    print('CV Mean Acc', cv_mean_acc, 'CV Std Acc', cv_std_acc)
    print('CV Mean MCC', cv_mean_mcc, 'CV Std MCC', cv_std_mcc)
    mlflow.log_metrics({
        f'cv mean acc': cv_mean_acc,
        f'cv std acc': cv_std_acc,
        f'cv mean mcc': cv_mean_mcc,
        f'cv std mcc': cv_std_mcc
    })

def train(splits, catnap, conf):
    virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq = parse_catnap_sequences(
        conf['KMER_LEN_VIRUS'], conf['KMER_STRIDE_VIRUS'], conf['KMER_LEN_ANTB'], conf['KMER_STRIDE_ANTB']
    )
    cv_metrics = []
    for i, cv_fold in enumerate(splits['cv']):
        train_ids, val_ids = cv_fold['train'], cv_fold['val']
        train_assays = [a for a in catnap if a[0] in train_ids]
        val_assays = [a for a in catnap if a[0] in val_ids]
        train_set = AssayDataset(train_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        val_set = AssayDataset(val_assays, antibody_light_seq, antibody_heavy_seq, virus_seq, virus_pngs_mask)
        loader_train = t.utils.data.DataLoader(train_set, conf['BATCH_SIZE'], shuffle = True, collate_fn = zero_padding, num_workers = 0)
        loader_val = t.utils.data.DataLoader(val_set, len(val_set), shuffle = False, collate_fn = zero_padding, num_workers = 0)
        model = ICERI2021Net_V2(conf).to(device)
        _, _, best = train_network(
            model, conf, loader_train, loader_val, i, conf['EPOCHS'], f'model_cv_{i}', MODELS_FOLDER
        )
        print(f'CV {i} Acc {best[ACCURACY]} MCC {best[MATTHEWS_CORRELATION_COEFFICIENT]}')
        cv_metrics.append(best)
    log_metrics(cv_metrics)
    return cv_metrics

def main():
    conf = read_yaml(CONF_ICERI_V2)
    splits = read_json_file(SPLITS_HOLD_OUT_ONE_CLUSTER)
    catnap = read_json_file(CATNAP_FLAT)
    metrics = train(splits, catnap, conf)

if __name__ == '__main__':
    main()