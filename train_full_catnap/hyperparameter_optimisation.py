import random
import optuna
from deep_hiv_ab_pred.train_full_catnap.constants import SPLITS_HOLD_OUT_ONE_CLUSTER
from deep_hiv_ab_pred.catnap.constants import CATNAP_FLAT
from deep_hiv_ab_pred.util.tools import read_json_file
from deep_hiv_ab_pred.training.constants import MATTHEWS_CORRELATION_COEFFICIENT
from deep_hiv_ab_pred.train_full_catnap.train_hold_out_one_cluster import train_hold_out_one_cluster
'''
# The size of the embeddings used as input to the recurrent neural network
    EMBEDDING_SIZE: 20

# Explanation for KMER: https://en.wikipedia.org/wiki/K-mer
KMER_LEN_ANTB: 51
KMER_STRIDE_ANTB: 25

KMER_LEN_VIRUS: 51
KMER_STRIDE_VIRUS: 25

# Training parameters
BATCH_SIZE: 2000
EPOCHS: 2
LEARNING_RATE: 1.e-4
GRAD_NORM_CLIP: 1.e-1

# Network structure
ANTIBODIES_LIGHT_RNN_HIDDEN_SIZE: 128
ANTIBODIES_HEAVY_RNN_HIDDEN_SIZE: 128

ANTIBODIES_LIGHT_RNN_NB_LAYERS: 2
ANTIBODIES_HEAVY_RNN_NB_LAYERS: 2
VIRUS_RNN_HIDDEN_NB_LAYERS: 2

EMBEDDING_DROPOUT: .0
ANTIBODIES_LIGHT_RNN_DROPOUT: .1
ANTIBODIES_HEAVY_RNN_DROPOUT: .1
VIRUS_RNN_DROPOUT: .1
FULLY_CONNECTED_DROPOUT: .1
'''
def propose(trial: optuna.trial.Trial):
    return {
        'EMBEDDING_SIZE': trial.suggest_int('EMBEDDING_SIZE', 2, 512)
    }

def get_objective_train_hold_out_one_cluster():
    splits = read_json_file(SPLITS_HOLD_OUT_ONE_CLUSTER)
    catnap = read_json_file(CATNAP_FLAT)
    def objective(trial):
        conf = propose(trial)
        try:
            cv_metrics = train_hold_out_one_cluster(splits, catnap, conf)
            cv_mean_mcc = cv_metrics[:, MATTHEWS_CORRELATION_COEFFICIENT].mean()
        except Exception as e:
            print(e)
            return 0
        return cv_mean_mcc
    return objective

def train_hold_out_one_cluster(splits, catnap, conf):
    return random.random()

if __name__ == '__main__':
    study = optuna.create_study()
    objective = get_objective_train_hold_out_one_cluster()
    study.optimize(objective, n_trials=5)
    print(study.best_params)