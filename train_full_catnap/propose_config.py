import numpy as np
import optuna
from deep_hiv_ab_pred.preprocessing.aminoacids import get_embeding_matrix

EMBED_SIZE = get_embeding_matrix().shape[1]

def choose(x, choices):
    choices_idx = np.linspace(0, 1, len(choices))
    diff = np.abs(choices_idx - x)
    return choices[diff.argmin()]

def handle_categorical_params(conf):
    POS_EMBED = choose(conf['POS_EMBED'], list(range(6, 129, 2)))
    # divs = np.array(divisors(EMBED_SIZE + POS_EMBED + 1))
    conf['POS_EMBED'] = POS_EMBED
    # conf['N_HEADS_ENCODER'] = choose(conf['N_HEADS_ENCODER'], divs)
    # conf['N_HEADS_DECODER'] = choose(conf['N_HEADS_DECODER'], divs)
    return conf

def wrap_propose(trial: optuna.trial.Trial):
    conf = propose(trial)
    return handle_categorical_params(conf)

def propose(trial: optuna.trial.Trial):
    return {
        'EPOCHS': 100,
        "BATCH_SIZE": trial.suggest_int('BATCH_SIZE', 50, 5000),
        "LEARNING_RATE": trial.suggest_loguniform('LEARNING_RATE', 1e-4, 1),
        "WARMUP": trial.suggest_int("WARMUP", 1000, 100_000, log = True),

        "EMBEDDING_DROPOUT": trial.suggest_float('EMBEDDING_DROPOUT', 0, .5),
        "FULLY_CONNECTED_DROPOUT": trial.suggest_float('FULLY_CONNECTED_DROPOUT', 0, .5),

        "N_HEADS_ENCODER": 1,
        "TRANS_HIDDEN_ENCODER": trial.suggest_int('TRANS_HIDDEN_ENCODER', 10, 1024),
        "TRANS_DROPOUT_ENCODER": trial.suggest_float('TRANS_DROPOUT_ENCODER', 0, .5),
        "TRANSF_ENCODER_LAYERS": 1,

        "N_HEADS_DECODER": 1,
        "TRANS_HIDDEN_DECODER": trial.suggest_int('TRANS_HIDDEN_DECODER', 10, 1024),
        "TRANS_DROPOUT_DECODER": trial.suggest_float('TRANS_DROPOUT_DECODER', 0, .5),
        "TRANSF_DECODER_LAYERS": 1,

        "POS_EMBED": trial.suggest_float('POS_EMBED', 0, 1),
    }