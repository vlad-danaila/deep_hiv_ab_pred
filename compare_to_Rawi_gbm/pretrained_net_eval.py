from deep_hiv_ab_pred.util.tools import get_experiment
from deep_hiv_ab_pred.global_constants import DEFAULT_CONF
import mlflow
import logging
from deep_hiv_ab_pred.compare_to_Rawi_gbm.hyperparameter_optimisation import get_data
from deep_hiv_ab_pred.util.metrics import log_metrics_from_lists
from deep_hiv_ab_pred.compare_to_Rawi_gbm.train_evaluate import pretrain_net
from deep_hiv_ab_pred.training.constants import MATTHEWS_CORRELATION_COEFFICIENT, ACCURACY, AUC
from deep_hiv_ab_pred.util.plotting import plot_epochs
import matplotlib.pyplot as plt

PRETRAINING = 'pretraining'

def eval_pretrained_net(experiment_name, proposed_epochs, tags = None):
    experiment_id = get_experiment(experiment_name)
    with mlflow.start_run(experiment_id = experiment_id, tags = tags):
        all_splits, catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, ab_to_types = get_data()
        mlflow.log_artifact(DEFAULT_CONF, 'conf.json')
        acc, mcc, auc = [], [], []
        for i, (antibody, splits) in enumerate(all_splits.items()):
            logging.info(f'{i}. Antibody {antibody}')
            metrics_train_per_epochs, metrics_test_per_epochs, best = pretrain_net(
                antibody, splits[PRETRAINING], catnap, conf, virus_seq, virus_pngs_mask, antibody_light_seq, antibody_heavy_seq, proposed_epochs)
            ideal_nb_epochs = [
                i + 1 for (i, m) in enumerate(metrics_test_per_epochs)
                if m[MATTHEWS_CORRELATION_COEFFICIENT] == best[MATTHEWS_CORRELATION_COEFFICIENT]
            ][-1]
            plot_epochs(metrics_train_per_epochs, metrics_test_per_epochs, title = antibody, save_file = f'plot_{antibody} ({ideal_nb_epochs})')
            acc.append(best[ACCURACY])
            mcc.append(best[MATTHEWS_CORRELATION_COEFFICIENT])
            auc.append(best[AUC])
            logging.info(f'Ideal nb epochs {ideal_nb_epochs}')
        plt.show()
        log_metrics_from_lists(acc, mcc, auc)