import matplotlib.pyplot as plt
from deep_hiv_ab_pred.training.constants import MATTHEWS_CORRELATION_COEFFICIENT, ACCURACY

def plot_train_test(train, test, title, y_title, show = True):
    plt.plot(range(len(train)), train, label = 'Train')
    plt.plot(range(len(test)), test, label = 'Test')
    plt.xlabel('Epochs')
    plt.ylabel(y_title)
    plt.title(title)
    plt.legend()
    # plt.savefig(title + '.png', dpi = 300, format = 'png')
    if show:
        plt.show()

def plot_epochs(train_metrics_list, test_metrics_list, show = True, title = 'Correlation'):
    test_correlation = list(map(lambda m: m[MATTHEWS_CORRELATION_COEFFICIENT], test_metrics_list))
    train_corellation = list(map(lambda m: m[MATTHEWS_CORRELATION_COEFFICIENT], train_metrics_list))
    plot_train_test(train_corellation, test_correlation, title, 'Correlation', show)