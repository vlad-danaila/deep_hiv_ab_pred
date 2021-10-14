import matplotlib.pyplot as plt
from training.constants import MATTHEWS_CORRELATION_COEFFICIENT, ACCURACY, LOSS

def plot_train_test(train, test, title, y_title):
    plt.plot(range(len(train)), train, label = 'Train')
    plt.plot(range(len(test)), test, label = 'Test')
    plt.xlabel('Epochs')
    plt.ylabel(y_title)
    plt.title(title)
    plt.legend()
    # plt.savefig(title + '.png', dpi = 300, format = 'png')
    plt.show()

"""This function plots the loss, accuracy and Matthews correlation coefficient across the training epochs."""

def plot_epochs(train_metrics_list, test_metrics_list):
    test_correlation = list(map(lambda m: m[MATTHEWS_CORRELATION_COEFFICIENT], test_metrics_list))
    test_accuracy = list(map(lambda m: m[ACCURACY], test_metrics_list))
    test_loss = list(map(lambda m: m[LOSS], test_metrics_list))

    train_corellation = list(map(lambda m: m[MATTHEWS_CORRELATION_COEFFICIENT], train_metrics_list))
    train_accuracy = list(map(lambda m: m[ACCURACY], train_metrics_list))
    train_loss = list(map(lambda m: m[LOSS], train_metrics_list))

    plot_train_test(train_corellation, test_correlation, 'Correlation', 'Correlation')
    plot_train_test(train_accuracy, test_accuracy, 'Accuracy', 'Accuracy')
    plot_train_test(train_loss, test_loss, 'Loss', 'Loss')
