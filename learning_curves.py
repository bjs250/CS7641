import utils 

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

def get_learning_curves(
    dataset,
    X_train, 
    y_train, 
    estimator,
    title,
    output,
    cv
    ):

    if dataset == 1:
        train_sizes=[10, 100, 1000, 5000, 10000, 20000], 
    if dataset == 2:
        train_sizes=[10, 100, 500, 850], 

    train_sizes, train_scores, valid_scores, fit_times, _ = learning_curve(
        estimator, 
        X_train, 
        y_train, 
        train_sizes=train_sizes, 
        cv=cv,
        return_times=True
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    ylim = (0.6, 1.1)
    _, axes = plt.subplots(1, 1, figsize=(20, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    axes.grid() 
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Train score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Test score")
    axes.legend(loc="lower right")

    plt.gcf().set_size_inches(6, 4)
    plt.savefig(output + utils.get_current_time() + ".png", dpi=300)
    plt.show()
