from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    X_folds = np.array_split(X, cv)
    y_folds = np.array_split(y, cv)

    train_score = np.empty(cv)
    test_score = np.empty(cv)

    for i in range(cv):
        train_x = X_folds.copy()
        del train_x[i]
        train_y = y_folds.copy()
        del train_y[i]
        train_x = np.concatenate(train_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        estimator.fit(train_x, train_y)
        train_results = estimator.predict(train_x)
        test_results = estimator.predict(X_folds[i])
        train_score[i] = scoring(train_y, train_results)
        test_score[i] = scoring(y_folds[i], test_results)

    return np.average(train_score), np.average(test_score)


