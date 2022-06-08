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

    train_results = np.empty(cv)
    test_results = np.empty(cv)
    train_y = np.empty(cv)

    for i in range(cv):
        train_x = np.concatenate(np.delete(X_folds, i))
        train_y[i] = np.concatenate(np.delete(y_folds, i))
        estimator.fit(train_x, train_y[i])
        train_results[i] = estimator.predict(train_x)
        test_results[i] = estimator.predict(X_folds[i])

    train_score = scoring(train_y, train_results)
    test_score = scoring(y_folds, test_results)

    return train_score, test_score


