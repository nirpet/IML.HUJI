from __future__ import annotations

from typing import Tuple, NoReturn

import numpy as np

from ...base import BaseEstimator


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, 1

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        feature = X[:0]
        self.threshold_, error = self._find_threshold(feature, y, self.sign_)
        self.j_ = 0
        for i in range(1, X.shape[1]):
            feature = X[:i]
            threshold, threshold_error = self._find_threshold(feature, y, self.sign_)
            if threshold_error < error:
                self.threshold_ = threshold
                self.j_ = i

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        responses = np.empty(X.shape[1])
        responses[X[self.j_] >= self.threshold_] = self.sign_
        responses[X[self.j_] < self.threshold_] = -self.sign_
        return responses

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        indices = values.argsort()
        labels = labels[indices]
        threshold_index = 0
        threshold_error = 1

        for i in range(labels.shape[0]):
            pred = np.empty(labels.shape)
            pred[:i] = -sign
            pred[i:] = sign
            pred = pred - labels
            error = np.mean(labels[np.sign(pred, labels) != self.sign_])
            if error < threshold_error:
                threshold_error = error
                threshold_index = i

        return values[indices[threshold_index]], threshold_error

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.predict(X)
        errors = y[np.sign(y_pred, y) != self.sign_]
        return np.mean(np.abs(errors))
