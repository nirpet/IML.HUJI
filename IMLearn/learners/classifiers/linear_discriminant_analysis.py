from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from IMLearn.metrics.loss_functions import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m = X.shape[0]
        self.classes_, n_k = np.unique(y, axis=0, return_counts=True)
        self.pi_ = n_k / m
        self.mu_ = np.empty(self.classes_.shape[0])
        for i in range(self.mu_.shape[0]):
            self.mu_[i] = (1.0 / n_k[i]) * np.sum(X[y == self.classes_[i]])

        mu_yi = np.vstack(self.mu_[y.astype(int)])
        self.cov_ = (1.0 / m) * np.dot((X - mu_yi).T, X - mu_yi)
        self._cov_inv = np.linalg.inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        a = self._cov_inv.dot(self.mu_)
        b = np.log(self.pi_) - 0.5 * np.dot(self.mu_.T, self._cov_inv.dot(self.mu_))
        return self.classes_[np.argmax(a.dot(X) + b)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihoods = np.empty(X.shape[0], self.classes_.shape[0])
        cov_det = np.linalg.det(self.cov_)
        for i in range(likelihoods.shape[1]):
            d = X[:, np.newaxis, :] - self.mu_[i]
            # taken from ex1 suggested solution
            mahalanobis = np.sum(d.dot(inv(self.cov_)) * d, axis=2).flatten()
            likelihoods[:, i] = self.pi_[i] * np.exp(-.5 * mahalanobis) / \
                                np.sqrt((2 * np.pi) ** len(X) * cov_det)

        return likelihoods

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
        return misclassification_error(y, y_pred)
