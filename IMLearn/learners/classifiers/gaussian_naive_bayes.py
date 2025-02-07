from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from IMLearn.metrics.loss_functions import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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
        self.mu_ = np.empty(shape=(self.classes_.shape[0], X.shape[1]))
        self.vars_ = np.empty(shape=(self.classes_.shape[0], X.shape[1]))
        for k in range(self.mu_.shape[0]):
            X_k = X[y == self.classes_[k]]
            self.mu_[k] = np.mean(X_k, axis=0)
            self.vars_[k] = np.var(X_k, ddof=1, axis=0)

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
        likelihood = self.likelihood(X)
        responses = np.empty(X.shape[0])
        for i in range(responses.shape[0]):
            responses[i] = np.argmax(likelihood[i])

        return responses

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

        likelihoods = np.zeros(shape=(X.shape[0], self.classes_.shape[0]))
        for i in range(likelihoods.shape[0]):
            for k in range(likelihoods.shape[1]):
                likelihoods[i][k] += np.log(self.pi_[k])
                for j in range(X.shape[1]):
                    likelihoods[i][k] += - 0.5 * np.log(self.vars_[k][j]) - \
                                         0.5 * ((X[i][j] - self.mu_[k][j]) ** 2 / self.vars_[k][j])

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
