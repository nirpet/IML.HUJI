import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

pio.templates.default = "simple_white"
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        np.load(file="../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        raise NotImplementedError()

        # Plot figure of loss as function of fitting iteration
        raise NotImplementedError()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        dataset = np.load(file="../datasets/" + f)
        y = dataset[:, -1]
        X = dataset[:, :-1]

        lda = LDA()
        lda.fit(X, y)
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)

        lda_y_pred = lda.predict(X)
        gnb_y_pred = gnb.predict(X)

        lda_accuracy = np.mean(lda_y_pred == y)
        gnb_accuracy = np.mean(gnb_y_pred == y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Linear discriminant analysis, Accuracy: " + str(lda_accuracy),
                                            "Gaussian naive Bayes, Accuracy: " + str(gnb_accuracy)],
                            horizontal_spacing=0.01, vertical_spacing=.03)
        fig.update_layout(title="Dataset: " + f, showlegend=False)
        lda_results = go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=lda_y_pred, symbol=y, size=15))
        lda_data = [lda_results]
        for i in range(lda.classes_.shape[0]):
            lda_ellipse = get_ellipse(lda.mu_[i], lda.cov_)
            lda_X = go.Scatter(x=[lda.mu_[i][0]], y=[lda.mu_[i][1]], mode="markers", showlegend=False,
                               marker=dict(color='black', symbol='x', size=30))
            lda_data.append(lda_ellipse)
            lda_data.append(lda_X)

        fig.add_traces(lda_data, rows=1, cols=1)

        gnb_results = go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=gnb_y_pred, symbol=y, size=15))
        gnb_data = [gnb_results]

        for i in range(gnb.classes_.shape[0]):
            cov_matrix = np.zeros(shape=(2, 2))
            cov_matrix[0][0] = gnb.vars_[i][0]
            cov_matrix[1][1] = gnb.vars_[i][1]
            gnb_X = go.Scatter(x=[gnb.mu_[i][0]], y=[gnb.mu_[i][1]], mode="markers", showlegend=False,
                               marker=dict(color='black', symbol='x', size=30))
            gnb_ellipse = get_ellipse(gnb.mu_[i], cov_matrix)
            gnb_data.append(gnb_ellipse)
            gnb_data.append(gnb_X)

        fig.add_traces(gnb_data, rows=1, cols=2)
        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
