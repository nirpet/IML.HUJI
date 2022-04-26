from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas as pd

from IMLearn.utils import split_train_test

pio.templates.default = "simple_white"


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
    raise NotImplementedError()


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        np.load(file="../datasets/"+f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        raise NotImplementedError()

        # Plot figure
        raise NotImplementedError()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        dataset = np.load(file="../datasets/"+f)
        y = dataset[:, -1]
        X = dataset[:, :-1]

        # Fit models and predict over training set
        train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(X), pd.Series(y))

        lda = LDA()
        lda.fit(train_x.to_numpy(), train_y.to_numpy())
        gna = GaussianNaiveBayes()
        gna.fit(train_x.to_numpy(), train_y.to_numpy())

        lda_y_pred = lda.predict(test_x.to_numpy())
        # gna_y_pred = gna.predict(test_x.to_numpy())

        lda_accuracy = np.mean(lda_y_pred == test_y)
        # gna_accuracy = np.mean(gna_y_pred == test_y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        # model_names = ["Linear discriminant analysis", "Gaussian naive Bayes"]
        # fig = make_subplots(rows=1, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in model_names],
        #                     horizontal_spacing=0.01, vertical_spacing=.03)
        #
        # fig.add_traces([decision_surface(m.fit(X, y).predict, lims[0], lims[1], showscale=False),
        #                     go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
        #                                marker=dict(color=y, symbol=symbols[y], colorscale=[custom[0], custom[-1]],
        #                                            line=dict(color="black", width=1)))],
        #                    rows=(i // 3) + 1, cols=(i % 3) + 1)
        # fig.add_traces([decision_surface(m.fit(X, y).predict, lims[0], lims[1], showscale=False),
        #                 go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
        #                            marker=dict(color=y, symbol=symbols[y], colorscale=[custom[0], custom[-1]],
        #                                        line=dict(color="black", width=1)))],
        #                rows=(i // 3) + 1, cols=(i % 3) + 1)
        #
        # fig.update_layout(title=rf"$\textbf{{(2) Decision Boundaries Of Models - {title} Dataset}}$",
        #                   margin=dict(t=100)) \
        #     .update_xaxes(visible=False).update_yaxes(visible=False)


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
