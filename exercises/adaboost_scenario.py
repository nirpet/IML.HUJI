import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    loss_per_learners_train = np.empty(n_learners)
    loss_per_learners_test = np.empty(n_learners)
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    for i in range(n_learners):
        loss_per_learners_train[i] = adaboost.partial_loss(train_X, train_y, i + 1)
        loss_per_learners_test[i] = adaboost.partial_loss(test_X, test_y, i + 1)

    fig1 = go.Scatter(x=np.arange(1, n_learners + 1), y=loss_per_learners_train, mode='markers+lines',
                      marker=dict(color="black"), name="train")
    fig2 = go.Scatter(x=np.arange(1, n_learners + 1), y=loss_per_learners_test, mode='markers+lines',
                      marker=dict(color="blue"), name="test")
    fig = go.Figure([fig1, fig2])
    fig.update_layout(title=rf"$\textbf{{Error by number of learners}}$",
                      xaxis={"title": "number of learners"},
                      yaxis={"title": "loss"})
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{number of learners: {m}}}$" for m in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)

    for i in range(len(T)):
        t = T[i]

        def predict(X):
            return adaboost.partial_predict(X, t)

        fig.add_traces([decision_surface(predict, lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, symbol='circle', colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"$\textbf{{Decision boundaries by number of learners}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # # Question 3: Decision surface of best performing ensemble
    best_ensemble_loss = 1
    best_ensemble_size = 1
    for i in range(1, n_learners + 1):
        loss = adaboost.partial_loss(test_X, test_y, i)
        if loss < best_ensemble_loss:
            best_ensemble_loss = loss
            best_ensemble_size = i

    def best_ensemble_predict(X):
        return adaboost.partial_predict(X, best_ensemble_size)

    best_ensemble = make_subplots(rows=1, cols=1, horizontal_spacing=0.01, vertical_spacing=.03)

    best_ensemble.add_traces([decision_surface(best_ensemble_predict, lims[0], lims[1], showscale=False),
                              go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                         marker=dict(color=test_y, symbol='circle', colorscale=[custom[0], custom[-1]],
                                                     line=dict(color="black", width=1)))],
                             rows=1, cols=1)
    best_ensemble.update_layout(
        title=rf"$\textbf{{Best ensemble size: {best_ensemble_size} Accuracy: {1 - best_ensemble_loss}}}$",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    best_ensemble.show()

    # Question 4: Decision surface with weighted samples
    point_size = adaboost.D_ / np.max(adaboost.D_) * 5
    train_vis = make_subplots(rows=1, cols=1, horizontal_spacing=0.01, vertical_spacing=.03)

    def full_ensemble_predict(X):
        return adaboost.partial_predict(X, 250)

    train_vis.add_traces([decision_surface(full_ensemble_predict, lims[0], lims[1], showscale=False),
                          go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                     marker=dict(size=point_size, color=train_y, symbol='circle',
                                                 colorscale=[custom[0], custom[-1]],
                                                 line=dict(color="black", width=1)))],
                         rows=1, cols=1)
    train_vis.update_layout(
        title=rf"$\textbf{{Weighted samples decisions boundaries}}$",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    train_vis.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
