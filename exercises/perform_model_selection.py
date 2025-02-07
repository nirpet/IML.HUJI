from __future__ import annotations

import plotly.graph_objects as go
import sklearn.linear_model
from sklearn import datasets

from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from IMLearn.metrics import mean_square_error, loss_functions
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test
from utils import *


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    epsilon = np.random.normal(0, noise, n_samples)
    X = np.linspace(-1.2, 2, n_samples)
    noiseless_y = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    y = noiseless_y + epsilon

    X_train, y_train, X_test, y_test = split_train_test(pd.Series(X), pd.Series(y), 2.0 / 3)
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    X_indices = X.argsort()
    fig0 = go.Scatter(x=X[X_indices], y=noiseless_y[X_indices], mode='markers',
                      marker=dict(color="black"), name="true noiseless model")
    fig1 = go.Scatter(x=X_train, y=y_train, mode='markers',
                      marker=dict(color="red"), name="train")
    fig2 = go.Scatter(x=X_test, y=y_test, mode='markers',
                      marker=dict(color="blue"), name="test")
    fig = go.Figure([fig0, fig1, fig2])
    fig.update_layout(
        title=rf"$\textbf{{The true model and train and test samples. Noise: {noise}, samples: {n_samples} }}$",
        xaxis={"title": "y"},
        yaxis={"title": "X"})
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    degrees = np.arange(11)
    train_scores = np.empty(11)
    validation_scores = np.empty(11)
    for k in degrees:
        train_scores[k], validation_scores[k] = cross_validate(PolynomialFitting(k),
                                                               X_train, y_train, mean_square_error)

    fig1 = go.Scatter(x=degrees, y=train_scores, mode='lines',
                      marker=dict(color="red"), name="train")
    fig2 = go.Scatter(x=degrees, y=validation_scores, mode='lines',
                      marker=dict(color="blue"), name="validation")
    fig = go.Figure([fig1, fig2])
    fig.update_layout(
        title=rf"$\textbf{{Average train and validation based on polynomial degree. Noise: {noise}, samples: {n_samples}}}$",
        xaxis={"title": "score"},
        yaxis={"title": "degrees"})
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    min_k = validation_scores.argmin()
    best_polynomial_degree_estimator = PolynomialFitting(min_k)
    best_polynomial_degree_estimator.fit(X_train, y_train)
    test_error = best_polynomial_degree_estimator.loss(X_test, y_test)
    print("best degree: " + str(min_k) + " error: " + str(round(test_error, 2)))
    print("validation score: " + str(round(validation_scores[min_k], 2)))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    diabetes = datasets.load_diabetes()
    X = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target)
    X_train = X[:n_samples].to_numpy()
    y_train = y[:n_samples].to_numpy()
    X_test = X[n_samples:].to_numpy()
    y_test = y[n_samples:].to_numpy()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    regularization = np.linspace(0, 2.5, n_evaluations)
    ridge_train_score = np.empty(n_evaluations)
    ridge_validation_score = np.empty(n_evaluations)
    lasso_train_score = np.empty(n_evaluations)
    lasso_validation_score = np.empty(n_evaluations)

    for k in range(n_evaluations):
        ridge_train_score[k], ridge_validation_score[k] = \
            cross_validate(RidgeRegression(regularization[k]), X_train, y_train, mean_square_error)
        lasso_train_score[k], lasso_validation_score[k] = \
            cross_validate(sklearn.linear_model.Lasso(alpha=regularization[k]), X_train, y_train, mean_square_error)

    fig1 = go.Scatter(x=regularization, y=ridge_train_score, mode='lines',
                      marker=dict(color="red"), name="train")
    fig2 = go.Scatter(x=regularization, y=ridge_validation_score, mode='lines',
                      marker=dict(color="blue"), name="validation")
    fig = go.Figure([fig1, fig2])
    fig.update_layout(
        title=rf"$\textbf{{Ridge average train and validation based on regularization param}}$",
        xaxis={"title": "score"},
        yaxis={"title": "lambda"})
    fig.show()

    fig1 = go.Scatter(x=regularization, y=lasso_train_score, mode='lines',
                      marker=dict(color="red"), name="train")
    fig2 = go.Scatter(x=regularization, y=lasso_validation_score, mode='lines',
                      marker=dict(color="blue"), name="validation")
    fig = go.Figure([fig1, fig2])
    fig.update_layout(
        title=rf"$\textbf{{Lasso average train and validation based on regularization param}}$",
        xaxis={"title": "score"},
        yaxis={"title": "lambda"})
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    min_lam_ridge = regularization[ridge_validation_score.argmin()]
    best_ridge_lam_estimator = RidgeRegression(min_lam_ridge)
    best_ridge_lam_estimator.fit(X_train, y_train)
    print("Ridge best regularization param: " + str(min_lam_ridge))

    min_lam_lasso = regularization[lasso_validation_score.argmin()]
    best_lasso_lam_estimator = sklearn.linear_model.Lasso(alpha=min_lam_lasso)
    best_lasso_lam_estimator.fit(X_train, y_train)
    print("Lasso best regularization param: " + str(min_lam_lasso))

    ridge_estimator = RidgeRegression(min_lam_ridge)
    lasso_estimator = sklearn.linear_model.Lasso(alpha=min_lam_lasso)
    least_squares_estimator = LinearRegression()

    ridge_estimator.fit(X_train, y_train)
    lasso_estimator.fit(X_train, y_train)
    least_squares_estimator.fit(X_train, y_train)

    ridge_score = ridge_estimator.loss(X_test, y_test)
    lasso_pred = lasso_estimator.predict(X_test)
    lasso_score = loss_functions.mean_square_error(y_test, lasso_pred)
    least_squares_score = least_squares_estimator.loss(X_test, y_test)

    print("Ridge score: " + str(round(ridge_score, 2)) + ", Lasso score: " + str(round(lasso_score, 2))
          + ", Least squares score: " + str(round(least_squares_score, 2)))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(100, 0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()
