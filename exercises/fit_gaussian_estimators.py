from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    univariate_gaussian = UnivariateGaussian()
    mean = 10
    var = 1
    sample_size = 1000

    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(mean, var, sample_size)
    univariate_gaussian.fit(X)
    print('(' + str(univariate_gaussian.mu_) + ', ' + str(univariate_gaussian.var_) + ')')

    # # Question 2 - Empirically showing sample mean is consistent
    sample_sizes = np.arange(10, 1010, 10)
    estimation_dist = np.empty(100)
    for i in range(100):
        estimation_dist[i] = np.abs(mean - np.mean(X[:sample_sizes[i]]))

    fig = go.Figure([go.Scatter(x=sample_sizes, y=estimation_dist, mode='markers+lines', marker=dict(color="black"))],
            layout=go.Layout(title_text=r"$\text{distance from actual mean by sample size}$", height=300))
    fig.show()


    # # Question 3 - Plotting Empirical PDF of fitted model
    # raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()
