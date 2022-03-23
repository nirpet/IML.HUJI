from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    univariate_gaussian = UnivariateGaussian()
    mean = 10
    sigma = 1
    sample_size = 1000

    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(mean, sigma, sample_size)
    univariate_gaussian.fit(X)
    print('(' + str(univariate_gaussian.mu_) + ', ' + str(univariate_gaussian.var_) + ')')

    # # Question 2 - Empirically showing sample mean is consistent
    sample_sizes = np.arange(10, 1010, 10)
    estimation_dist = np.empty(100)
    for i in range(100):
        estimation_dist[i] = np.abs(mean - np.mean(X[:sample_sizes[i]]))

    fig = go.Figure([go.Scatter(x=sample_sizes, y=estimation_dist, mode='markers+lines', marker=dict(color="black"))],
                    layout=go.Layout(title_text=r"$\text{Distance from actual mean by sample size}$",
                                     xaxis_title="sample size",
                                     yaxis_title="distance",
                                     width=800,
                                     height=600))
    fig.show()

    # # Question 3 - Plotting Empirical PDF of fitted model
    X.sort()
    pdf_values = univariate_gaussian.pdf(X)
    fig2 = go.Figure([go.Scatter(x=X, y=pdf_values, mode='markers+lines', marker=dict(color="black"))],
                     layout=go.Layout(title_text=r"$\text{PDF values of ordered samples}$",
                                      xaxis_title="sample",
                                      yaxis_title="PDF",
                                      width=800,
                                      height=600))
    fig2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    multivariate_gaussian = MultivariateGaussian()
    mean = [0, 0, 4, 0]
    cov = [[1, 0.2, 0, 0.5],
           [0.2, 2, 0, 0],
           [0, 0, 1, 0],
           [0.5, 0, 0, 1]]

    X = np.random.multivariate_normal(mean, cov, 1000)
    multivariate_gaussian.fit(X)

    print(multivariate_gaussian.mu_)
    print(multivariate_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    mean_values = np.linspace(-10, 10, 200)
    log_likelihood_data = np.empty((200, 200))
    for i in range(200):
        for j in range(200):
            mu = [mean_values[i], 0, mean_values[j], 0]
            log_likelihood_data[i][j] = multivariate_gaussian.log_likelihood(mu, cov, X)

    fig = go.Figure([go.Heatmap(x=mean_values, y=mean_values, z=log_likelihood_data,
                                colorbar=dict(title='log likelihood'))],
                    layout=go.Layout(title_text=r"$\text{Log likelihood of mean values of the given sample}$",
                                     xaxis_title="f3",
                                     yaxis_title="f1",
                                     width=600,
                                     height=600))
    fig.show()

    # Question 6 - Likelihood evaluation
    # The maximum likelihood is achieved for the following values, f1=-0.050, f3=3.970


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
