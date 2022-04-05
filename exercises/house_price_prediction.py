from plotly.subplots import make_subplots

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = df.drop(columns='id')
    df = df.drop(columns='date')
    df = df.dropna()
    # todo: remove corrupted data, add features, and create dummies
    df = pd.get_dummies(df, columns=['zipcode'])
    samples = df.drop(columns='price')
    response = pd.Series(df['price'].values)
    return samples, response


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    for (feature_name, feature_data) in X.iteritems():
        cov = np.cov(feature_data, y)[0][0]
        data_std = np.std(feature_data)
        response_std = np.std(y)
        pcf = cov / (data_std * response_std)
        title = feature_name + " Pearson Correlation " + str(pcf)
        fig = go.Figure(
            [go.Scatter(x=feature_data, y=y, mode='markers', marker=dict(color="black"))],
            layout=go.Layout(title_text=r"$\text{" + title + "}$",
                             xaxis_title=feature_name,
                             yaxis_title='price',
                             width=800,
                             height=600))
        fig.write_image(output_path + "\\" + feature_name + ".jpeg")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    loss_per_percent = np.empty(91)
    sample_percent = np.arange(90) + 10

    for j in range(10):
        for i in sample_percent:
            lr = LinearRegression(include_intercept=True)
            last_index = int(len(train_x) * i / 100)
            lr.fit(X.to_numpy()[:last_index], y.to_numpy()[:last_index])
            results = lr.predict(test_x.to_numpy())
            loss_per_percent[i - 10] = lr.loss(results, test_y.to_numpy())

    xx, yy = np.meshgrid(np.arange(91) + 9, loss_per_percent)
    z = xx ** 2 + yy ** 2
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter'}, {'type': 'scene'}]])

    fig.add_traces(data=[
        go.Contour(z=z, colorscale='Electric', showscale=False),
        go.Surface(x=sample_percent, y=loss_per_percent, z=z, opacity=.8, colorscale='Electric',
                   contours=dict(z=dict(show=True)))],
        rows=[1, 1], cols=[1, 2])

    fig.update_layout(width=800, height=300, scene_aspectmode="cube",
                      scene=dict(camera=dict(eye=dict(x=-1.5, y=-1.5, z=.2))))
    fig.show()
