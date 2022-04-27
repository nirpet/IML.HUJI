from plotly.subplots import make_subplots

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    df = pd.get_dummies(df, columns=['zipcode'])
    df = df[df['bedrooms'] < 20]
    samples = df.drop(columns='price')
    response = df['price']
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
        cov = np.cov(feature_data, y)[0][1]
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
    std_per_percent = np.empty(91)
    max_range_per_percent = np.empty(91)
    min_range_per_percent = np.empty(91)
    sample_percent = np.arange(90) + 10

    for i in sample_percent:
        loss_per_run = np.empty(10)
        for j in range(10):
            lr = LinearRegression(include_intercept=True)
            X_shuffled = train_x.sample(frac=1)
            y_shuffled = train_y.reindex_like(X_shuffled)
            last_index = int(len(train_x) * i / 100)
            lr.fit(X_shuffled.to_numpy()[:last_index], y_shuffled.to_numpy()[:last_index])
            loss = lr.loss(test_x.to_numpy(), test_y.to_numpy())
            loss_per_run[j] = loss

        loss_per_percent[i - 10] = loss_per_run.mean()
        std_per_percent[i - 10] = loss_per_run.std()

    fig = go.Figure([go.Scatter(x=sample_percent, y=loss_per_percent, mode="markers+lines", name="Mean Prediction",
                                line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
                     go.Scatter(x=sample_percent, y=loss_per_percent - 2 * std_per_percent, fill=None, mode="lines",
                                line=dict(color="lightgrey"), showlegend=False),
                     go.Scatter(x=sample_percent, y=loss_per_percent + 2 * std_per_percent, fill='tonexty',
                                mode="lines",
                                line=dict(color="lightgrey"), showlegend=False)])

    fig.update_layout(
        title_text=rf"$\text{{Loss and confidence per percentage of train size}}$",
        xaxis={"title": r"$sample percent$"},
        yaxis={"title": r"$loss$"})
    fig.show()
