import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "simple_white"
from datetime import datetime


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    dateparse = lambda dates: [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dates]
    full_data = pd.read_csv(filename, parse_dates=['Date'], date_parser=dateparse)
    full_data.dropna()
    day_of_year = np.empty(len(full_data['Date']))
    for i in range(len(full_data['Date'])):
        day_of_year[i] = full_data['Date'][i].timetuple().tm_yday

    full_data.insert(day_of_year.shape[0], 'DayOfYear', day_of_year)
    return full_data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data('City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    israel_temps = data.loc[data['Country'] == 'Israel']

    fig = go.Figure([go.Scatter(x=israel_temps['DayOfYear'], y=israel_temps['Temperature'], mode='markers+lines',
                                marker=dict(color="black"))],
                    layout=go.Layout(title_text=r"$\text{Distance from actual mean by sample size}$",
                                     xaxis_title="Day of year",
                                     yaxis_title="temperature",
                                     width=800,
                                     height=600))
    fig.show()

    # Question 2 - Exploring data for specific country
    standard_deviations_per_months = israel_temps.groupby('Month')['Temperature'].agg('std')

    fig = px.bar(standard_deviations_per_months, x='month', y='std of temperature')
    fig.show()

    # Question 3 - Exploring differences between countries
    country_temp = data.groupby('Country', 'Month')['Temperature'].agg('avg', 'std')
    fig = px.line(country_temp, x="month", y="average temp", error_y=country_temp['std'],
                  title='Month average temperature of different countries')
    fig.show()

    # Question 4 - Fitting model for different values of `k`

    train_x, train_y, test_x, test_y = split_train_test(israel_temps.loc[:, israel_temps.columns != 'Temperature'],
                                                        israel_temps['Temperature'])
    loss_per_degree = np.empty(10)

    for k in range(1, 10):
        pf = PolynomialFitting(k)
        pf.fit(train_x, test_y)
        loss_per_degree[k - 1] = round(pf.loss(test_x, test_y), 2)

    fig = px.bar(loss_per_degree, x='degree', y='loss', title='Loss per polynomial fitting degree')
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    k = 5
    pf = PolynomialFitting(k)
    countries = data['Country'].unique()
    error_by_country = np.empty(len(countries))

    for i in len(countries):
        country = countries[i]
        country_temps = data.loc[data['Country'] == country]
        samples = country_temps.loc[:, country_temps.columns != 'Temperature']
        results = country_temps['Temperature']
        train_x, train_y, test_x, test_y = split_train_test(samples, results)
        pf.fit(train_x, train_y)
        error_by_country[i] = pf.loss(test_x, test_y)

    fig = px.bar(error_by_country, x='country', y='loss', title='Loss per country')
    fig.show()
