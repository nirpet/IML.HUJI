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

    dateparse = lambda dates: [datetime.strptime(d, '%Y-%m-%d') for d in dates]
    full_data = pd.read_csv(filename, parse_dates=['Date'], date_parser=dateparse)
    day_of_year = np.empty(len(full_data['Date']))
    for i in range(len(full_data['Date'])):
        day_of_year[i] = full_data['Date'][i].timetuple().tm_yday

    full_data.insert(full_data.shape[1], 'DayOfYear', day_of_year)
    full_data = full_data.drop(columns='Date')
    return full_data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('../datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    israel_temps = df.loc[df['Country'] == 'Israel']

    fig = px.scatter(x=israel_temps['DayOfYear'], y=israel_temps['Temp'],
                     color=israel_temps['Year'].astype(str),
                     title="Israel temperatures based on day of year")
    fig.show()

    # Question 2 - Exploring data for specific country
    standard_deviations_per_months = israel_temps.groupby('Month')['Temp'].agg('std')

    fig = px.bar(standard_deviations_per_months, y='Temp',
                 title="Israel temperatures std by month")
    fig.show()

    # Question 3 - Exploring differences between countries
    country_temp = df.groupby(['Country', 'Month'])['Temp'].agg(['mean', 'std'])
    # fig = px.line(country_temp, x='Month', y='mean', color='Country', error_y='std',
    #               title='Month average temperature of different countries')
    # fig.show()

    # Question 4 - Fitting model for different values of `k`

    samples = df.drop(columns='Temp')
    samples = pd.get_dummies(data=samples, columns=['Country', 'City'])
    response = pd.Series(df['Temp'].values)
    train_x, train_y, test_x, test_y = split_train_test(samples, response)
    loss_per_degree = np.empty(10)

    for k in range(1, 10):
        pf = PolynomialFitting(k)
        pf.fit(train_x.to_numpy(), train_y.to_numpy())
        loss_per_degree[k - 1] = round(pf.loss(test_x.to_numpy(), test_y.to_numpy()), 2)

    fig = px.bar(y=loss_per_degree, title='Loss per polynomial fitting degree')
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    k = 3
    pf = PolynomialFitting(k)
    countries = df['Country'].unique()
    error_by_country = np.empty(len(countries))

    for i in range(len(countries)):
        country = countries[i]
        country_temps = df.loc[df['Country'] == country]
        samples = df.drop(columns='Temp')
        samples = pd.get_dummies(data=samples, columns=['Country', 'City'])
        response = pd.Series(df['Temp'].values)
        train_x, train_y, test_x, test_y = split_train_test(samples, response)
        pf.fit(train_x.to_numpy(), train_y.to_numpy())
        error_by_country[i] = pf.loss(test_x.to_numpy(), test_y.to_numpy())

    fig = px.bar(x=countries, y=error_by_country, title='Error per country')
    fig.show()
