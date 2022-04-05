import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

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
    data_frame = pd.read_csv(filename, parse_dates=['Date'], date_parser=dateparse)
    day_of_year = np.empty(len(data_frame['Date']))
    for i in range(len(data_frame['Date'])):
        day_of_year[i] = data_frame['Date'][i].timetuple().tm_yday

    data_frame.insert(data_frame.shape[1], 'DayOfYear', day_of_year)
    data_frame = data_frame.drop(columns='Date')
    data_frame = data_frame[data_frame['Temp'] >= -35]
    return data_frame


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
    country_temp = country_temp.reset_index()
    fig = px.line(country_temp, x='Month', y='mean', color='Country', error_y='std', line_group='Country',
                  title='Month average temperature of different countries')
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    samples = pd.Series(israel_temps['DayOfYear'].values)
    response = pd.Series(israel_temps['Temp'].values)
    train_x, train_y, test_x, test_y = split_train_test(samples, response)
    loss_per_degree = np.empty(10)

    for k in range(1, 11):
        pf = PolynomialFitting(k)
        pf.fit(train_x.to_numpy(), train_y.to_numpy())
        loss_per_degree[k - 1] = round(pf.loss(test_x.to_numpy(), test_y.to_numpy()), 2)

    fig = px.bar(x=range(1, 11), y=loss_per_degree, title='Loss per polynomial fitting degree')
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    k = 5
    pf = PolynomialFitting(k)
    pf.fit(samples.to_numpy(), response.to_numpy())

    countries = df['Country'].unique()
    error_by_country = np.empty(len(countries))

    for i in range(len(countries)):
        country = countries[i]
        country_temps = df[df['Country'] == country]
        samples = pd.Series(country_temps['DayOfYear'].values)
        response = pd.Series(country_temps['Temp'].values)
        error_by_country[i] = pf.loss(samples.to_numpy(), response.to_numpy())

    fig = px.bar(x=countries, y=error_by_country, title='Error per country')
    fig.show()
