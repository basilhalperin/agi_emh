import datetime
import numpy as np
import pandas as pd


def get_last_date(df):
    return pd.Series(
        {
            country: datetime.datetime.strftime(
                df[country].dropna().index[-1],
                '%Y-%m-%d'
            )
            for country in df
        }
    )


def get_last_value(df, last_dates):
    data = {}
    for country in last_dates.index:
        date = last_dates[country]
        data[country] = df[country].loc[date]
    return pd.Series(data)



def predictions_main(daily_rs_by_horizon, stackeds, regs, horizons=[10, 20]):
    # get last date
    last_daily_dates = {}
    for h in horizons:
        last_daily_dates[h] = get_last_date(daily_rs_by_horizon[h])

    # get last value
    last_daily_value = {}
    for h in horizons:
        last_daily_value[h] = get_last_value(daily_rs_by_horizon[h], last_daily_dates[h])

    # predictions
    for h in horizons:
        horizon_str = '{0}y'.format(h)

        num_data_points = len(stackeds[horizon_str][h])
        print("For horizon {0}, number of data points:".format(horizon_str), num_data_points)
        print("\n")

        print("{0}y real yield, last daily available value:".format(h))
        for country in last_daily_value[h].index:
            date = last_daily_dates[h].loc[country]
            rounded_value = np.round(last_daily_value[h].loc[country], 3)
            print(country, "({0}): {1}%".format(date, rounded_value))
        print("\n")

        print("Equation to project g based on r:")

        m = np.round(regs[horizon_str][h].params['r'], 3)
        b = np.round(regs[horizon_str][h].params['Intercept'], 3)
        print("y = {0}x + {1}".format(m, b))
        print("\n")

        print("Projection for g over next {0}y by country:".format(h))
        print(b + last_daily_value[h])
        print("\n\n----------------------------------------------\n\n")