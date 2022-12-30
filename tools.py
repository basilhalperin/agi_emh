import numpy as np
import pandas as pd
import statsmodels.api as sm


def get_rolling_regs_attr(regs, lambda_func):
    return pd.DataFrame({
        i: lambda_func(regs[i])
        if regs[i] is not np.nan
        else np.nan
        for i in regs
    })


def rolling_regression(data, formula, window, nan_if_missing_data=True):
    regs = {}
    for i in range(window, len(data)):
        start_date = data.index[i-window]
        end_date = data.index[i]
        tmp_data = data.loc[start_date:end_date]

        if tmp_data.unstack().isnull().sum() == 0:
            regs[end_date] = sm.OLS.from_formula(formula=formula, data=tmp_data).fit()
        else:
            if nan_if_missing_data:
                regs[end_date] = np.nan
            else:
                regs[end_date] = sm.OLS.from_formula(formula=formula, data=tmp_data).fit()

    params = get_rolling_regs_attr(regs, lambda x: x.params)
    bse = get_rolling_regs_attr(regs, lambda x: x.bse)

    return regs, params, bse


def inflation_ar1(srs, window):
    tmp = pd.DataFrame({
        'inflation': srs,
        'lag_inflation': srs.shift(1),
    })

    regs, params, bses = rolling_regression(
        tmp, 'inflation ~ lag_inflation', window
    )

    forecast = params.loc['lag_inflation']*tmp['inflation'] + params.loc['Intercept']
    forecast = forecast.shift(1)

    return regs, params, bses, forecast


def get_inflation_ar1(inflation, window=20):
    regss = {}
    paramss = {}
    bses = {}
    ar_forecasts = {}
    for country in inflation:
        srs = inflation[country]
        (regss[country], paramss[country], bses[country],
         ar_forecasts[country]) = inflation_ar1(srs, window)

    ar_forecasts = pd.DataFrame(ar_forecasts)
    return ar_forecasts, regss, paramss, bses
