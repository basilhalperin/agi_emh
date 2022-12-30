import requests
import io
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as pdr
import zipfile
import copy


SAVE_PATH = 'saved_data/'
US_TIPS_URL = 'https://www.federalreserve.gov/data/yield-curve-tables/feds200805.csv'
UK_TIPS_URL = 'https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/glcrealddata.zip'
UK_BE_URL = 'https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/glcinflationddata.zip'
UK_DATA_SAVE_PATH = SAVE_PATH + 'UK_data/'
MANUAL_DATA_PATH = 'manual_data/AU_CA_10.xlsx'
MANUAL_KEYS_MAP = {
    'Australia 10-Year Real Yield': {'country': 'Australia', 'horizon': 10},
    'Canada 10-Year Real Yield': {'country': 'Canada', 'horizon': 10},
}
RGDP_FRED_KEYS = {
    'GDPC1': 'US',
    'CLVMNACSCAB1GQUK': 'UK',
    # DOUBLE CHECK BELOW HERE -- e.g. units are correct
    'NGDPRSAXDCCAQ': 'Canada',
    'NGDPRSAXDCAUQ': 'Australia',
    #'CLVMNACSCAB1GQFR': 'France',
    #'CLVMNACSCAB1GQDE': 'Germany',
    #'JPNRGDPEXP': 'Japan',
    #'NGDPRSAXDCZAQ': 'South Africa',
    #'CLVMNACSCAB1GQSE': 'Sweden',
    #'NAEXKP01ILQ652S': 'Israel',
}
CPI_FRED_KEYS = {
    'CPIAUCSL': 'US',
    'CPRPTT01GBM661N': 'UK',
}
US_HISTORICAL_URL = 'https://data.nber.org/data/abc/abcq.csv'
UK_HISTORICAL_RPI_URL = 'https://www.ons.gov.uk/generator?format=csv&uri=/economy/inflationandpriceindices/timeseries/chaw/mm23'
JST_URL = "https://www.macrohistory.net/app/download/9834512569/JSTdatasetR5.xlsx"
HORIZONS = [5, 10, 15, 20, 30]
END = '2021'


def last(x):
    return x.last()


def bfill(x):
    return x.bfill()


def get_us_srs(df, k, freq=None, resample=last, end=None, horizons=[5, 10, 15, 20]):
    horizon_to_horizon_str = {horizon: ('0' + str(horizon))[-2:] for horizon in horizons}
    horizon_to_k = {horizon: k+horizon_to_horizon_str[horizon] for horizon in horizons}
    result = {}
    for horizon in horizon_to_k:
        name = horizon_to_k[horizon]
        tmp = df[name]
        if freq:
            tmp = resample(tmp.resample(freq))
        if end:
            tmp = tmp.loc[:end]
        result[horizon] = tmp
    return pd.DataFrame(result)


def load_us_tips_fed(url=US_TIPS_URL, freq='A', resample=last, end=END):
    """
    Documentation: https://www.federalreserve.gov/data/tips-yield-curve-and-inflation-compensation.htm
    """
    response = requests.get(url)
    csv_file = io.BytesIO(response.content)
    us_daily = pd.read_csv(csv_file, skiprows=18, index_col=0)
    us_daily.index = pd.to_datetime(us_daily.index)
    us_daily.index.name = None

    us_be = get_us_srs(us_daily, 'BKEVEN', freq=freq, resample=resample, end=end)
    us_r = get_us_srs(us_daily, 'TIPSY', freq=freq, resample=resample, end=end)

    return us_be, us_r, us_daily


def load_us_tips_30(freq='A', resample=last, end=END):
    tips30 = pdr.DataReader('DFII30', 'fred', start='1900')['DFII30']
    tips30.index.name = None
    if freq:
        tips30 = resample(tips30.resample(freq))
    if end:
        tips30 = tips30.loc[:end]
    return tips30


def load_us_tips(
    url=US_TIPS_URL, freq='A', resample=last, end=END,
    save=False, save_path=SAVE_PATH,
):
    us_be, us_r, us_daily = load_us_tips_fed(url, freq, resample, end)
    tips30 = load_us_tips_30(freq, resample, end)
    us_r[30] = tips30

    if save:
        us_be.to_csv(save_path+'us_be.csv')
        us_r.to_csv(save_path+'us_r.csv')
        us_daily.to_csv(save_path+'us_raw_tips.csv')

    return us_be, us_r, us_daily


def download_uk(url=UK_TIPS_URL, uk_data_save_path=UK_DATA_SAVE_PATH, save_name='BOE_raw.zip'):
    """
    Documentation: https://www.bankofengland.co.uk/statistics/yield-curves
    """
    response = requests.get(url)

    filename = uk_data_save_path + save_name
    with open(filename, 'wb') as f:
        f.write(response.content)

    cwd = os.getcwd()
    os.chdir(uk_data_save_path)
    with zipfile.ZipFile('BOE_raw.zip', 'r') as zip_ref:
        zip_ref.extractall()
    os.chdir(cwd)


def clean_uk(
    uk_data_save_path=UK_DATA_SAVE_PATH, filename_k='GLC Real daily',
    sheetname_k='4.  real spot curve', alt_sheetname_k='4. spot curve',
    freq='A', resample=last, end=END, horizons=HORIZONS,
    save=False, save_path=SAVE_PATH, save_name='r',
):
    uk_filenames = [uk_data_save_path + '/' + x for x in os.listdir(uk_data_save_path)
                    if filename_k in x and '~$' not in x]
    uk_all = {}
    for f in uk_filenames:
        sheet_names = pd.ExcelFile(f).sheet_names
        if sheetname_k in sheet_names:
            df = pd.read_excel(f, sheet_name=sheetname_k,
                               skiprows=3, header=0, index_col=0)
        elif alt_sheetname_k in sheet_names:
            df = pd.read_excel(f, sheet_name=alt_sheetname_k,
                               skiprows=3, header=0, index_col=0)
        else:
            raise ValueError("Expected sheet name(s) not found for {0} :|".format(f))
        df = df.iloc[1:]  # blank first row below the header
        uk_all[f] = df

    uk_daily = pd.concat(list(uk_all.values()), axis=0)
    uk_daily.index.name = None

    if freq:
        uk_r = resample(uk_daily.resample(freq))
        uk_r = uk_r[horizons]
    else:
        uk_r = uk_daily[horizons]
    if end:
        uk_r = uk_r.loc[:end]

    try:
        uk_r.columns = uk_r.columns.astype(int)
    except:
        pass

    if save:
        uk_r.to_csv(save_path+'uk_tips_{0}.csv'.format(save_name))
        uk_daily.to_csv(save_path+'uk_raw_{0}.csv'.format(save_name))

    return uk_daily, uk_r


def load_uk(
    url=UK_TIPS_URL, uk_data_save_path=UK_DATA_SAVE_PATH,
    freq='A', resample=last, end=END,
    filename_k='GLC Real daily', sheetname_k='4.  real spot curve', save_name='r',
    save=False, save_path=SAVE_PATH
):
    download_uk(url=url, uk_data_save_path=uk_data_save_path)
    uk_daily, uk_r = clean_uk(uk_data_save_path=uk_data_save_path, freq=freq, resample=resample, end=end,
                              filename_k=filename_k, sheetname_k=sheetname_k, save_name=save_name,
                              save=save, save_path=save_path)
    return uk_daily, uk_r


def load_uk_be(
):
    """This function just exists for easy reference"""
    uk_daily, uk_r = load_uk(
        url=UK_BE_URL, filename_k='GLC Inflation daily', sheetname_k='4.  inf spot curve', save_name='be',
    )
    return uk_daily, uk_r


def load_manual(path=MANUAL_DATA_PATH, keys_map=MANUAL_KEYS_MAP, freq='A', resample=last, end=END):
    df = pd.read_excel(path, index_col=0)
    countries = set([keys_map[key]['country'] for key in keys_map])
    horizons = set([keys_map[key]['horizon'] for key in keys_map])

    manual_rs_by_horizon = {h: {} for h in horizons}
    manual_rs_by_country = {
        country: {
            horizon: None
            for horizon in horizons
        }
        for country in countries
    }
    for key in keys_map:
        country = keys_map[key]['country']
        horizon = keys_map[key]['horizon']
        manual_rs_by_horizon[horizon][country] = df[key].dropna()
        manual_rs_by_country[country][horizon] = df[key].dropna()

    for h in horizons:
        manual_rs_by_horizon[h] = pd.DataFrame(manual_rs_by_horizon[h])
        manual_rs_by_horizon[h] = manual_rs_by_horizon[h].loc[:end]

    for country in countries:
        manual_rs_by_country[country] = pd.DataFrame(manual_rs_by_country[country])
        manual_rs_by_country[country] = manual_rs_by_country[country].loc[:end]

    if resample or freq:
        for h in horizons:
            manual_rs_by_horizon[h] = resample(manual_rs_by_horizon[h].resample(freq))
        for country in countries:
            manual_rs_by_country[country] = resample(manual_rs_by_country[country].resample(freq))

    return manual_rs_by_horizon, manual_rs_by_country


def dict_by_country_to_by_horizon(dict_by_country):
    all_horizons = [dict_by_country[country].columns.tolist() for country in dict_by_country]
    all_horizons = np.hstack(all_horizons)
    #all_horizons = all_horizons.astype(int)
    all_horizons = set(all_horizons)

    dict_by_horizon = {h: {} for h in all_horizons}
    for country in dict_by_country:
        for h in all_horizons:
            if h in dict_by_country[country].columns:
                dict_by_horizon[h][country] = dict_by_country[country][h]

    dict_by_horizon = {h: pd.DataFrame(dict_by_horizon[h]) for h in dict_by_horizon}
    return dict_by_horizon


def reindex_gdp(gdp):
    if type(gdp) == pd.DataFrame:
        new = {}
        for col in gdp:
            start = gdp[col].dropna().index[0]
            new[col] = gdp[col]/gdp[col].loc[start]
        new = pd.DataFrame(new)
    elif type(gdp) == pd.Series:
        start = gdp.dropna().index[0]
        new = gdp/gdp.loc[start]
    else:
        new = pd.Series()
    return new


def load_fred(
    keys=RGDP_FRED_KEYS, freq='A', resample=last, end='2021', reindex=True,
    save=False, save_path=SAVE_PATH, save_name='rgdp'
):
    rgdp = pdr.DataReader(list(keys.keys()), 'fred', start='1800', end=end)
    rgdp = rgdp.rename(columns=keys)
    if freq:
        rgdp = resample(rgdp.resample(freq))
    rgdp.index.name = None

    if reindex:
        rgdp = reindex_gdp(rgdp)

    if save:
        rgdp.to_csv(save_path+'{0}.csv'.format(save_name))

    return rgdp


def load_us_historical_gdp(
        url=US_HISTORICAL_URL, freq='A', resample=last, reindex=True,
        save=False, save_path=SAVE_PATH
):
    """
    Documentation: https://www.nber.org/research/data/tables-american-business-cycle
    """
    # possibly this is on FRED somewhere too
    df = pd.read_csv(url)
    df['date'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df[['GNP', 'RGNP72', 'GNPDEF']]
    df.columns = ['NGDP', 'RGDP', 'GNPDEF']
    if freq:
        df = resample(df.resample(freq))
    if reindex:
        df = reindex_gdp(df)
    if save:
        df.to_csv(save_path+'us_historical.csv')

    return df['RGDP'], df['NGDP'], df['GNPDEF']


def load_uk_historical_gdp(
        historical_rpi_url=UK_HISTORICAL_RPI_URL, reindex=True,
        save=False, save_path=SAVE_PATH,
):
    # TODO: historical UK R/N GDP

    r = requests.get(historical_rpi_url)
    rpi2 = pd.read_csv(io.BytesIO(r.content), skiprows=9, header=None, index_col=0)
    rpi2 = rpi2[1]
    rpi2.name = None
    rpi2.index.name = None

    # has annual/quarterly/monthly; just take annual
    desired = [x for x in rpi2.index if ' ' not in str(x)]
    rpi2 = rpi2.loc[desired]
    rpi2.index = pd.to_datetime(rpi2.index.astype(str))
    rpi2 = rpi2.resample('A').last()  # align dates

    if reindex:
        rpi2 = reindex_gdp(rpi2)

    if save:
        rpi2.to_csv(save_path+'uk_historical.csv')

    return pd.Series(dtype=float), pd.Series(dtype=float), rpi2


def combine_historical(
        rgdp, rgdp_historical, end=END,
        save=False, save_path='saved_data/', save_name=''
):
    rgdp_combo = {}
    for country in rgdp:
        tmp = pd.DataFrame({
            'modern': rgdp[country],
            'historical': rgdp_historical[country] if country in rgdp_historical else None
        })

        start_modern = tmp['modern'].dropna().index[0]
        start_historical = tmp['historical'].dropna()
        if len(start_historical) > 0:
            start_historical = start_historical.index[0]
            start = max(start_modern, start_historical)
        else:
            start = start_modern

        tmp = tmp/tmp.loc[start]
        rgdp_combo[country] = tmp.dropna(axis=1, how='all').median(axis=1)
    rgdp_combo = pd.DataFrame(rgdp_combo)

    if end:
        rgdp_combo = rgdp_combo.loc[:end]

    if save:
        rgdp_combo.to_csv(save_path+save_name+'.csv')

    return rgdp_combo


def change_next_x(df, horizon):
    result = (df.shift(-horizon)/df)**(1/horizon) - 1
    result *= 100
    return result


def get_change_over_horizons(rgdp, horizons=HORIZONS):
    return {
        k: change_next_x(rgdp, k)
        for k in horizons
    }


def load_jst(url=JST_URL):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    data = pd.read_excel(url, sheet_name='Data')
    des = pd.read_excel(url, sheet_name='Variable description', header=None)

    return data, des
