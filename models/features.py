from arch.unitroot import PhillipsPerron
from datetime import date
import numpy as np
import pandas as pd
from scipy.signal import periodogram, welch
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller, kpss


def compute_acf(data: pd.Series):
    n_lags = len(data) // 2 + 1
    values = pd.DataFrame(data.values)
    dataframe = pd.concat([values.shift(i) for i in range(n_lags)], axis=1)
    dataframe.columns = ['t'] + ['t+' + str(i) for i in range(1, n_lags)]
    return dataframe.corr()['t'].values, n_lags


def get_k_periods(data: pd.Series, k: int, method: str = 'periodogram'):
    if method == 'periodogram':
        _, y = periodogram(data.values)
    else:
        _, y = welch(data.values)
    values_and_lags = [(value, lag) for value, lag in zip(y, range(len(y)))]
    values_and_lags.sort(reverse=True)
    return [lag for _, lag in values_and_lags[:k]]


def get_stationary_tests_results(data: pd.Series, methods: list[str] = None, regression: list[str] = None):
    if methods is None:
        methods = ['adf']
    if regression is None:
        regression = ['c']

    results = dict()

    for method in methods:
        for regressor in regression:
            if method == 'adf':
                results[method + '_' + regressor] = adfuller(data, regression=regressor)[1]
            elif method == 'kpss' and regressor != 'ctt':
                results[method + '_' + regressor] = kpss(data, regression=regressor)[1]
            elif method == 'pp' and regressor != 'ctt':
                results[method + '_' + regressor] = PhillipsPerron(data, trend=regressor).pvalue

    return results


def get_fstats_in_peak(data: pd.Series, peak: date = None):
    if peak is None:
        peak = data.idxmax()

    group1 = data[:peak]
    group2 = data[peak:]
    variance1 = np.var(group1, ddof=1)
    variance2 = np.var(group2, ddof=1)
    f_value = variance1 / variance2
    df1 = len(group1) - 1
    df2 = len(group2) - 1
    p_value = stats.f.cdf(f_value, df1, df2)

    return data[peak], p_value
