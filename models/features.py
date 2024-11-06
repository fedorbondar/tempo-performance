from datetime import date
import numpy as np
import pandas as pd

from scipy.signal import periodogram, welch
import scipy.stats as stats
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen


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


def get_mean_var(data: pd.Series):
    return np.mean(data), np.var(data)


def get_week_daily_means(data: pd.Series, ignore_weekends: bool = False):
    if ignore_weekends:
        result = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    else:
        result = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

    for key in result.keys():
        sum_values = 0
        n_values = 0
        for idx in data.index:
            if idx.weekday() == key:
                sum_values += data.iloc[key]
                n_values += 1
        if n_values == 0:
            result[key] = 0
        else:
            result[key] = sum_values / n_values

    return result


def get_co_integration(data: pd.Series, patterns: list[str] = None):
    if patterns is None:
        patterns = ['daily']

    result = dict()

    for pattern in patterns:
        if pattern == 'daily':
            daily_series = pd.Series([8.0] * (len(data) - 1) + [0.0], index=data.index)
            johansen = coint_johansen(pd.DataFrame([data, daily_series]).T, 0, 1)
            traces = johansen.lr1
            critical_values = johansen.cvt
            result['daily'] = [1 if traces[0] > critical_value else 0 for critical_value in critical_values[0]]

    return result
