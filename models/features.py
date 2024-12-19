from datetime import date

import numpy as np
import pandas as pd

from scipy.signal import periodogram, welch
import scipy.stats as stats

from arch.unitroot import PhillipsPerron
from arch.utility.exceptions import InfeasibleTestException
from statsmodels.tools.sm_exceptions import InterpolationWarning

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import warnings
warnings.filterwarnings("ignore", category=InterpolationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


DEFAULT_CO_INTEGRATION_RESULT = {
    'daily0': 1,
    'daily1': 1,
    'daily2': 1,
    'weekly0': 1,
    'weekly1': 1,
    'weekly2': 1,
}


def compute_acf(data: pd.Series):
    """
    Manual computation of Auto-Correlation Function.
    :param data: time series with physical `author`'s worklog.
    :return: values of Auto-Correlation Function and number of lags used.
    """
    n_lags = len(data) // 2 + 1
    values = pd.DataFrame(data.values)
    dataframe = pd.concat([values.shift(i) for i in range(n_lags)], axis=1)
    dataframe.columns = ['t'] + ['t+' + str(i) for i in range(1, n_lags)]
    return dataframe.corr()['t'].values, n_lags


def get_k_periods(data: pd.Series, k: int, method: str = 'periodogram'):
    """
    Compute least k periods based on periodogram of given time series.
    :param data: time series with physical `author`'s worklog.
    :param k: number of periods.
    :param method: either `periodogram` or `welch` (different methods to estimate power spectral density).
    :return: list of k lags that are considered as values of periods.
    """
    if method == 'periodogram':
        _, y = periodogram(data.values)
    else:
        _, y = welch(data.values)
    values_and_lags = [(value, lag) for value, lag in zip(y, range(len(y)))]
    values_and_lags.sort(reverse=True)
    return [lag for _, lag in values_and_lags[:k]]


def get_stationary_tests_results(data: pd.Series, methods: list[str] = None, regression: list[str] = None,
                                 significance_level: float = 0.05):
    """
    Perform statistic tests to find given time series stationary or not.
    :param data: time series with physical `author`'s worklog.
    :param methods: tests to perform: Augmented Dickey-Fuller (`adf`), Kwiatkowski-Phillips-Schmidt-Shin (`kpss`) or Phillips-Perron (`pp`).
    :param regression: types of regression used: constant (`c`), constant & trend (`ct`) or parabolic trend (`ctt`).
    :param significance_level: level of significance to reject null hypothesis.
    :return: dictionary with keys `<method>_<regressor>` and values as results of corresponding tests.
    """
    if methods is None:
        methods = ['adf']
    if regression is None:
        regression = ['c']

    is_constant = len(data.unique()) == 1

    results = dict()

    for method in methods:
        for regressor in regression:
            if method == 'adf':
                results[method + '_' + regressor] = (
                    is_constant or adfuller(data, regression=regressor)[1] > significance_level
                )
            elif (method == 'kpss' or method == 'pp') and regressor == 'ctt':
                continue
            elif method == 'kpss':
                try:
                    results[method + '_' + regressor] = (
                        is_constant or kpss(data, regression=regressor)[1] < significance_level
                    )
                except OverflowError:
                    results[method + '_' + regressor] = True
            elif method == 'pp':
                try:
                    results[method + '_' + regressor] = (
                        is_constant or PhillipsPerron(data, trend=regressor).pvalue > significance_level
                    )
                except InfeasibleTestException:
                    results[method + '_' + regressor] = True
                except ValueError:
                    results[method + '_' + regressor] = True
            results[method + '_' + regressor] = 1 if results[method + '_' + regressor] else 0

    return results


def get_fstats_in_peak(data: pd.Series, peak: date = None, significance_level: float = 0.05):
    """
    Perform Chow test for structural break.
    :param data: time series with physical `author`'s worklog.
    :param peak: supposed point of structural break in data (if set `None`, argmax in `data` is used).
    :param significance_level: level of significance to reject null hypothesis.
    :return: value of `data` in `peak` and either True if there is no structural break in `data` in `peak` or False.
    """
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

    return data[peak], p_value < significance_level


def get_mean_var(data: pd.Series):
    """
    Get mean and variance of data.
    :param data: time series with physical `author`'s worklog.
    :return: mean and variance values.
    """
    return np.mean(data), np.var(data)


def get_week_daily_means(data: pd.Series, ignore_weekends: bool = False):
    """
    Compute mean value of logged time for each day of week in `data`.
    :param data: time series with physical `author`'s worklog.
    :param ignore_weekends: whether to ignore logged time during weekends or not.
    :return: dict with keys 0-4 or 0-6 corresponding to days of week consequently and mean values of logged time.
    """
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


def get_co_integration(data: pd.Series, patterns: list[str] = None, ignore_weekends: bool = False):
    """
    Perform Johansen co-integration test.
    :param data: time series with physical `author`'s worklog.
    :param patterns: either `daily` (`author` logs work every work day) or `weekly` (`author` logs work once a week).
    :param ignore_weekends: whether to ignore logged time during weekends or not.
    :return: dict with following keys: `<pattern>0` indicates if there is no co-integration relation, `<pattern>1` if there is one, `<pattern>2` if there is more than one.
    """
    if patterns is None:
        patterns = ['daily']

    start_day_of_week = data.index[0].weekday()

    result = dict()

    for pattern in patterns:
        if pattern == 'daily':
            daily_pattern = [8.0] * 5 if ignore_weekends else [8.0] * 5 + [0.0] * 2
            daily_pattern *= 2 * len(data) // len(daily_pattern)
            daily_pattern = daily_pattern[start_day_of_week: start_day_of_week + len(data)]
            # avoid absolute constant pattern
            daily_pattern[-1] = 0.0
            compare_series = pd.Series(daily_pattern, index=data.index)
        if pattern == 'weekly':
            weekly_pattern = [0.0] * 4 + [40.0] if ignore_weekends else [0.0] * 4 + [40.0] + [0.0] * 2
            weekly_pattern *= 2 * len(data) // len(weekly_pattern)
            weekly_pattern = weekly_pattern[start_day_of_week: start_day_of_week + len(data)]
            compare_series = pd.Series(weekly_pattern, index=data.index)

        try:
            johansen = coint_johansen(pd.DataFrame([data, compare_series]).T, 0, 1)
            traces = johansen.lr1
            critical_values = johansen.cvt
            result_list = [1 if traces[0] > critical_value else 0 for critical_value in critical_values[0]]
            for i, value in enumerate(result_list):
                result[pattern + str(i)] = value
        except np.linalg.LinAlgError:
            return DEFAULT_CO_INTEGRATION_RESULT

    return result
