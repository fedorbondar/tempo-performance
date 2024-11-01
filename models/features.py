import pandas as pd
from scipy.signal import periodogram, welch


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
