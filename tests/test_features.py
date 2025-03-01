import unittest

from utils.data_builder import DataBuilder
from utils.data_loader import DataLoader

from models.features import get_mean_var
from models.features import get_k_periods
from models.features import get_co_integration
from models.features import get_fstats_in_peak
from models.features import get_week_daily_means
from models.features import get_stationary_tests_results


class TestMetrics(unittest.TestCase):
    def setUp(self):
        dl = DataLoader('../data_sample/tempo_db_masked_sample.csv')
        db = DataBuilder(dl.get_data())
        self.date_from = '2024-10-01'
        self.date_until = '2024-10-14'
        self.series = db.create_series_logged_time('author0', self.date_from, self.date_until,
                                                   ignore_weekends=True)

    def test_get_k_periods(self):
        values = get_k_periods(self.series, 3)
        self.assertEqual(values, [3, 2, 5])

    def test_get_stationary_tests_results(self):
        correct_result = {'adf_c': 0, 'adf_ct': 0, 'adf_ctt': 1,
                          'kpss_c': 0, 'kpss_ct': 0, 'pp_c': 0, 'pp_ct': 0}
        values = get_stationary_tests_results(self.series, ['adf', 'pp', 'kpss'], ['c', 'ct', 'ctt'])
        self.assertEqual(values, correct_result)

    def test_get_fstats_in_peak(self):
        values = get_fstats_in_peak(self.series)
        self.assertEqual(values, (24.0, False))

    def test_get_mean_var(self):
        values = get_mean_var(self.series)
        self.assertEqual(values, (8.0, 76.8))

    def test_get_week_daily_means(self):
        correct_result = {0: 8.0,
                          1: 8.0,
                          2: 0.0,
                          3: 0.0,
                          4: 24.0}
        values = get_week_daily_means(self.series, ignore_weekends=True)
        self.assertEqual(values, correct_result)

    def test_get_co_integration(self):
        correct_result = {'daily0': 0, 'daily1': 0, 'daily2': 0,
                          'weekly0': 1, 'weekly1': 1, 'weekly2': 0}
        values = get_co_integration(self.series, patterns=['daily', 'weekly'], ignore_weekends=True)
        self.assertEqual(values, correct_result)
