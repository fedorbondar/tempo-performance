import unittest
import numpy as np

from utils.data_builder import DataBuilder
from utils.data_loader import DataLoader

from metrics.tempo_based import compute_absent_rate
from metrics.tempo_based import compute_support_tasks_rate
from metrics.tempo_based import compute_initiative_completion_rate
from metrics.tempo_based import compute_initiative_share_by_domain


class TestMetrics(unittest.TestCase):
    def setUp(self):
        dl = DataLoader('../data_sample/tempo_db_masked_sample.csv')
        db = DataBuilder(dl.get_data())
        self.date_from = '2024-10-01'
        self.date_until = '2024-10-14'
        self.data = db.get_employee_worklog_in_period('author0', self.date_from, self.date_until)
        self.data_domain = db.get_domain_worklog_in_period('domain0',  self.date_from, self.date_until)

    def test_icr_computation(self):
        metric = compute_initiative_completion_rate(self.data, self.date_from, self.date_until)
        self.assertEqual(metric, np.float64(0.07499999999999996))

    def test_suptr_computation(self):
        metric = compute_support_tasks_rate(self.data, self.date_from, self.date_until)
        self.assertEqual(metric, 1.0)

    def test_ar_computation(self):
        metric = compute_absent_rate(self.data, self.date_from, self.date_until)
        self.assertEqual(metric, 1.0)

    def test_isd_computation(self):
        metric = compute_initiative_share_by_domain(self.data, self.data_domain)
        self.assertEqual(metric, 1.0)


if __name__ == '__main__':
    unittest.main()
