from features import *
from metrics.tempo_based import compute_weighted_target
from utils.data_builder import DataBuilder


def create_dataset(db: DataBuilder, date_from: str, date_until: str, ignore_weekends: bool = False,
                   n_periods: int = 3, strategy: str = 'even'):
    domains = db.data['domain'].unique()
    domains_data = [db.get_domain_worklog_in_period(domain, date_from, date_until) for domain in domains]

    result = dict()

    for domain_data in domains_data:
        for author in domain_data['author'].unique():

            # features
            author_time_series = db.create_series_logged_time(author, date_from, date_until, ignore_weekends)
            periods = dict(zip(
                ['period' + str(i + 1) for i in range(n_periods)],
                get_k_periods(author_time_series, n_periods)
            ))
            stationary_tests = get_stationary_tests_results(author_time_series, ['adf', 'pp', 'kpss'],
                                                            ['c', 'ct', 'ctt'])
            maximum, p_value = get_fstats_in_peak(author_time_series)
            structural_shift = {'max': maximum, 'shift': p_value}
            mean, var = get_mean_var(author_time_series)
            static_features = {'mean': mean, 'var': var}
            week_daily_means = get_week_daily_means(author_time_series, ignore_weekends)
            co_integration = get_co_integration(author_time_series, ['daily', 'weekly'], ignore_weekends)

            # target
            author_worklog = db.get_employee_worklog_in_period(author, date_from, date_until)
            target = {
                'target': compute_weighted_target(author_worklog, domain_data, date_from, date_until, strategy)
            }

            full_features = (periods | stationary_tests | structural_shift | static_features | week_daily_means |
                             co_integration | target)

            if not result:
                for key, value in full_features.items():
                    result[key] = [value]
            else:
                for key, value in full_features.items():
                    result[key].append(value)

        if not domain_data.empty:
            print(f'{domain_data["domain"].unique()[0]} proceeded')

    return pd.DataFrame.from_dict(result)
