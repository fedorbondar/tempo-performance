from features import *
from metrics.tempo_based import *
from utils.data_builder import DataBuilder


def create_dataset_in_period(db: DataBuilder, date_from: str, date_until: str, ignore_weekends: bool = False,
                             n_periods: int = 3, strategy: str = 'even', add_single_metrics: bool = False):
    domains = db.data['domain'].unique()
    domains_data = [db.get_domain_worklog_in_period(domain, date_from, date_until) for domain in domains]

    result = dict()
    index = []

    for domain_data in domains_data:
        for author in domain_data['author'].unique():
            index.append(f'{author}_{date_from}_{date_until}')

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

            if add_single_metrics:
                single_metrics = {
                    'icr': compute_initiative_completion_rate(author_worklog, date_from, date_until),
                    'suptr': compute_support_tasks_rate(author_worklog, date_from, date_until),
                    'ar': compute_absent_rate(author_worklog, date_from, date_until),
                    'isd': compute_initiative_share_by_domain(author_worklog, domain_data),
                }
                target = single_metrics | target

            full_features = (periods | stationary_tests | structural_shift | static_features | week_daily_means |
                             co_integration | target)

            if not result:
                for key, value in full_features.items():
                    result[key] = [value]
            else:
                for key, value in full_features.items():
                    result[key].append(value)

        if not domain_data.empty:
            print(f'{domain_data["domain"].unique()[0]} proceeded for dates {date_from} - {date_until}')

    result['author'] = index

    return pd.DataFrame.from_dict(result).set_index('author')


def create_dataset(db: DataBuilder, dates: list[tuple[str, str]], ignore_weekends: bool = False,
                   n_periods: int = 3, strategy: str = 'even', add_single_metrics: bool = False):
    result = None
    for date_from, date_until in dates:
        dataset = create_dataset_in_period(db, date_from, date_until, ignore_weekends, n_periods, strategy,
                                           add_single_metrics)
        if result is None:
            result = dataset
        else:
            result = pd.concat([result, dataset])

    return result
