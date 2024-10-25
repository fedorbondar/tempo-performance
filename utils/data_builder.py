import pandas as pd
from datetime import datetime, timedelta

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
START_DAY_TIME = ' 00:00:00'
END_DAY_TIME = ' 23:59:59'


class DataBuilder:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.__preprocess_date_time_cols()

    def __preprocess_date_time_cols(self):
        self.data['updated'] = pd.to_datetime(self.data['updated'], format=DATETIME_FORMAT)

    def get_employee_worklog_in_period(self, author: str, date_from: str, date_until: str):
        datetime_from = datetime.strptime(date_from + START_DAY_TIME, DATETIME_FORMAT)
        datetime_until = datetime.strptime(date_until + END_DAY_TIME, DATETIME_FORMAT)

        result_data = self.data.loc[self.data['author'] == author]
        result_data = result_data.loc[self.data['updated'] >= datetime_from].loc[self.data['updated'] <= datetime_until]

        return result_data

    def create_series_logged_time(self, author: str, date_from: str, date_until: str):
        datetime_from = datetime.strptime(date_from + START_DAY_TIME, DATETIME_FORMAT).date()
        datetime_until = datetime.strptime(date_until + END_DAY_TIME, DATETIME_FORMAT).date()
        delta = datetime_until - datetime_from

        filtered_data = self.get_employee_worklog_in_period(author, date_from, date_until)

        days_range = [datetime_from + timedelta(days=x) for x in range(delta.days + 1)]
        worklog_dict = dict(zip(days_range, [0] * len(days_range)))

        for hour, updated in zip(filtered_data['hour'], filtered_data['updated']):
            worklog_dict[updated.date()] += hour

        return pd.Series(worklog_dict.values(), index=worklog_dict.keys())

