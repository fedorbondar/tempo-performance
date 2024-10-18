import pandas as pd
from datetime import datetime

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
