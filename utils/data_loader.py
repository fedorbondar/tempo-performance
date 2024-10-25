import pandas as pd

DB_INDICES_COLUMN_NAME = 'Unnamed: 0'


class DataLoader:
    def __init__(self, path: str, ignore_columns: list = None):
        self.path = path
        self.data: pd.DataFrame

        self.ignore_columns = [DB_INDICES_COLUMN_NAME]
        if ignore_columns is not None:
            self.ignore_columns += ignore_columns

        self.__load_data()

    def __load_data(self):
        data = pd.read_csv(self.path, encoding='utf-8', index_col=0)
        for column in self.ignore_columns:
            if column in data.columns:
                data.drop(columns=self.ignore_columns, inplace=True)
        self.data = data

    def get_data(self):
        return self.data
