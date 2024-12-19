import pandas as pd

DB_INDICES_COLUMN_NAME = 'Unnamed: 0'


class DataLoader:
    """
    Class for reading Jira tempo csv data. Can ignore specified columns optionally.

    Assumes csv data to contain the following columns:

    * `issuekey` -- issue's short name in Jira.
    * `date` -- date on which the work was logged on.
    * `hour` -- amount of work hours logged on issue.
    * `author` -- login of worker who logged time.
    * `comment` -- comment left on logged time ("Worked on issue ..." by default).
    * `updated` -- datetime of logging event.
    * `issue_type` -- type of issue on which time was logged.
    * `issue_summary` -- summary of issue on which time was logged.
    * `domain` -- team which `author` belongs to.
    """
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
