import pandas as pd

DEFAULT_COLUMNS_TO_MASK = ['issuekey', 'author', 'domain']


class DataMasking:
    """
    Class for applying masks on vulnerable corporate data. Only for data visibility purpose.
    """
    def __init__(self, data: pd.DataFrame, columns: list = None):
        self.original_data: pd.DataFrame = data
        self.masked_data = self.original_data.copy()
        self.masks = {}
        self.unmasks = {}

        self.columns: list[str] = DEFAULT_COLUMNS_TO_MASK
        if columns is not None:
            self.columns = columns

        for column_name in self.columns:
            self.__create_masking(column_name)

    def __create_masking(self, column_name: str):
        """
        Creates masking by replacing column values with impersonal values.
        Saves masked data as well as masking and unmasking dictionaries.

        Example: column `author` values will be replaced with `author0, author1, author2` etc.
        :param column_name: column name to apply masking on.
        """
        unique_column_values = self.original_data[column_name].unique()
        masks = [column_name + str(idx) for idx in range(len(unique_column_values))]

        masking_map = {}
        unmasking_map = {}
        for idx in range(len(unique_column_values)):
            masking_map[unique_column_values[idx]] = masks[idx]
            unmasking_map[masks[idx]] = unique_column_values[idx]

        self.masks[column_name] = masking_map
        self.unmasks[column_name] = unmasking_map

        # pd.Series.map is faster than pd.DataFrame.replace in this case
        self.masked_data[column_name] = self.masked_data[column_name].map(masking_map.get)

    def get_masked_data(self):
        return self.masked_data

    def get_masks(self):
        return self.masks

    def get_unmasks(self):
        return self.unmasks
