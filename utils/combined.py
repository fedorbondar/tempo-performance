import json
from utils.data_loader import DataLoader
from utils.data_masking import DataMasking


def load_data_and_save_masked(path_to_data: str, path_to_output_folder: str):
    dl = DataLoader(path_to_data)
    dm = DataMasking(dl.get_data())
    json.dump(dm.get_masks(), open(path_to_output_folder + "/masks.json", 'w'))
    json.dump(dm.get_unmasks(), open(path_to_output_folder + "/unmasks.json", 'w'))
    dm.get_masked_data().to_csv(path_to_output_folder + '/data_masked.csv')