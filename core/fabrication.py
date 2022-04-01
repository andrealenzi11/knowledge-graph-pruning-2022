from config.config import COUNTRIES, FB15K237, WN18RR, YAGO310, CODEXSMALL, \
    COUNTRIES_MODELS_FOLDER_PATH, FB15K237_MODELS_FOLDER_PATH, WN18RR_MODELS_FOLDER_PATH, \
    YAGO310_MODELS_FOLDER_PATH, CODEXSMALL_MODELS_FOLDER_PATH, COUNTRIES_TUNING_FOLDER_PATH, \
    FB15K237_TUNING_FOLDER_PATH, WN18RR_TUNING_FOLDER_PATH, YAGO310_TUNING_FOLDER_PATH, CODEXSMALL_TUNING_FOLDER_PATH


class DatasetPathFactory:

    def __init__(self, dataset_name: str):
        self.valid_datasets_names = {
            COUNTRIES,
            FB15K237,
            WN18RR,
            YAGO310,
            CODEXSMALL,
        }
        self.dataset_name = str(dataset_name).upper().strip()
        if self.dataset_name not in self.valid_datasets_names:
            raise ValueError(f"Invalid dataset name: '{str(dataset_name)}'! \n"
                             f"\t\t Specify one of the following values: \n"
                             f"\t\t {self.valid_datasets_names} \n")

    def get_models_folder_path(self) -> str:
        if self.dataset_name == COUNTRIES:
            return COUNTRIES_MODELS_FOLDER_PATH
        elif self.dataset_name == FB15K237:
            return FB15K237_MODELS_FOLDER_PATH
        elif self.dataset_name == WN18RR:
            return WN18RR_MODELS_FOLDER_PATH
        elif self.dataset_name == YAGO310:
            return YAGO310_MODELS_FOLDER_PATH
        elif self.dataset_name == CODEXSMALL:
            return CODEXSMALL_MODELS_FOLDER_PATH
        else:
            raise ValueError(f"Invalid dataset name!")

    def get_tuning_folder_path(self) -> str:
        if self.dataset_name == COUNTRIES:
            return COUNTRIES_TUNING_FOLDER_PATH
        elif self.dataset_name == FB15K237:
            return FB15K237_TUNING_FOLDER_PATH
        elif self.dataset_name == WN18RR:
            return WN18RR_TUNING_FOLDER_PATH
        elif self.dataset_name == YAGO310:
            return YAGO310_TUNING_FOLDER_PATH
        elif self.dataset_name == CODEXSMALL:
            return CODEXSMALL_TUNING_FOLDER_PATH
        else:
            raise ValueError(f"Invalid dataset name!")

