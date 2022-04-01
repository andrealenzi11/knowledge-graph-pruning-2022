from config import COUNTRIES, FB15K237, WN18RR, YAGO310, CODEXSMALL, \
    COUNTRIES_MODELS_FOLDER_PATH, FB15K237_MODELS_FOLDER_PATH, WN18RR_MODELS_FOLDER_PATH, \
    YAGO310_MODELS_FOLDER_PATH, CODEXSMALL_MODELS_FOLDER_PATH


class DatasetModelsFolderPathFactory:

    def __init__(self):
        self.valid_datasets_names = {
            COUNTRIES,
            FB15K237,
            WN18RR,
            YAGO310,
            CODEXSMALL,
        }

    def get(self, dataset_name: str) -> str:
        dataset_name = str(dataset_name).upper().strip()
        if dataset_name == COUNTRIES:
            return COUNTRIES_MODELS_FOLDER_PATH
        elif dataset_name == FB15K237:
            return FB15K237_MODELS_FOLDER_PATH
        elif dataset_name == WN18RR:
            return WN18RR_MODELS_FOLDER_PATH
        elif dataset_name == YAGO310:
            return YAGO310_MODELS_FOLDER_PATH
        elif dataset_name == CODEXSMALL:
            return CODEXSMALL_MODELS_FOLDER_PATH
        else:
            raise ValueError(f"Invalid dataset name: '{str(dataset_name)}'! \n"
                             f"\t\t Specify one of the following values: \n"
                             f"\t\t {self.valid_datasets_names} \n")
