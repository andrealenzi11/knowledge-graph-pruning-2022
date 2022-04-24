import json
import os
from typing import Dict, Optional

from src.config.config import COUNTRIES, FB15K237, WN18RR, YAGO310, CODEXSMALL, NATIONS, \
    TRANSE, DISTMULT, TRANSH, COMPLEX, HOLE, CONVE, ROTATE, PAIRRE, AUTOSF, BOXE
from src.dao.dataset_loading import DatasetPathFactory

all_datasets_names = [
    FB15K237,  # "FB15K237"
    WN18RR,  # "WN18RR"
    YAGO310,  # "YAGO310"
    COUNTRIES,  # "COUNTRIES"
    CODEXSMALL,  # "CODEXSMALL"
    NATIONS,  # "NATIONS"
]

valid_kge_models = [
    # RESCAL,
    TRANSE,
    DISTMULT,
    TRANSH,
    COMPLEX,
    HOLE,
    CONVE,
    ROTATE,
    PAIRRE,
    AUTOSF,
    BOXE,
]


def get_best_hyper_parameters_diz(current_dataset_name: str,
                                  current_model_name: str) -> Optional[Dict[str, float]]:
    """
        It tries to read from the file system the configuration with the best hyper-parameters
        for the specified model and dataset:
            - if therese is the json file: it returns the best configuration as a dict;
            - Otherwise: it returns None value;
    """
    if current_dataset_name not in all_datasets_names:
        raise ValueError(f"Invalid dataset name '{current_dataset_name}'!")
    if current_model_name not in valid_kge_models:
        raise ValueError(f"Invalid model name '{current_model_name}'!")
    dataset_tuning_folder_path = DatasetPathFactory(dataset_name=current_dataset_name).get_tuning_folder_path()
    assert current_dataset_name in dataset_tuning_folder_path
    in_file_path = os.path.join(dataset_tuning_folder_path, f"{current_model_name}_study.json")
    assert current_model_name in in_file_path
    if os.path.isfile(in_file_path):
        with open(in_file_path, 'r') as fr:
            study_diz: dict = json.load(fr)
            best_hyper_params_diz: dict = study_diz["best_params"]
            assert isinstance(best_hyper_params_diz, dict)
            # BoxE special case (bug)
            if (current_model_name == BOXE) and ("loss.adversarial_temperature" in best_hyper_params_diz):
                del best_hyper_params_diz["loss.adversarial_temperature"]
            return best_hyper_params_diz
    else:
        return None


def parse_best_hyper_parameters_diz(best_hyper_params_diz: Optional[Dict[str, float]]) -> Dict[str, Optional[dict]]:
    """
        Parse the best hyper-parameters dict and return it in the right format:

        Example:
            from ===>
                    {
                        'model.embedding_dim': 240,
                        'model.p': 2,
                        'loss.margin': 30,
                        'loss.adversarial_temperature': 0.8865,
                        'optimizer.lr': 0.00092,
                        'negative_sampler.num_negs_per_pos': 54,
                        'training.num_epochs': 150,
                        'training.batch_size': 128
                    }
            to ===>
                    {
                        "model": {'embedding_dim': 240, 'p': 2},
                        "training": {'num_epochs': 150, 'batch_size': 128},
                        "loss": {'margin': 30, 'adversarial_temperature': 0.8865},
                        "regularizer": None,
                        "optimizer": {'lr': 0.00092},
                        "negative_sampler": {'num_negs_per_pos': 54},
                    }
    """
    result = {
        "model": None,
        "training": None,
        "loss": None,
        "regularizer": None,
        "optimizer": None,
        "negative_sampler": None,
    }
    if best_hyper_params_diz is None:
        return result
    else:
        for h_name, h_value in best_hyper_params_diz.items():
            arr_tmp = h_name.split(".")
            assert len(arr_tmp) == 2
            component, hp = arr_tmp[0].strip(), arr_tmp[1].strip()
            assert component in result.keys()
            # case 1: initialize a new dictionary for the component
            if result[component] is None:
                result[component] = {hp: h_value}
            # case 2: append a new key-value pair to an existent component dictionary
            elif isinstance(result[component], dict):
                result[component][hp] = h_value
            # error: raise an exception
            else:
                raise TypeError(f"Invalid type for key '{component}' in result diz!")
        return result
