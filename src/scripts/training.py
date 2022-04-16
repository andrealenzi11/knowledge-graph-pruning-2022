import json
import os
from typing import Dict, Optional

from src.config.config import COUNTRIES, FB15K237, WN18RR, YAGO310, CODEXSMALL, NATIONS, \
    create_non_existent_folder, \
    ORIGINAL, NOISE_10, NOISE_20, NOISE_30, \
    RESCAL, TRANSE, DISTMULT, TRANSH, COMPLEX, HOLE, CONVE, ROTATE, PAIRRE, AUTOSF, BOXE
from src.core.pykeen_wrapper import get_train_test_validation, train, store, load
from src.dao.dataset_loading import DatasetPathFactory, TsvDatasetLoader

all_datasets_names = [
    FB15K237,  # "FB15K237"
    WN18RR,  # "WN18RR"
    YAGO310,  # "YAGO310"
    COUNTRIES,  # "COUNTRIES"
    CODEXSMALL,  # "CODEXSMALL"
    NATIONS,  # "NATIONS"
]

valid_kge_models = [
    RESCAL,
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


def get_best_hyper_parameters_diz(current_dataset_name: str, current_model_name: str) -> Optional[Dict[str, float]]:
    if current_dataset_name not in all_datasets_names:
        raise ValueError(f"Invalid dataset name '{current_dataset_name}'!")
    if current_model_name not in valid_kge_models:
        raise ValueError(f"Invalid model name '{current_model_name}'!")
    dataset_tuning_folder_path = DatasetPathFactory(dataset_name=current_dataset_name).get_tuning_folder_path()
    assert dataset_name in dataset_tuning_folder_path
    in_file_path = os.path.join(dataset_tuning_folder_path, f"{current_model_name}_study.json")
    assert current_model_name in in_file_path
    if os.path.isfile(in_file_path):
        with open(in_file_path, 'r') as fr:
            study_diz = json.load(fr)
            best_hyper_params_diz = study_diz["best_params"]
            # BoxE special case (bug)
            if (current_model_name == BOXE) and ("loss.adversarial_temperature" in best_hyper_params_diz):
                del best_hyper_params_diz["loss.adversarial_temperature"]
            return study_diz["best_params"]
    else:
        return None


if __name__ == '__main__':

    # === Set your training configuration === #
    # COUNTRIES, WN18RR, FB15K237, YAGO310, CODEXSMALL, NATIONS
    dataset_name: str = COUNTRIES
    force_training: bool = True
    # num_epochs = 200  # default: 5
    # batch_size = 256  # default: 256
    stopper = None  # "early" | None

    all_datasets_names = {COUNTRIES, WN18RR, FB15K237, YAGO310, CODEXSMALL, NATIONS}
    print(f"all_datasets_names: {all_datasets_names}")

    dataset_models_folder_path = DatasetPathFactory(dataset_name=dataset_name).get_models_folder_path()

    print(f"\n{'*' * 80}")
    print("TRAINING CONFIGURATION")
    print(f"\t\t dataset_name: {dataset_name}")
    print(f"\t\t dataset_models_folder_path: {dataset_models_folder_path}")
    print(f"\t\t force_training: {force_training}")
    # print(f"\t\t num_epochs: {num_epochs}")
    # print(f"\t\t batch_size {batch_size}")
    print(f"\t\t stopper: {stopper}")
    print(f"{'*' * 80}\n\n")

    for noise_level in [
        ORIGINAL,
        # NOISE_5,
        NOISE_10,
        # NOISE_15,
        NOISE_20,
        NOISE_30,
    ]:
        print(f"\n\n#################### {noise_level} ####################\n")
        datasets_loader = TsvDatasetLoader(dataset_name=dataset_name, noise_level=noise_level)
        training_path, validation_path, testing_path = \
            datasets_loader.get_training_validation_testing_dfs_paths(noisy_test_flag=False)

        training, testing, validation = get_train_test_validation(training_set_path=training_path,
                                                                  test_set_path=testing_path,
                                                                  validation_set_path=validation_path,
                                                                  create_inverse_triples=False)
        print("\t (*) training:")
        print(f"\t\t\t path={training_path}")
        print(f"\t\t\t #triples={training.num_triples}  | "
              f" #entities={training.num_entities}  | "
              f" #relations={training.num_relations} \n")
        print("\t (*) validation:")
        print(f"\t\t\t path={validation_path}")
        print(f"\t\t\t #triples={validation.num_triples}  | "
              f" #entities={validation.num_entities}  | "
              f" #relations={validation.num_relations} \n")
        print("\t (*) testing:")
        print(f"\t\t\t path={testing_path}")
        print(f"\t\t\t #triples={testing.num_triples}  | "
              f" #entities={testing.num_entities}  | "
              f" #relations={testing.num_relations} \n")

        print("\n\n>>> Start KGE models training...\n")

        for model_name, year in [
            # (RESCAL, 2011),
            # (TRANSE, 2013),
            # (DISTMULT, 2014),
            # (TRANSH, 2014),
            # (COMPLEX, 2016),
            # (HOLE, 2016),
            # (CONVE, 2018),
            # (ROTATE, 2019),
            # (PAIRRE, 2020),
            # (AUTOSF, 2020),
            (BOXE, 2020),
        ]:
            print(f"\n>>>>>>>>>>>>>>>>>>>> {model_name} ({year}) <<<<<<<<<<<<<<<<<<<<")

            out_folder_path = os.path.join(dataset_models_folder_path, noise_level, model_name)
            print(f"\n\t - out_folder_path: {out_folder_path}")
            create_non_existent_folder(folder_path=out_folder_path)

            # Try to Load an already trained KGE model from File System
            if not force_training:
                print(f"\t - Try to loading already trained '{model_name}' model...")
                try:
                    pipeline_result = load(in_dir_path=out_folder_path)
                    print(f"\t <> '{model_name}' model loaded from File System!")
                except FileNotFoundError:
                    print(f"\t <> '{model_name}' model NOT already present in the File System!")
                    force_training = True

            # Train a new KGE model and store it on File System
            if force_training:

                print("\t - Load best Hyper-parameters")
                kwargs_diz = {
                    "model": None,
                    "training": None,
                    "loss": None,
                    "regularizer": None,
                    "optimizer": None,
                    "negative_sampler": None,
                }
                try:
                    # hyperparams_diz = HYPERPARAMS_CONFIG[dataset_name][model_name]
                    hyperparams_diz = get_best_hyper_parameters_diz(current_dataset_name=dataset_name,
                                                                    current_model_name=model_name)
                    for k, v in hyperparams_diz.items():
                        arr_tmp = k.split(".")
                        assert len(arr_tmp) == 2
                        component, hp = arr_tmp[0].strip(), arr_tmp[1].strip()
                        if kwargs_diz[component] is None:
                            kwargs_diz[component] = {hp: v}
                        else:
                            kwargs_diz[component][hp] = v
                except KeyError as k_err:
                    print(k_err)
                print(f"\t - kwargs_diz: {kwargs_diz}")

                print(f"\t - Training '{model_name}' model...")
                pipeline_result = train(
                    training=training,
                    testing=testing,
                    validation=validation,
                    model_name=model_name,
                    model_kwargs=kwargs_diz["model"],
                    training_kwargs=kwargs_diz["training"],
                    loss_kwargs=kwargs_diz["loss"],
                    regularizer_kwargs=kwargs_diz["regularizer"],
                    optimizer_kwargs=kwargs_diz["optimizer"],
                    negative_sampler_kwargs=kwargs_diz["negative_sampler"],
                )

                print(f"- Saving '{model_name}' model......")
                store(result_model=pipeline_result,
                      out_dir_path=out_folder_path)

            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")

        print(f"\n###########################################################\n")
