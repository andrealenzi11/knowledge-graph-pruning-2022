import configparser
import json
import os

import numpy as np
import pandas as pd
import torch
from pykeen.evaluation import RankBasedEvaluator

from src.config.config import COUNTRIES, FB15K237, WN18RR, YAGO310, CODEXSMALL, NATIONS, \
    MR, MRR, HITS_AT_1, HITS_AT_3, HITS_AT_5, HITS_AT_10, \
    BOTH_STRATEGY, HEAD_STRATEGY, TAIL_STRATEGY, \
    REALISTIC_STRATEGY, OPTIMISTIC_STRATEGY, PESSIMISTIC_STRATEGY, \
    ORIGINAL, NOISE_10, NOISE_20, NOISE_30, RESCAL, CONVE, \
    FB15K237_RESULTS_FOLDER_PATH, WN18RR_RESULTS_FOLDER_PATH, YAGO310_RESULTS_FOLDER_PATH, \
    COUNTRIES_RESULTS_FOLDER_PATH, CODEXSMALL_RESULTS_FOLDER_PATH, NATIONS_RESULTS_FOLDER_PATH, TOTAL_RANDOM
from src.core.pykeen_wrapper import get_train_test_validation, print_partitions_info
from src.dao.dataset_loading import DatasetPathFactory, TsvDatasetLoader

# set pandas visualization options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

datasets_names_results_folder_map = {
    COUNTRIES: COUNTRIES_RESULTS_FOLDER_PATH,
    WN18RR: WN18RR_RESULTS_FOLDER_PATH,
    FB15K237: FB15K237_RESULTS_FOLDER_PATH,
    YAGO310: YAGO310_RESULTS_FOLDER_PATH,
    CODEXSMALL: CODEXSMALL_RESULTS_FOLDER_PATH,
    NATIONS: NATIONS_RESULTS_FOLDER_PATH
}
for k, v in datasets_names_results_folder_map.items():
    print(f"datasets_name={k} | dataset_results_folder={v}")

all_noise_levels = {ORIGINAL, TOTAL_RANDOM, NOISE_10, NOISE_20, NOISE_30}
print(f"all_noise_levels: {all_noise_levels}")

all_metrics = {MR, MRR, HITS_AT_1, HITS_AT_3, HITS_AT_5, HITS_AT_10}
print(f"all_metrics: {all_metrics}")

all_strategies_1 = {BOTH_STRATEGY, HEAD_STRATEGY, TAIL_STRATEGY}
print(f"all_strategies_1: {all_strategies_1}")

all_strategies_2 = {REALISTIC_STRATEGY, OPTIMISTIC_STRATEGY, PESSIMISTIC_STRATEGY}
print(f"all_strategies_2: {all_strategies_2}")

if __name__ == '__main__':

    # ===== Link Prediction Evaluation - Configuration ===== #
    config = configparser.ConfigParser()
    if not os.path.isfile('dataset_local.ini'):
        raise FileNotFoundError("Create your 'dataset_local.ini' file in the 'src.scripts.evaluation' package "
                                "starting from the 'dataset.ini' template!")
    config.read('dataset_local.ini')
    dataset_name = config['dataset_info']['dataset_name']
    load_precomputed_results_flag = True
    force_inference = True
    force_saving = True
    strategy1: str = BOTH_STRATEGY  # "both" | "head" | "tail"
    strategy2: str = REALISTIC_STRATEGY  # "realistic" | "optimistic" | "pessimistic"
    device = "cpu"  # torch.device("cuda:0")
    selected_metrics = {
        MR,
        MRR,
        HITS_AT_1,
        HITS_AT_3,
        HITS_AT_5,
        HITS_AT_10,
    }
    # ====================================================== #

    if dataset_name not in set(datasets_names_results_folder_map.keys()):
        raise ValueError(f"Invalid dataset name '{dataset_name}'!")

    if strategy1 not in {"both", "head", "tail"}:
        raise ValueError(f"Invalid Strategy1 '{strategy1}'!")

    if strategy2 not in {"realistic", "optimistic", "pessimistic"}:
        raise ValueError(f"Invalid Strategy2 '{strategy2}'!")

    dataset_results_folder_path = datasets_names_results_folder_map[dataset_name]
    assert dataset_name in dataset_results_folder_path

    dataset_models_folder_path = DatasetPathFactory(dataset_name=dataset_name).get_models_folder_path()
    assert dataset_name in dataset_models_folder_path

    print("\n> Link Prediction Evaluation - Configuration")
    print(f"{'*' * 80}")
    print(f"\t\t dataset_name: {dataset_name}")
    print(f"\t\t dataset_models_folder_path: {dataset_models_folder_path}")
    print(f"\t\t dataset_results_folder_path: {dataset_results_folder_path}")
    print(f"\t\t load_precomputed_results_flag: {load_precomputed_results_flag}")
    print(f"\t\t force_inference: {force_inference}")
    print(f"\t\t force_saving: {force_saving}")
    print(f"\t\t strategy1: {strategy1}")
    print(f"\t\t strategy2: {strategy2}")
    print(f"\t\t selected_metrics: {selected_metrics}")
    print(f"\t\t device: {device}")
    print(f"{'*' * 80}\n")

    # ========== original dataset ========== #
    print("\n Loading original dataset...")
    datasets_loader_original = TsvDatasetLoader(dataset_name=dataset_name, noise_level=ORIGINAL)
    # paths
    training_original_path, validation_original_path, testing_original_path = \
        datasets_loader_original.get_training_validation_testing_dfs_paths(noisy_test_flag=False)
    assert "training" in training_original_path
    assert ORIGINAL in training_original_path
    assert "validation" in validation_original_path
    assert ORIGINAL in validation_original_path
    assert "testing" in testing_original_path
    assert ORIGINAL in testing_original_path
    # triples factories
    training_original, testing_original, validation_original = \
        get_train_test_validation(training_set_path=training_original_path,
                                  test_set_path=testing_original_path,
                                  validation_set_path=validation_original_path,
                                  create_inverse_triples=False)
    print_partitions_info(training_triples=training_original,
                          training_triples_path=training_original_path,
                          validation_triples=validation_original,
                          validation_triples_path=validation_original_path,
                          testing_triples=testing_original,
                          testing_triples_path=testing_original_path)

    # Iteration over noise levels
    records = {}
    for noise_level in [
        TOTAL_RANDOM,
        ORIGINAL,
        NOISE_10,
        NOISE_20,
        NOISE_30,
    ]:
        print(f"\n\n#################### {noise_level} ####################\n")
        in_folder_path = os.path.join(dataset_models_folder_path, noise_level)
        assert dataset_name in in_folder_path
        assert noise_level in in_folder_path

        datasets_loader = TsvDatasetLoader(dataset_name=dataset_name, noise_level=noise_level)
        training_path, validation_path, testing_path = \
            datasets_loader.get_training_validation_testing_dfs_paths(noisy_test_flag=True)
        assert "training" in training_path
        assert noise_level in training_path
        assert dataset_name in training_path
        assert "validation" in validation_path
        assert noise_level in validation_path
        assert dataset_name in validation_path
        assert "testing" in testing_path
        assert noise_level in testing_path
        assert dataset_name in testing_path
        training, testing, validation = get_train_test_validation(training_set_path=training_path,
                                                                  test_set_path=testing_path,
                                                                  validation_set_path=validation_path,
                                                                  create_inverse_triples=False)
        print_partitions_info(training_triples=training,
                              training_triples_path=training_path,
                              validation_triples=validation,
                              validation_triples_path=validation_path,
                              testing_triples=testing,
                              testing_triples_path=testing_path)

        # Iteration over the trained models
        for model_name in sorted(os.listdir(in_folder_path)):

            print(f"\n >>>>> model_name: {model_name}")

            # Check on model name
            if model_name == RESCAL:
                continue
            elif model_name == CONVE:
                batch_size = 8
            else:
                batch_size = 256

            # ===== Read the evaluation metrics previously computed during the training of the models ===== #
            if load_precomputed_results_flag:

                # input results.json file path
                in_file_json = os.path.join(in_folder_path, model_name, "results.json")
                print(f"Load performance results computed during training from '{in_file_json}'...")
                print(in_file_json)
                assert model_name in in_file_json
                assert noise_level in in_file_json

                # if results.json is not already present skip to the next iteration
                if not os.path.isfile(in_file_json):
                    print("file json not present! \n")
                    continue

                # Read json file with precomputed evaluation metrics
                with open(in_file_json, "r") as json_file:
                    results = json.load(json_file)
                results_diz = results["metrics"]

            # ===== Perform Inference on test set with the trained models and compute evaluation metrics ===== #
            else:

                file_json = os.path.join(dataset_results_folder_path, f"{model_name}_{noise_level}_results.json")

                if force_inference or (not os.path.isfile(file_json)):
                    print("Inference on testing set and evaluation...")

                    # input model path
                    in_file_model = os.path.join(in_folder_path, model_name, "trained_model.pkl")
                    print(in_file_model)
                    assert model_name in in_file_model
                    assert noise_level in in_file_model

                    # if model wa not already trained, skip to the next iteration
                    if not os.path.isfile(in_file_model):
                        print("model not present! \n")
                        continue

                    # Load model from FS
                    my_pykeen_model = torch.load(in_file_model)

                    # Define evaluator
                    evaluator = RankBasedEvaluator(
                        filtered=True,  # Note: this is True by default; we're just being explicit
                    )

                    # Evaluate your model with not only testing triples
                    results = evaluator.evaluate(
                        model=my_pykeen_model,
                        mapped_triples=testing_original.mapped_triples,
                        additional_filter_triples=[
                            training.mapped_triples,  # filter on training triples with noisy
                            validation.mapped_triples,  # filter on validation triples with noisy
                        ],
                        batch_size=batch_size,
                        slice_size=None,
                        device=device,
                        use_tqdm=True,
                    )
                    results_diz = results.to_dict()

                    # write json file on FS
                    with open(file_json, "w") as json_file:
                        json.dump(obj=results_diz, fp=json_file, indent=4, ensure_ascii=True)

                else:
                    print(f"Load performance results from '{file_json}'...")
                    with open(file_json, "r") as json_file:
                        results_diz = json.load(fp=json_file)

            # Parsing of Metrics from results dictionary
            assert isinstance(results_diz, dict)
            assert strategy1 in results_diz
            assert isinstance(results_diz[strategy1], dict)
            assert strategy2 in results_diz[strategy1]
            mr = round(results_diz[strategy1][strategy2]["arithmetic_mean_rank"], 1)
            mrr = round(results_diz[strategy1][strategy2]["inverse_harmonic_mean_rank"], 3)
            hits_at_1 = round(results_diz[strategy1][strategy2]["hits_at_1"], 3)
            hits_at_3 = round(results_diz[strategy1][strategy2]["hits_at_3"], 3)
            hits_at_5 = round(results_diz[strategy1][strategy2]["hits_at_5"], 3)
            hits_at_10 = round(results_diz[strategy1][strategy2]["hits_at_10"], 3)

            # Update internal current record diz
            current_record = dict()
            if noise_level == ORIGINAL:
                noise_level_k = ""
            else:
                noise_level_k = noise_level
            if MR in selected_metrics:
                current_record[f"{MR}_{noise_level_k}".rstrip("_")] = mr
            if MRR in selected_metrics:
                current_record[f"{MRR}_{noise_level_k}".rstrip("_")] = mrr
            if HITS_AT_1 in selected_metrics:
                current_record[f"{HITS_AT_1}_{noise_level_k}".rstrip("_")] = hits_at_1
            if HITS_AT_3 in selected_metrics:
                current_record[f"{HITS_AT_3}_{noise_level_k}".rstrip("_")] = hits_at_3
            if HITS_AT_5 in selected_metrics:
                current_record[f"{HITS_AT_5}_{noise_level_k}".rstrip("_")] = hits_at_5
            if HITS_AT_10 in selected_metrics:
                current_record[f"hits_at_X_{noise_level_k}".rstrip("_")] = hits_at_10

            # Update external records diz
            if model_name not in records:
                records[model_name] = current_record  # insert new record
            else:
                records[model_name] = {**records[model_name], **current_record}  # update (merge)

    # Summarize and export the results after the end of iterations
    print("\n\n\n >>> Build DataFrame...")
    df_results = pd.DataFrame(data=list(records.values()),
                              index=list(records.keys())).T

    # Format for a better view the dataframe with the link prediction performance metrics
    df_results = df_results.sort_index(inplace=False, axis=0, ascending=True)
    diz_results_records = df_results.to_dict(orient="records")
    diz_results_index = list(df_results.index.values)
    step = 5
    i = 0
    new_records = []
    new_index = []
    for i_name, record in zip(diz_results_index, diz_results_records):
        if i % step == 0:
            new_index.append(f"(*)")
            new_records.append({k: np.nan for k in df_results.columns})
        new_index.append(i_name)
        new_records.append(record)
        i += 1
    df_results2 = pd.DataFrame(data=new_records, index=new_index)

    # Print the dataframe with the link prediction performance metrics
    print("\n>>> df info:")
    print(df_results2.info(memory_usage="deep"))
    print("\n>>> df overview:")
    print(df_results2)

    # Export to FS the dataframe with the link prediction performance metrics
    print("\n >>> Export DataFrame to FS...")
    out_path = os.path.join(dataset_results_folder_path, f"link_prediction_{strategy1}_{strategy2}_results.xlsx")
    print(f"\t out_path: {out_path}")
    assert dataset_name in out_path
    assert out_path.endswith("results.xlsx")
    assert str(out_path.split(os.path.sep)[-1]).startswith("link_prediction")
    if (os.path.isfile(out_path)) and (not force_saving):
        raise OSError(f"'{out_path}' already exists!")
    df_results2.to_excel(out_path, header=True, index=True, encoding="utf-8", engine="openpyxl")
