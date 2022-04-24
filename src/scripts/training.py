import argparse
import os

from pykeen.version import VERSION as PYKEEN_VERSION
from torch.version import __version__ as torch_version

from src.config.config import COUNTRIES, FB15K237, WN18RR, YAGO310, CODEXSMALL, NATIONS, \
    create_non_existent_folder, \
    ORIGINAL, NOISE_10, NOISE_20, NOISE_30, \
    TRANSE, DISTMULT, TRANSH, COMPLEX, HOLE, CONVE, ROTATE, PAIRRE, AUTOSF, BOXE, MODELS_DIR
from src.core.hyper_configuration_parsing import get_best_hyper_parameters_diz, parse_best_hyper_parameters_diz
from src.core.pykeen_wrapper import get_train_test_validation, train, store, load, print_partitions_info
from src.dao.dataset_loading import DatasetPathFactory, TsvDatasetLoader
from src.utils.cuda_info import print_cuda_info
from src.utils.printing import print_and_write

all_datasets_names = [
    FB15K237,  # "FB15K237"
    WN18RR,  # "WN18RR"
    YAGO310,  # "YAGO310"
    COUNTRIES,  # "COUNTRIES"
    CODEXSMALL,  # "CODEXSMALL"
    NATIONS,  # "NATIONS"
]

valid_kge_models = [
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


if __name__ == '__main__':

    fp = os.path.join(MODELS_DIR, "training_log.txt")

    with open(fp, "w") as fw_log:

        # Instantiate the parser for the command line arguments
        args_parser = argparse.ArgumentParser()

        # Add command line arguments entries
        args_parser.add_argument('dataset',
                                 help='Dataset Name',
                                 type=str,
                                 choices=all_datasets_names)
        args_parser.add_argument('-f', '--force',
                                 dest="force",
                                 help='Boolean flag for force training or not',
                                 type=int,
                                 required=False,
                                 default=1)

        # Parse command line arguments
        cl_args = args_parser.parse_args()

        # Access to the command line arguments
        print_and_write(out_file=fw_log, text=f"Argument values: \n\t {cl_args} \n")
        dataset_name = str(cl_args.dataset).upper().strip()
        print_and_write(out_file=fw_log, text=f"dataset_name: {dataset_name}")
        force_training = bool(cl_args.force)
        print_and_write(out_file=fw_log, text=f"force_training: {force_training}")

        # Get the folder path for the specified dataset where store the trained models
        dataset_models_folder_path = DatasetPathFactory(dataset_name=dataset_name).get_models_folder_path()
        assert dataset_name in dataset_models_folder_path

        # check on cuda
        print_cuda_info()
        print_and_write(out_file=fw_log, text=f"\n{'*' * 80}")
        print_and_write(out_file=fw_log, text="TRAINING CONFIGURATION")
        print_and_write(out_file=fw_log, text=f"\t\t dataset_name: {dataset_name}")
        print_and_write(out_file=fw_log, text=f"\t\t dataset_models_folder_path: {dataset_models_folder_path}")
        print_and_write(out_file=fw_log, text=f"\t\t force_training: {force_training}")
        print_and_write(out_file=fw_log, text=f"\t\t pykeen version: {PYKEEN_VERSION}")
        print_and_write(out_file=fw_log, text=f"\t\t torch version: {torch_version}")
        print_and_write(out_file=fw_log, text=f"{'*' * 80}\n\n")

        # === Iterate over noise levels === #
        for noise_level in [
            ORIGINAL,
            NOISE_10,
            NOISE_20,
            NOISE_30,
        ]:
            print_and_write(out_file=fw_log, text=f"\n\n#################### {noise_level} ####################\n")
            datasets_loader = TsvDatasetLoader(dataset_name=dataset_name, noise_level=noise_level)

            training_path, validation_path, testing_path = \
                datasets_loader.get_training_validation_testing_dfs_paths(noisy_test_flag=False)
            assert "training" in training_path
            assert noise_level in training_path
            assert dataset_name in training_path
            assert "validation" in validation_path
            assert noise_level in validation_path
            assert dataset_name in validation_path
            assert "testing" in testing_path
            assert ORIGINAL in testing_path  # test for link prediction task on the original testing set (without noise)
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

            print_and_write(out_file=fw_log, text="\n\n>>> Start KGE models training...\n")

            # === Iterate over KGE models === #
            for model_name, year in [
                # (RESCAL, 2011),
                (TRANSE, 2013),
                (DISTMULT, 2014),
                (TRANSH, 2014),
                (COMPLEX, 2016),
                (HOLE, 2016),
                (CONVE, 2018),
                (ROTATE, 2019),
                (PAIRRE, 2020),
                (AUTOSF, 2020),
                (BOXE, 2020),
            ]:
                print_and_write(out_file=fw_log,
                                text=f"\n>>>>>>>>>>>>>>>>>>>> {model_name} ({year}) <<<<<<<<<<<<<<<<<<<<")

                out_folder_path = os.path.join(dataset_models_folder_path, noise_level, model_name)
                assert dataset_name in out_folder_path
                assert noise_level in out_folder_path
                assert model_name in out_folder_path
                print_and_write(out_file=fw_log, text=f"\n\t - out_folder_path: {out_folder_path}")
                create_non_existent_folder(folder_path=out_folder_path)

                # Try to Load an already trained KGE model from File System
                if force_training:
                    force_training2 = True
                else:
                    print_and_write(out_file=fw_log,
                                    text=f"\t - Try to loading already trained '{model_name}' model...")
                    try:
                        pipeline_result = load(in_dir_path=out_folder_path)
                        print_and_write(out_file=fw_log, text=f"\t <> '{model_name}' model loaded from File System!")
                        force_training2 = False
                    except FileNotFoundError:
                        print_and_write(out_file=fw_log,
                                        text=f"\t <> '{model_name}' model NOT already present in the File System!")
                        force_training2 = True

                # Train a new KGE model and store it on File System
                if force_training2:
                    print_and_write(out_file=fw_log, text="\t - Load best Hyper-parameters")
                    hyperparams_diz = get_best_hyper_parameters_diz(current_dataset_name=dataset_name,
                                                                    current_model_name=model_name)
                    print_and_write(out_file=fw_log, text=f"\t\t <> retrieved hyper-parameters: {hyperparams_diz}")
                    kwargs_diz = parse_best_hyper_parameters_diz(best_hyper_params_diz=hyperparams_diz)
                    print_and_write(out_file=fw_log, text=f"\t\t <> parsed hyper-parameters: {kwargs_diz}")

                    print_and_write(out_file=fw_log, text=f"\t - Training '{model_name}' model...")
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

                    print_and_write(out_file=fw_log, text=f"- Saving '{model_name}' model......")
                    store(result_model=pipeline_result,
                          out_dir_path=out_folder_path)

                print_and_write(out_file=fw_log, text=f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")

            print_and_write(out_file=fw_log, text=f"\n###########################################################\n")

        print_and_write(out_file=fw_log, text="EXIT 0")
