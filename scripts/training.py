import os

from pykeen.models import *

from config import COUNTRIES, FB15K237, WN18RR, YAGO310, CODEXSMALL, \
    create_non_existent_folder, \
    NOISE_5, NOISE_10, ORIGINAL, NOISE_15
from core.fabrication import DatasetModelsFolderPathFactory
from core.pykeen_wrapper import get_train_test_validation, train, store, load
from dao.dataset_loading import TsvDatasetLoader

if __name__ == '__main__':

    # === Set your training configuration === #
    dataset_name: str = CODEXSMALL  # COUNTRIES, WN18RR, FB15K237, YAGO310, CODEXSMALL
    force_training: bool = False
    num_epochs = 200  # default: 5
    batch_size = 256  # default: 256
    stopper = "early"  # "early" | None

    all_datasets_names = {COUNTRIES, WN18RR, FB15K237, YAGO310, CODEXSMALL}
    print(f"all_datasets_names: {all_datasets_names}")
    dataset_models_folder_path = DatasetModelsFolderPathFactory().get(dataset_name=dataset_name)

    print(f"\n{'*' * 80}")
    print("TRAINING CONFIGURATION")
    print(f"\t\t dataset_name: {dataset_name}")
    print(f"\t\t dataset_models_folder_path: {dataset_models_folder_path}")
    print(f"\t\t force_training: {force_training}")
    print(f"\t\t num_epochs: {num_epochs}")
    print(f"\t\t batch_size {batch_size}")
    print(f"\t\t stopper: {stopper}")
    print(f"{'*' * 80}\n\n")

    for noise_level in [
        ORIGINAL,
        NOISE_5,
        NOISE_10,
        NOISE_15,
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

        for model_name, model_class_name, year in [
            ("RESCAL", RESCAL, 2011),
            ("TransE", TransE, 2013),
            ("DistMult", DistMult, 2014),
            ("TransH", TransH, 2014),
            ("TransR", TransR, 2015),
            ("TransD", TransD, 2015),
            ("ComplEx", ComplEx, 2016),
            ("HolE", HolE, 2016),
            ("ConvE", ConvE, 2018),
            # ("ConvKB", ConvKB, 2018), # MemoryError: The current model can't be trained on this hardware
            # ("RGCN", RGCN, 2018),  # RuntimeError: CUDA out of memory (sometimes)
            ("RotatE", RotatE, 2019),
            ("PairRE", PairRE, 2020),
            ("AutoSF", AutoSF, 2020),
            ("BoxE", BoxE, 2020),
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
                print(f"\t - Training '{model_name}' model...")
                pipeline_result = train(training=training,
                                        testing=testing,
                                        validation=validation,
                                        kge_model_obj=model_class_name,
                                        num_epochs=num_epochs,
                                        batch_size=batch_size,
                                        stopper=stopper)
                print(f"- Saving '{model_name}' model......")
                store(result_model=pipeline_result,
                      out_dir_path=out_folder_path)

            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")

        print(f"\n###########################################################\n")
