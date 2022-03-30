import os

from pykeen.models import *

from config import COUNTRIES, FB15K237, WN18RR, YAGO310, \
    COUNTRIES_MODELS_FOLDER_PATH, FB15K237_MODELS_FOLDER_PATH, WN18RR_MODELS_FOLDER_PATH, YAGO310_MODELS_FOLDER_PATH, \
    create_non_existent_folder, \
    NOISE_1, NOISE_5, NOISE_10, ORIGINAL
from core.pykeen_wrapper import get_train_test_validation, train, store, load
from dao.dataset_loading import TsvDatasetLoader


if __name__ == '__main__':

    # Specify a Valid option: COUNTRIES, WN18RR, FB15K237, YAGO310
    DATASET_NAME: str = COUNTRIES
    FORCE_TRAINING: bool = True

    if DATASET_NAME == COUNTRIES:
        DATASET_MODELS_FOLDER_PATH = COUNTRIES_MODELS_FOLDER_PATH
    elif DATASET_NAME == FB15K237:
        DATASET_MODELS_FOLDER_PATH = FB15K237_MODELS_FOLDER_PATH
    elif DATASET_NAME == WN18RR:
        DATASET_MODELS_FOLDER_PATH = WN18RR_MODELS_FOLDER_PATH
    elif DATASET_NAME == YAGO310:
        DATASET_MODELS_FOLDER_PATH = YAGO310_MODELS_FOLDER_PATH
    else:
        raise ValueError(f"Invalid dataset name: '{str(DATASET_NAME)}'! \n"
                         f"\t\t Specify one of the following values: \n"
                         f"\t\t [{COUNTRIES}, {FB15K237}, {WN18RR}, {YAGO310}] \n")

    print(f"\n{'*' * 80}")
    print(DATASET_NAME)
    print(DATASET_MODELS_FOLDER_PATH)
    print(f"{'*' * 80}\n\n")

    for noise_level in [
        ORIGINAL,
        NOISE_1,
        NOISE_5,
        NOISE_10,
    ]:
        print(f"\n\n#################### {noise_level} ####################\n")
        datasets_loader = TsvDatasetLoader(dataset_name=DATASET_NAME, noise_level=noise_level)
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
            # ("ConvKB", ConvKB, 2018), # MemoryError: The current model can't be trained on this hardware with these parameters.
            ("RGCN", RGCN, 2018),  # RuntimeError: CUDA out of memory (sometimes)
            ("RotatE", RotatE, 2019),
            ("PairRE", PairRE, 2020),
            ("AutoSF", AutoSF, 2020),
            ("BoxE", BoxE, 2020),
            # === MemoryError: The current model can't be trained on this hardware with these parameters ===  #
            # ("TuckER", TuckER, 2019),
            # =============================================================================================== #
            # === AssertionError: assert triples_factory.create_inverse_triples === #
            # ("CompGCN", CompGCN, 2020),
            # ===================================================================== #
            # === ValueError: The provided triples factory does not create inverse triples === #
            # ("NodePiece", NodePiece, 2021),
            # ================================================================================ #
        ]:
            print(f"\n>>>>>>>>>>>>>>>>>>>> {model_name} ({year}) <<<<<<<<<<<<<<<<<<<<")

            out_folder_path = os.path.join(DATASET_MODELS_FOLDER_PATH, noise_level, model_name)
            print(f"\n out_folder_path: {out_folder_path}")
            create_non_existent_folder(folder_path=out_folder_path)

            if FORCE_TRAINING:
                print("Training...")
                pipeline_result = train(training=training,
                                        testing=testing,
                                        validation=validation,
                                        kge_model_obj=model_class_name,
                                        num_epochs=100,
                                        batch_size=256,
                                        stopper=None)
                store(result_model=pipeline_result, out_dir_path=out_folder_path)
            else:
                try:
                    print("Try to loading already trained KGE model...")
                    pipeline_result = load(in_dir_path=out_folder_path)
                    print("KGE model load from File System!")
                except FileNotFoundError:
                    print("Training...")
                    pipeline_result = train(training=training,
                                            testing=testing,
                                            validation=validation,
                                            kge_model_obj=model_class_name)
                    store(result_model=pipeline_result, out_dir_path=out_folder_path)

            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")

        print(f"\n###########################################################\n")
