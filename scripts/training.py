import os

from pykeen.models import AutoSF, BoxE, ComplEx, ConvE, DistMult, PairRE, \
    RESCAL, RGCN, RotatE, TransD, TransE, TransH, TransR

from config import COUNTRIES, FB15K237, WN18RR, YAGO310, \
    COUNTRIES_MODELS_FOLDER_PATH, FB15K237_MODELS_FOLDER_PATH, WN18RR_MODELS_FOLDER_PATH, YAGO310_MODELS_FOLDER_PATH, \
    create_non_existent_folder, \
    NOISE_1, NOISE_5, NOISE_10
from core.pykeen_wrapper import get_train_test_validation, train, store, load
from dao.dataset_loading import TsvDatasetLoader


if __name__ == '__main__':

    # Specify a Valid option: COUNTRIES, FB15K237, WN18RR, YAGO310
    DATASET_NAME: str = YAGO310
    FORCE_TRAINING: bool = False

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

    print(f"\n{'*'*80}")
    print(DATASET_NAME)
    print(DATASET_MODELS_FOLDER_PATH)
    print(f"{'*'*80}\n\n")

    for noise_level in [
        NOISE_1,
        NOISE_5,
        NOISE_10,
    ]:
        print(f"\n\n#################### {noise_level} ####################\n")
        datasets_loader = TsvDatasetLoader(dataset_name=DATASET_NAME, noise_level=noise_level)
        training_path, validation_path, testing_path = datasets_loader.get_training_validation_testing_dfs_paths()
        print(training_path)
        print(validation_path)
        print(testing_path)

        training, testing, validation = get_train_test_validation(training_set_path=training_path,
                                                                  test_set_path=testing_path,
                                                                  validation_set_path=validation_path,
                                                                  create_inverse_triples=False)

        # "TuckEr", "NodePiece", "CompGCN" do not work!

        for model_name, model_class_name, year in [
            ("RESCAL", RESCAL, 2011),
            ("TransE", TransE, 2013),
            ("DistMult", DistMult, 2014),
            ("TransH", TransH, 2014),
            ("TransR", TransR, 2015),
            ("TransD", TransD, 2015),
            ("ComplEx", ComplEx, 2016),
            # ("ConvE", ConvE, 2018),  # MemoryError: The current model can't be evaluated on this hardware
                                       # with these parameters, as evaluation batch_size=1 is too big and slicing
                                       # is not implemented for this model yet.
            # ("RGCN", RGCN, 2018),    # RuntimeError: CUDA out of memory
            ("RotatE", RotatE, 2019),
            ("PairRE", PairRE, 2020),
            ("AutoSF", AutoSF, 2020),
            ("BoxE", BoxE, 2020),
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
                                        kge_model_obj=model_class_name)
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
