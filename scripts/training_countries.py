import os

from pykeen.models import AutoSF, BoxE, ComplEx, ConvE, DistMult, PairRE, \
    RESCAL, RGCN, RotatE, TransD, TransE, TransH, TransR

from config import COUNTRIES, NOISE_1, NOISE_5, NOISE_10, COUNTRIES_MODELS_FOLDER_PATH, create_non_existent_folder
from core.pykeen_wrapper import get_train_test_validation, train, store
from dao.dataset_loading import TsvDatasetLoader


if __name__ == '__main__':

    for noise_level in [
        NOISE_1,
        NOISE_5,
        NOISE_10,
    ]:
        print(f"\n\n#################### {noise_level} ####################\n")
        datasets_loader = TsvDatasetLoader(dataset_name=COUNTRIES, noise_level=noise_level)
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
            ("ConvE", ConvE, 2018),
            ("RGCN", RGCN, 2018),
            ("RotatE", RotatE, 2019),
            ("PairRE", PairRE, 2020),
            ("AutoSF", AutoSF, 2020),
            ("BoxE", BoxE, 2020),
        ]:
            print(f"\n>>>>>>>>>>>>>>>>>>>> {model_name} ({year}) <<<<<<<<<<<<<<<<<<<<")
            pipeline_result = train(training=training,
                                    testing=testing,
                                    validation=validation,
                                    kge_model_obj=model_class_name)
            out_folder_path = os.path.join(COUNTRIES_MODELS_FOLDER_PATH, noise_level, model_name)
            print(f"\n out_folder_path: {out_folder_path}")
            create_non_existent_folder(folder_path=out_folder_path)
            store(result_model=pipeline_result, out_dir_path=out_folder_path)
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")

        print(f"\n###########################################################\n")
