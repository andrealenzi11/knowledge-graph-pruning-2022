import os

from src.config.config import ORIGINAL, NOISE_10, NOISE_20, NOISE_30, NOISE_100, \
    TRAINING_TSV, TRAINING_Y_FAKE_TSV, \
    VALIDATION_TSV, VALIDATION_Y_FAKE_TSV, \
    TESTING_TSV, TESTING_Y_FAKE_TSV, \
    RANDOM_SEED_TRAINING, RANDOM_SEED_VALIDATION, RANDOM_SEED_TESTING, \
    FB15K237, WN18RR, YAGO310, COUNTRIES, CODEXSMALL, NATIONS, \
    FB15K237_DATASETS_FOLDER_PATH, WN18RR_DATASETS_FOLDER_PATH, YAGO310_DATASETS_FOLDER_PATH, \
    COUNTRIES_DATASETS_FOLDER_PATH, CODEXSMALL_DATASETS_FOLDER_PATH, NATIONS_DATASETS_FOLDER_PATH
from src.core.noise_generation import DeterministicNoiseGenerator
from src.dao.dataset_loading import TsvDatasetLoader


if __name__ == '__main__':

    for dataset_name, dataset_folder in [
        (FB15K237, FB15K237_DATASETS_FOLDER_PATH),
        (WN18RR, WN18RR_DATASETS_FOLDER_PATH),
        (YAGO310, YAGO310_DATASETS_FOLDER_PATH),
        (COUNTRIES, COUNTRIES_DATASETS_FOLDER_PATH),
        (CODEXSMALL, CODEXSMALL_DATASETS_FOLDER_PATH),
        (NATIONS, NATIONS_DATASETS_FOLDER_PATH),
    ]:

        print(f"\n\n>>>>>>>>>>>>>>>>>>>> {dataset_name} <<<<<<<<<<<<<<<<<<<<")
        assert dataset_name in dataset_folder

        tsv_dataset_loader = TsvDatasetLoader(dataset_name=dataset_name,
                                              noise_level=ORIGINAL)

        print(f"\n >>> Original Dataset '{dataset_name}' Info:")

        assert "training" in tsv_dataset_loader.in_path_noisy_df_training
        assert ORIGINAL in tsv_dataset_loader.in_path_noisy_df_training
        assert "validation" in tsv_dataset_loader.in_path_noisy_df_validation
        assert ORIGINAL in tsv_dataset_loader.in_path_noisy_df_validation
        assert "testing" in tsv_dataset_loader.in_path_noisy_df_testing
        assert ORIGINAL in tsv_dataset_loader.in_path_noisy_df_testing

        print(" - Paths:")
        df_training, df_validation, df_testing = \
            tsv_dataset_loader.get_training_validation_testing_dfs(noisy_test_flag=False)

        print(" - Shapes:")
        print(f"\t\t\t training_shape={df_training.shape} \n"
              f"\t\t\t validation_shape={df_validation.shape} \n"
              f"\t\t\t testing_shape={df_testing.shape} \n")

        noise_generator = DeterministicNoiseGenerator(training_df=df_training,
                                                      validation_df=df_validation,
                                                      testing_df=df_testing,
                                                      random_states_training=RANDOM_SEED_TRAINING,
                                                      random_states_validation=RANDOM_SEED_VALIDATION,
                                                      random_states_testing=RANDOM_SEED_TESTING)

        for noise_percentage_num, noise_percentage_folder in [
            (10, NOISE_10),
            (20, NOISE_20),
            (30, NOISE_30),
            (100, NOISE_100),
        ]:
            print(f"\n{'-' * 80}")
            print(f"{noise_percentage_num}  |  {noise_percentage_folder}")
            assert str(noise_percentage_num) in noise_percentage_folder

            noisy_dataset = noise_generator.generate_noisy_dataset(noise_percentage=noise_percentage_num)

            print(f"\n ### NOISY TRAINING ({noise_percentage_num}%) ###")
            print(noisy_dataset.training_df.shape)
            print(len(noisy_dataset.training_y_fake))
            assert noisy_dataset.training_df.shape[0] == len(noisy_dataset.training_y_fake)
            assert noisy_dataset.training_df.shape[0] > df_training.shape[0]
            assert len(noisy_dataset.training_y_fake) > df_training.shape[0]
            training_df_out_path = os.path.join(dataset_folder, noise_percentage_folder, TRAINING_TSV)
            training_y_fake_out_path = os.path.join(dataset_folder, noise_percentage_folder, TRAINING_Y_FAKE_TSV)
            print(training_df_out_path)
            print(training_y_fake_out_path)
            assert "training" in training_df_out_path
            assert noise_percentage_folder in training_df_out_path
            assert "training" in training_y_fake_out_path
            assert noise_percentage_folder in training_y_fake_out_path
            noisy_dataset.training_df.to_csv(
                path_or_buf=training_df_out_path,
                sep="\t", header=False, index=False, encoding="utf-8"
            )
            noisy_dataset.training_y_fake.to_csv(
                path_or_buf=training_y_fake_out_path,
                sep="\t", header=False, index=False, encoding="utf-8"
            )

            print(f"\n ### NOISY VALIDATION ({noise_percentage_num}%) ###")
            print(noisy_dataset.validation_df.shape)
            print(len(noisy_dataset.validation_y_fake))
            assert noisy_dataset.validation_df.shape[0] == len(noisy_dataset.validation_y_fake)
            assert noisy_dataset.validation_df.shape[0] > df_validation.shape[0]
            assert len(noisy_dataset.validation_y_fake) > df_validation.shape[0]
            validation_df_out_path = os.path.join(dataset_folder, noise_percentage_folder, VALIDATION_TSV)
            validation_y_fake_out_path = os.path.join(dataset_folder, noise_percentage_folder, VALIDATION_Y_FAKE_TSV)
            print(validation_df_out_path)
            print(validation_y_fake_out_path)
            assert "validation" in validation_df_out_path
            assert noise_percentage_folder in validation_df_out_path
            assert "validation" in validation_y_fake_out_path
            assert noise_percentage_folder in validation_y_fake_out_path
            noisy_dataset.validation_df.to_csv(
                path_or_buf=validation_df_out_path,
                sep="\t", header=False, index=False, encoding="utf-8"
            )
            noisy_dataset.validation_y_fake.to_csv(
                path_or_buf=validation_y_fake_out_path,
                sep="\t", header=False, index=False, encoding="utf-8"
            )

            print(f"\n ### NOISY TESTING ({noise_percentage_num}%) ###")
            print(noisy_dataset.testing_df.shape)
            print(len(noisy_dataset.testing_y_fake))
            assert noisy_dataset.testing_df.shape[0] == len(noisy_dataset.testing_y_fake)
            assert noisy_dataset.testing_df.shape[0] > df_testing.shape[0]
            assert len(noisy_dataset.testing_y_fake) > df_testing.shape[0]
            testing_df_out_path = os.path.join(dataset_folder, noise_percentage_folder, TESTING_TSV)
            testing_y_fake_out_path = os.path.join(dataset_folder, noise_percentage_folder, TESTING_Y_FAKE_TSV)
            print(testing_df_out_path)
            print(testing_y_fake_out_path)
            assert "testing" in testing_df_out_path
            assert noise_percentage_folder in testing_df_out_path
            assert "testing" in testing_y_fake_out_path
            assert noise_percentage_folder in testing_y_fake_out_path
            noisy_dataset.testing_df.to_csv(
                path_or_buf=testing_df_out_path,
                sep="\t", header=False, index=False, encoding="utf-8"
            )
            noisy_dataset.testing_y_fake.to_csv(
                path_or_buf=testing_y_fake_out_path,
                sep="\t", header=False, index=False, encoding="utf-8"
            )

            print(f"{'-' * 80}\n")
