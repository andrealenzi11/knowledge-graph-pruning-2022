import os

from src.config.config import ORIGINAL, TRAINING_TSV, TRAINING_Y_FAKE_TSV, \
    VALIDATION_TSV, VALIDATION_Y_FAKE_TSV, \
    TESTING_TSV, TESTING_Y_FAKE_TSV, \
    RANDOM_SEED_TRAINING, RANDOM_SEED_VALIDATION, RANDOM_SEED_TESTING, \
    FB15K237, WN18RR, YAGO310, COUNTRIES, CODEXSMALL, NATIONS, \
    FB15K237_DATASETS_FOLDER_PATH, WN18RR_DATASETS_FOLDER_PATH, YAGO310_DATASETS_FOLDER_PATH, \
    COUNTRIES_DATASETS_FOLDER_PATH, CODEXSMALL_DATASETS_FOLDER_PATH, NATIONS_DATASETS_FOLDER_PATH, TOTAL_RANDOM
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

        random_dataset = noise_generator.generate_random_dataset()

        print(f"\n\n\n ### RANDOM TRAINING  ###")
        print(random_dataset.training_df.shape)
        print(len(random_dataset.training_y_fake))
        assert random_dataset.training_df.shape[0] == len(random_dataset.training_y_fake)
        training_df_out_path = os.path.join(dataset_folder, TOTAL_RANDOM, TRAINING_TSV)
        training_y_fake_out_path = os.path.join(dataset_folder, TOTAL_RANDOM, TRAINING_Y_FAKE_TSV)
        print(training_df_out_path)
        print(training_y_fake_out_path)
        assert "training" in training_df_out_path
        assert TOTAL_RANDOM in training_df_out_path
        assert "training" in training_y_fake_out_path
        assert TOTAL_RANDOM in training_y_fake_out_path
        random_dataset.training_df.to_csv(
            path_or_buf=training_df_out_path,
            sep="\t", header=False, index=False, encoding="utf-8"
        )
        random_dataset.training_y_fake.to_csv(
            path_or_buf=training_y_fake_out_path,
            sep="\t", header=False, index=False, encoding="utf-8"
        )

        print(f"\n ### RANDOM VALIDATION ###")
        print(random_dataset.validation_df.shape)
        print(len(random_dataset.validation_y_fake))
        assert random_dataset.validation_df.shape[0] == len(random_dataset.validation_y_fake)
        validation_df_out_path = os.path.join(dataset_folder, TOTAL_RANDOM, VALIDATION_TSV)
        validation_y_fake_out_path = os.path.join(dataset_folder, TOTAL_RANDOM, VALIDATION_Y_FAKE_TSV)
        print(validation_df_out_path)
        print(validation_y_fake_out_path)
        assert "validation" in validation_df_out_path
        assert TOTAL_RANDOM in validation_df_out_path
        assert "validation" in validation_y_fake_out_path
        assert TOTAL_RANDOM in validation_y_fake_out_path
        random_dataset.validation_df.to_csv(
            path_or_buf=validation_df_out_path,
            sep="\t", header=False, index=False, encoding="utf-8"
        )
        random_dataset.validation_y_fake.to_csv(
            path_or_buf=validation_y_fake_out_path,
            sep="\t", header=False, index=False, encoding="utf-8"
        )

        print(f"\n ### RANDOM TESTING ###")
        print(random_dataset.testing_df.shape)
        print(len(random_dataset.testing_y_fake))
        assert random_dataset.testing_df.shape[0] == len(random_dataset.testing_y_fake)
        testing_df_out_path = os.path.join(dataset_folder, TOTAL_RANDOM, TESTING_TSV)
        testing_y_fake_out_path = os.path.join(dataset_folder, TOTAL_RANDOM, TESTING_Y_FAKE_TSV)
        print(testing_df_out_path)
        print(testing_y_fake_out_path)
        assert "testing" in testing_df_out_path
        assert TOTAL_RANDOM in testing_df_out_path
        assert "testing" in testing_y_fake_out_path
        assert TOTAL_RANDOM in testing_y_fake_out_path
        random_dataset.testing_df.to_csv(
            path_or_buf=testing_df_out_path,
            sep="\t", header=False, index=False, encoding="utf-8"
        )
        random_dataset.testing_y_fake.to_csv(
            path_or_buf=testing_y_fake_out_path,
            sep="\t", header=False, index=False, encoding="utf-8"
        )

        del dataset_name, dataset_folder, tsv_dataset_loader, df_training, df_validation, df_testing
        del noise_generator, random_dataset
        del training_df_out_path, training_y_fake_out_path
        del validation_df_out_path, validation_y_fake_out_path,
        del testing_df_out_path, testing_y_fake_out_path
        print(f"{'-' * 80}\n")
