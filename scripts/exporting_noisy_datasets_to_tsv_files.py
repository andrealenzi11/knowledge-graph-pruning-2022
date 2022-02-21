from core.noise_generation import NoiseGenerator
from dao.data_model import DatasetName, NoiseLevel
from dao.dataset_loading import TsvDatasetLoader


if __name__ == '__main__':

    tsv_dataset_loader = TsvDatasetLoader(dataset_name=DatasetName.YAGO310.value,
                                          noise_level=NoiseLevel.ZERO.value)

    df_training, df_validation, df_testing = tsv_dataset_loader.get_training_validation_testing_dfs()
    print(f"training_shape={df_training.shape} \n"
          f"validation_shape={df_validation.shape} \n"
          f"testing_shape={df_testing.shape} \n")

    noise_generator = NoiseGenerator(training_df=df_training,
                                     validation_df=df_validation,
                                     testing_df=df_testing)
    noise_generator.train()
    # noise_generator.store_model()
    noisy_dataset = noise_generator.generate_noisy_dataset(noise_percentage=5)

    print("\n\n ### NOISY TRAINING ###")
    print(noisy_dataset.training_df.shape)
    print(len(noisy_dataset.training_fake_y))
    print(noisy_dataset.training_df)

    print("\n\n ### NOISY VALIDATION ###")
    print(noisy_dataset.validation_df.shape)
    print(len(noisy_dataset.validation_fake_y))
    print(noisy_dataset.validation_df)

    print("\n\n ### NOISY TESTING ###")
    print(noisy_dataset.testing_df.shape)
    print(len(noisy_dataset.testing_fake_y))
    print(noisy_dataset.testing_df)
