import os

from config import FB15K237, FB15K237_FOLDER_PATH, \
    WN18RR, WN18RR_FOLDER_PATH, \
    YAGO310, YAGO310_FOLDER_PATH, \
    COUNTRIES_FOLDER_PATH, COUNTRIES
from core.noise_generation import NoiseGenerator
from dao.data_model import NoiseLevel
from dao.dataset_loading import TsvDatasetLoader

if __name__ == '__main__':

    for dataset_name, dataset_folder in [
        (FB15K237, FB15K237_FOLDER_PATH),
        (WN18RR, WN18RR_FOLDER_PATH),
        (YAGO310, YAGO310_FOLDER_PATH),
        (COUNTRIES, COUNTRIES_FOLDER_PATH),
    ]:

        print(f"\n\n>>>>>>>>>>>>>>>>>>>> {dataset_name} <<<<<<<<<<<<<<<<<<<<")

        tsv_dataset_loader = TsvDatasetLoader(dataset_name=dataset_name,
                                              noise_level=NoiseLevel.ZERO.value)

        df_training, df_validation, df_testing = tsv_dataset_loader.get_training_validation_testing_dfs()
        print(f"training_shape={df_training.shape} \n"
              f"validation_shape={df_validation.shape} \n"
              f"testing_shape={df_testing.shape} \n")

        model_name = "cbgan_model_v1"
        noise_generator = NoiseGenerator(models_folder_path=dataset_folder,
                                         training_df=df_training,
                                         validation_df=df_validation,
                                         testing_df=df_testing,
                                         training_sample=5000,
                                         batch_size=1000,
                                         epochs=150)
        try:
            noise_generator.load_model(model_name=model_name)
            print("model load from FS!")
        except FileNotFoundError as fnf_err:
            print(fnf_err)
            noise_generator.train()
            noise_generator.store_model(model_name=model_name)

        for noise_percentage_num, noise_percentage_folder in [
            (1, "noise_1"),
            (5, "noise_5"),
            (10, "noise_10"),
        ]:
            print(f"\n{'-' * 80}")
            print(f"{noise_percentage_num}  |  {noise_percentage_folder}")

            noisy_dataset = noise_generator.generate_noisy_dataset(noise_percentage=noise_percentage_num)

            print(f"\n ### NOISY TRAINING ({noise_percentage_num}%) ###")
            print(noisy_dataset.training_df.shape)
            print(len(noisy_dataset.training_y_fake))
            noisy_dataset.training_df.to_csv(
                path_or_buf=os.path.join(dataset_folder, noise_percentage_folder, "training.tsv"),
                sep="\t", header=True, index=False, encoding="utf-8"
            )
            noisy_dataset.training_y_fake.to_csv(
                path_or_buf=os.path.join(dataset_folder, noise_percentage_folder, "training_y_fake.tsv"),
                sep="\t", header=True, index=False, encoding="utf-8"
            )

            print(f"\n ### NOISY VALIDATION ({noise_percentage_num}%) ###")
            print(noisy_dataset.validation_df.shape)
            print(len(noisy_dataset.validation_y_fake))
            noisy_dataset.validation_df.to_csv(
                path_or_buf=os.path.join(dataset_folder, noise_percentage_folder, "validation.tsv"),
                sep="\t", header=True, index=False, encoding="utf-8"
            )
            noisy_dataset.validation_y_fake.to_csv(
                path_or_buf=os.path.join(dataset_folder, noise_percentage_folder, "validation_y_fake.tsv"),
                sep="\t", header=True, index=False, encoding="utf-8"
            )

            print(f"\n ### NOISY TESTING ({noise_percentage_num}%) ###")
            print(noisy_dataset.testing_df.shape)
            print(len(noisy_dataset.testing_y_fake))
            noisy_dataset.testing_df.to_csv(
                path_or_buf=os.path.join(dataset_folder, noise_percentage_folder, "testing.tsv"),
                sep="\t", header=True, index=False, encoding="utf-8"
            )
            noisy_dataset.testing_y_fake.to_csv(
                path_or_buf=os.path.join(dataset_folder, noise_percentage_folder, "testing_y_fake.tsv"),
                sep="\t", header=True, index=False, encoding="utf-8"
            )

            print(f"{'-' * 80}\n")
