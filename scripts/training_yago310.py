from config import YAGO310, NOISE_1, NOISE_5, NOISE_10
from dao.dataset_loading import TsvDatasetLoader

if __name__ == '__main__':

    for noise_level in [
        NOISE_1,
        NOISE_5,
        NOISE_10,
    ]:
        print(f"\n\n#################### {noise_level} ####################\n")
        datasets_loader = TsvDatasetLoader(dataset_name=YAGO310, noise_level=noise_level)
        training_path, validation_path, testing_path = datasets_loader.get_training_validation_testing_dfs_paths()
        print(training_path)
        print(validation_path)
        print(testing_path)
