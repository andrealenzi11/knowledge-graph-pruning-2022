from dao.data_model import DatasetName, DatasetPartition, NoiseLevel
from dao.dataset_loading import TsvDatasetLoader

if __name__ == '__main__':
    df = TsvDatasetLoader(dataset_name=DatasetName.FB15K237,
                          dataset_partition=DatasetPartition.TRAINING,
                          noise_level=NoiseLevel.ZERO).get_tsv_dataset()
    print(df)
