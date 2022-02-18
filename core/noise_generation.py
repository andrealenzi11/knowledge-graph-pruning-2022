from sdv.tabular import GaussianCopula


class NoiseGenerator:

    def __init__(self, in_folder_path: str):
        self.in_folder_path = in_folder_path
        self.model = GaussianCopula()

    def train(self):
        pass

    def generate_noise_1(self):
        pass

    def generate_noise_5(self):
        pass

    def generate_noise_10(self):
        pass
