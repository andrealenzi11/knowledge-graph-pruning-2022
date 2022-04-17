import torch


def print_cuda_info():
    print("\n-----------------------------------------------------------------")
    print(f">>> torch version {torch.__version__}")
    print(f">>> cuda is available: {torch.cuda.is_available()}")
    print(f">>> cuda version: {torch.version.cuda}")
    print(f">>> device count: {torch.cuda.device_count()}")
    print(f">>> index of currently selected cuda device: {torch.cuda.current_device()}")
    print(f">>> cuda.device(0): {torch.cuda.device(0)}")
    print(f">>> cuda device name: {torch.cuda.get_device_name(0)}")
    print(f">>> cuda device properties: {torch.cuda.get_device_properties(0)}")
    print(f">>> Try to access 'cuda:0' ...")
    dev = torch.device("cuda:0")
    print(f"\t dev: {dev}")
    t1 = torch.randn(3,).to(dev)
    print(f"\t t1 = {t1}")
    print(f"\t t1 is cuda: {t1.is_cuda}")
    print("-----------------------------------------------------------------\n")


if __name__ == '__main__':
    print_cuda_info()
