import torch


def get_cuda_info() -> str:
    dev = torch.device("cuda:0")
    t1 = torch.randn(3, ).to(dev)
    out_s1 = f"\n{'-'*80} \n " \
             f">>> torch version {torch.__version__} \n " \
             f">>> cuda is available: {torch.cuda.is_available()} \n " \
             f">>> cuda version: {torch.version.cuda} \n " \
             f">>> device count: {torch.cuda.device_count()} \n " \
             f">>> index of currently selected cuda device: {torch.cuda.current_device()} \n " \
             f"cuda.device(0): {torch.cuda.device(0)} \n " \
             f">>> cuda device name: {torch.cuda.get_device_name(0)} \n " \
             f">>> cuda device properties: {torch.cuda.get_device_properties(0)} \n " \
             f">>> Try to access 'cuda:0' ... \n " \
             f"\t dev: {dev} \n " \
             f"\t t1 = {t1} \n " \
             f"\t t1 is cuda: {t1.is_cuda} \n " \
             f"{'-'*80} \n"
    return out_s1


def print_cuda_info():
    cuda_info_str = get_cuda_info()
    print(cuda_info_str)


if __name__ == '__main__':
    print_cuda_info()
