import torch
import torchvision


def check_cuda():
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"Torchvision 版本: {torchvision.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"GPU 数量: {device_count}")

        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")
            print(f"  计算能力: {torch.cuda.get_device_capability(i)}")

        print(f"当前设备索引: {torch.cuda.current_device()}")
        print(f"当前设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")


if __name__ == "__main__":
    check_cuda() 