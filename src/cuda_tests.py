import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the CUDA device count
    device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices: {device_count}")

    # Get the CUDA device name
    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA device name: {device_name}")

    # Get the CUDA version
    cuda_version = torch.version.cuda
    print(f"CUDA version: {cuda_version}")
else:
    print("CUDA is not available on this system.")