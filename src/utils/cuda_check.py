import torch

# ==========================
# CHECK CUDA AVAILABILITY
# ==========================

# Check CUDA availability and device
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device name : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# Simple test tensor on CUDA
device = torch.device("cuda")
x = torch.ones(1, 1).to(device)
print(f"Il tensore è su: {x.device}")