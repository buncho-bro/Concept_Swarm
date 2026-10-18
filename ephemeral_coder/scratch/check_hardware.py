import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"VRAM Allocated: {torch.cuda.memory_allocated(0)/1024**2:.1f}MB")
    print(f"VRAM Reserved: {torch.cuda.memory_reserved(0)/1024**2:.1f}MB")
else:
    print("Using CPU")
