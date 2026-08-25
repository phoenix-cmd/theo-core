import sys, platform, json

print("=" * 60)
print("HARDWARE / ENVIRONMENT PREFLIGHT REPORT")
print("=" * 60)
print()
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Processor: {platform.processor()}")
print(f"Machine: {platform.machine()}")
print()

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    for i in range(torch.cuda.device_count()):
        print(f"torch.cuda.get_device_name({i}): {torch.cuda.get_device_name(i)}")
        cap = torch.cuda.get_device_capability(i)
        print(f"torch.cuda.get_device_capability({i}): {cap}")
        mem = torch.cuda.get_device_properties(i)
        print(f"  Total memory: {mem.total_mem / 1024**3:.1f} GB")
        print(f"  Compute capability: {mem.major}.{mem.minor}")
else:
    print("No CUDA GPU detected by PyTorch.")
print()
print(f"CUDA compiled: {torch.version.cuda is not None}")
try:
    if torch.backends.cudnn.is_available():
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    else:
        print("cuDNN: not available")
except Exception as e:
    print(f"cuDNN check error: {e}")
print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
print(f"XPU available: {hasattr(torch, 'xpu') and torch.xpu.is_available()}")
try:
    print(f"Default device: {torch.get_default_device()}")
except AttributeError:
    print("Default device: N/A (older PyTorch)")
print(f"CPU threads: {torch.get_num_threads()}")
print(f"CPU inter-op threads: {torch.get_num_interop_threads()}")
print()

import numpy as np
print(f"NumPy version: {np.__version__}")

import transformers
print(f"transformers version: {transformers.__version__}")

try:
    import peft
    print(f"peft version: {peft.__version__}")
except ImportError:
    print("peft: NOT INSTALLED")

try:
    import sklearn
    print(f"sklearn version: {sklearn.__version__}")
except ImportError:
    print("sklearn: NOT INSTALLED")

print()
print("=" * 60)
print("END PREFLIGHT REPORT")
print("=" * 60)
