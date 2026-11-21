#!/usr/bin/env python3
"""Install the independent SAM2 comparator in the current Python environment.

Use a virtual environment with Python >=3.10. Then: python install_sam2.py
This downloads PyTorch/TorchVision and the pinned official Meta repository.
The optional CUDA extension is disabled because this project runs on CPU.
No user files are uploaded and no Classroom action is performed.
"""
import os
import subprocess
import sys

if sys.version_info<(3,10):raise SystemExit('SAM2 requires Python 3.10 or later.')
subprocess.run([sys.executable,'-m','pip','install','torch>=2.5.1','torchvision>=0.20.1','setuptools>=77','wheel'],check=True)
env=os.environ.copy();env['SAM2_BUILD_CUDA']='0'
subprocess.run([sys.executable,'-m','pip','install','--no-build-isolation',
    'git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4'],env=env,check=True)
