# @title Infrastructure: check the runtime and create this run's folder
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import time
import urllib.request
import uuid
import zipfile

SESSION_START = time.perf_counter()
if platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise RuntimeError('The canonical environment is Linux x86_64 Google Colab with a T4 GPU.')
try:
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
except FileNotFoundError as exc:
    raise RuntimeError('Select Runtime > Change runtime type > T4 GPU before Run all.') from exc
if gpu.returncode:
    raise RuntimeError('Select a fresh T4 GPU runtime; CUDA was not detected.')
print(gpu.stdout.strip())
ROOT = Path.cwd() / 'outputs' / 'image_segmentation' / uuid.uuid4().hex[:12]
ROOT.mkdir(parents=True)
