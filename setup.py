from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import sys
import os

PYTORCH_PATH = "/opt/pytorch"

class CustomInstallCommand(install):
    def run(self):
        if os.path.exists(PYTORCH_PATH):
            sys.path.insert(0, PYTORCH_PATH)
            try:
                import torch
                print(f"Found existing PyTorch installation: {torch.__version__}")
            except ImportError:
                print(f"PyTorch not found in {PYTORCH_PATH}. Installing via pip...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
        else:
            print(f"{PYTORCH_PATH} not found. Installing PyTorch via pip...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
        
        install.run(self)

with open('environment/requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="dna_mdl",
    version="0.1.0",
    description="DNA sequence modeling tools for internal analyses",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=required,
    python_requires=">=3.7",
    cmdclass={
        'install': CustomInstallCommand,
    },
)
