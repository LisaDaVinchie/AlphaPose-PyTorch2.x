# AlphaPose - PyTorch2.x

## Installation

Tested for python 3.11.15

```bash
python3 -m venv venv_alphapose

source venv_alphapose/bin/activate

pip install -U pip setuptools

pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126

export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

pip install cython

pip install ultralytics

sudo apt-get install libyaml-dev

python setup.py build develop
```

