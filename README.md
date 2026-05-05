# AlphaPose - PyTorch2.x

## Overview

Alphapose adaptation to `Python > 3.8`, `PyTorch > 1.13` and `YOLO26`.

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

pip install natsort
pip install cython-bbox
```

Then go to `setup.py` and replace lines 176-182

```python
def is_installed(package_name):
    #from pip._internal.utils.misc import get_installed_distributions
    import pkg_resources
    for p in pkg_resources.working_set:
        if package_name in p.egg_name():
            return True
    return False
```

with

```python
def is_installed(package_name):
    from importlib.util import find_spec
    return find_spec(package_name) is not None
```

finally

```bash
pip install . --no-build-isolation
python setup.py build_ext --inplace
```

in `alphapose/utils` create the file `safe_import.py`

```python
def import_tkinter():
    try:
        from tkinter import _flatten
    except ImportError:
        def _flatten(seq):
            def _inner(seq):
                for item in seq:
                    if isinstance(item, (list, tuple)):
                        yield from _inner(item)
                    else:
                        yield item
            return tuple(_inner(seq))
```

Then in the files `halpe_26`, `halpe_68`, `halpe_136`, `halpe_coco_wholebody_26`, `halpe_coco_wholebody_136`, `single_hand` and `coco_wholebody` in the `alphapose/dataset/` folder replace

```python
from tkinter import _flatten
```

with

```python
from alphapose.utils.safe_import import import_tkinter
import_tkinter()
```

## Usage

```bash
python -m scripts.inference_yolo26 \
    --cfg ./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
    --checkpoint ./model_files/fast_res50_256x192.pth \
    --det-weights ./detector/yolo26/data/yolo26m.pt \
    --source 0
```
