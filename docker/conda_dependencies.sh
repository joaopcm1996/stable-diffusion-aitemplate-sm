#!/bin/bash

CONDA_ENV_PATH=$1

conda create -y -n stablediff_env python=3.8
source ~/anaconda3/etc/profile.d/conda.sh
source activate stablediff_env
export PYTHONNOUSERSITE=True
pip install torch click
pip install transformers ftfy scipy accelerate
pip install diffusers
pip install transformers[onnxruntime]
pip install conda-pack

# Install AI Template
git clone --recursive https://github.com/facebookincubator/AITemplate
cd AITemplate/python
python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall

# Switch out file to support A10G GPU's
cd .. && cd ..
cp detect_target.py "$CONDA_ENV_PATH"/stablediff_env/lib/python3.8/site-packages/aitemplate/testing/detect_target.py

conda-pack