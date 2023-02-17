#!/bin/bash

MODEL_REPO_NAME=$1

wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -b -p /opt/conda
export PATH=/opt/conda/bin:$PATH

# Untar conda env file
cd /home/condpackenv/
tar -xf stablediff_env.tar.gz

# Activate env
cd /home/
source activate condpackenv/

# Clone and navigate to AITemplate repo; replace download pipeline file
git clone --recursive https://github.com/facebookincubator/AITemplate
cp /model_repository/workspace/download_pipeline.py AITemplate/examples/05_stable_diffusion/scripts/download_pipeline.py

# Compile models
cd AITemplate/examples/05_stable_diffusion/
python scripts/download_pipeline.py
python scripts/compile.py

# Move compiled models to Triton model repo
MODEL_NAME=$(ls "$MODEL_REPO_NAME"/)
mv tmp/ /model_repository/"$MODEL_REPO_NAME"/"$MODEL_NAME"/