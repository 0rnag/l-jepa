#!/bin/bash

# setup for running on cloud
set -e
set -x 
mkdir -p models figures data

cd data
wget -N https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
cd ..

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
touch setup_complete.flag

echo "Setup completed successfully"