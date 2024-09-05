#!/bin/bash

mkdir models
mkdir figures
mkdir data
cd data
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
cd ..

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
