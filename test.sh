#!/bin/bash

export PATH="/usr/local/cuda-11.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH"

if pgrep -x "python3.8" > /dev/null
then
    echo "Running"
else
    echo "Stopped"
    /usr/bin/python3.8 /home/tensorflow/word2vec/train.py
fi
