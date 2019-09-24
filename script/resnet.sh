#!/bin/bash

set -e

export PYTHONUNBUFFERED="True"

LOG="resnet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
COMMIT="commit `git rev-parse HEAD`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
echo "$COMMIT"

model=$1
dataset=cifar10
epochs=164

set -x
python3 main.py cifar.python \
	--dataset ${dataset} --arch ${model} --save_path ./snapshots/${dataset}_${model}_${epochs} --epochs ${epochs} \
	--schedule 82 123 --gammas 0.1 0.1 --learning_rate 0.1 --decay 0.0001 --batch_size 128 --workers 16 --ngpu 2
