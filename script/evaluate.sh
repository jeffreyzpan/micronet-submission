#!/bin/bash

set -e
NET=$1
dataset=cifar10
epochs=164

set -x
python main.py cifar.python --evaluate \
	--dataset ${dataset} --arch ${NET} --pretrained ./snapshots/finetune_${dataset}_${NET}_${epochs}/model_restored.pth.tar \
	--workers 16 --ngpu 2
