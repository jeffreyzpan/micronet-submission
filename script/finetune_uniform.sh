#!/bin/bash

set -e

NET=$1
BITS=$2

dataset=cifar10
epochs=100
pretrained=snapshots/finetune_cifar10_${NET}_300/model_best.pth.tar

set -x
python main_old.py ../../datasets/CIFAR10/cifar --resume ${pretrained} \
	--dataset ${dataset} --arch rnq${NET} --save_path ./snapshots/uniform_${BITS}_quantized_${dataset}_rnq${NET}_${epochs} --epochs ${epochs} \
	--schedule 30 60 90 --gammas 0.1 0.1 0.1 --learning_rate 0.01 --decay 0.0001 --batch_size 128 --workers 16 --ngpu 4 --aq_type 'uniform' --aq_bits ${BITS}
