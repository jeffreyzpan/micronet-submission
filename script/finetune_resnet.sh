#!/bin/bash

set -e

NET=$1

dataset=cifar10
epochs=300
pretrained=pretrained/cifar10_${NET}_164/model_best.pth.tar

set -x
python main_old.py ../../datasets/CIFAR10/cifar --pretrained ${pretrained} \
	--dataset ${dataset} --arch ${NET} --save_path ./snapshots/finetune_${dataset}_${NET}_${epochs} --epochs ${epochs} \
	--schedule 120 180 240 --gammas 0.1 0.1 0.1 --learning_rate 0.1 --decay 0.0001 --batch_size 128 --workers 16 --ngpu 4
