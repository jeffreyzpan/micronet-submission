#!/bin/bash

set -e

NET=$1

set -x
python ../zip.py --arch ${NET} --dir zip \
    --model snapshots/finetune_cifar10_${NET}_164/model_best.pth.tar \
    --out snapshots/finetune_cifar10_${NET}_164/model_zipped.pth.tar
