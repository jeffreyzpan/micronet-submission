#!/bin/bash

set -e

NET=$1

set -x
python ../zip.py --arch ${NET} --dir unzip \
    --model snapshots/finetune_cifar10_${NET}_164/model_zipped.pth.tar \
    --out snapshots/finetune_cifar10_${NET}_164/model_restored.pth.tar
