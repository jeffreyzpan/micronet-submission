This is an anonymous repository for justifying our experiment results. 

# Requirements
* Python 3.6
* Pytorch 0.3.1
* ImageNet Dataset (The default path to imagenet dataset is **/ssd/dataset/imagenet**. If you use a different path, please refer to **Code/ImageNet/data_providers/imagenet.py** and modify the default path accordlingly.)

# Pre-trained Models

## Proxyless (Mobile) on ImageNet
Please run the following command under the folder of **scripts** to check our optimized model for mobile.
```bash
bash eval_imagenet_mobile.sh 
```
Then you will get
```bash
test_loss: 1.104153	 test_acc1: 74.594000	 test_acc5: 92.202000
```

## Proxyless (CPU) on ImageNet
Please run the following command under the folder of **scripts** to check our optimized model for CPU
```bash
bash eval_imagenet_cpu.sh 
```
Then you will get
```bash
test_loss: 1.072649	 test_acc1: 75.290000	 test_acc5: 92.394000
```

## Proxyless (GPU) on ImageNet
Please run the following command under the folder of **scripts** to check our optimized model for GPU
```bash
bash eval_imagenet_gpu.sh 
```
Then you will get
```bash
test_loss: 1.073663	 test_acc1: 75.084000	 test_acc5: 92.538000
```

Note after we finetune our training settings to match previous NAS works, the model accuracy results are further improved compared to the early submitted version.

## Proxyless on CIFAR-10
Please run the following command under the folder of **scripts** to check our optimized model on CIFAR-10
```bash
bash eval_cifar.sh 
```
Then you will get
```bash
test_loss: 0.082096	 test_acc1: 97.920000
```
