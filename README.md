# Micronet Submission
This is my submission to the 2019 MicroNet Challenge hosted at NeurIPS 2019.

## Methodology

This submission used [Proxyless NAS](https://github.com/mit-han-lab/ProxylessNAS) to first search an architecture optimized for the CIFAR100 dataset. 

Next, we used [TTQ](https://github.com/czhu95/ternarynet) to quantize all our layers to two-bit weights, excluding our depthwise convolutions, as they have fewer parameters. 

Finally, we used [HAQ](https://github.com/mit-han-lab/haq-release) to quantize the remaining layers to 8-bit precision. 

## Results

 Models                   | Parameters | FLOPS | Top1 Acc (%) |
| ------------------------ | -------------- | ------------ | ------------ |
| WideResNet-28-10 (baseline)   |       36.5M      |     10.49B    |    81.7     |
| Ours |       1.01M      |     1.213B    |    82.44     |

## Usage

### Evaluate

To evaluated our pretrained checkpoint, run:
```
python run_exp.py --resume --quantize --quantize_dw --path Nets/cifar100_final
```

where the `--resume` flag specifies a pretrained checkpoint, the `--quantize` flag uses TTQ, and the `--quantize_dw` flag uses HAQ.

### Finetune Full Precision Model

To quantize a pretrained full-precision model, use:
```
python run_exp.py --train --valid --resume --quantize --gpu <list of gpus> --path <path to pretrained full-precision model>
```




