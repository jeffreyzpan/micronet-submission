# Micronet Submission
This is our submission to the 2019 MicroNet Challenge hosted at NeurIPS 2019.

Submitters:

Jeffrey Pan (<jpan21@andover.edu>),
Kuan Wang (<kuanwang@mit.edu>), 
Han Cai (<hancai@mit.edu>), 
Song Han (<songhan@mit.edu>)

## Methodology

This submission used [Proxyless NAS](https://github.com/mit-han-lab/ProxylessNAS) to first search an architecture optimized for the CIFAR100 dataset. 

Then, we used [TTQ](https://github.com/czhu95/ternarynet) to quantize all our layers to two-bit weights, excluding our depthwise convolutions, as they have fewer parameters. 

### Parameter Calculations

The total number of parameters is computed by the ```count_parameters``` function in ```models/utils.py```. For each TTQ-quantized layer in the network, the parameter storage is decreased from 32-bit to 2-bit, but two more 32-bit parameters are required for the scaling factors of the layer. Thus, the parameter storage of a TTQ-quantized layer is n/16 + 2, where n is the original number of parameters.

### FLOPS Calculations

The total number of FLOPS can be found in the function ```net_flops``` in the file ```models/networks/run_manager.py```, which calls the ```get_flops``` functions of each layer in ```models/modules/layers.py```. As we do not do any FLOP quantization, the total number of FLOPS is the sum of the FLOPS in each layer.

## Results

 Models                   | # of Parameters |Parameter Storage | FLOPS | Top1 Acc (%) |
| ------------------------ | --------------| -------------- | ------------ | ------------ |
| WideResNet-28-10 (baseline)|    36.5M   |       139.2MB      |     10.49B    |    81.7     |
| Ours |       5.77M      |      3.85MB        |    1.213B    |    82.44     |

## Usage

### Evaluate

To evaluated our pretrained checkpoint, run:
```
python run_exp.py --resume --quantize --gpu <list of gpus> --path Nets/cifar100_final
```

where the `--resume` flag specifies a pretrained checkpoint and the `--quantize` flag uses TTQ.

### Finetune Full Precision Model

To quantize a pretrained full-precision model, use:
```
python run_exp.py --train --valid --resume --quantize --gpu <list of gpus> --path <path to pretrained full-precision model>
```




