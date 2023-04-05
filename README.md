# Semilayer-Wise-Mixed-Precision-Quantization
Semilayer-Wise-Mixed-Precision-Quantization is a Python module for quantization neural networks. This module has the following features.
* You can try the mixed-precision quantization algorithm in "A Mixed-Precision Quantization Method without Retraining or Accuracy Degradation Using Semilayers".
* It can be applied to convolution layers.
* Pretrained model should be the one described in lines 16~18 of resnet.py

## Requirements
* Python (>=3.7.6)
* Torch (>=1.5.0)
* Torchvision (>=0.60a0+82fd1c8)

## How to run the examples
### Run inference 
1. preparation
Perform the following procedure.
* Clone this Github
2. Move to sample code directory \
`cd /examples/<sample>`
3. Select and run ○○_main.py (ex.resnet18_main.py) \
`python3 resnet18_main.py`