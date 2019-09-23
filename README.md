# SCAN: A Scalabel Neural Networks Framework for Compact and Efficient Models
A pytorch implementation of paper *SCAN: A Scalabel Neural Networks Framework for Compact and Efficient Models*.

## Requirements
Install PyTorch>=1.0.0, torchvision>=0.2.0.

Download and process the CIFAR datasets by torchvision.

## How to train
    python train.py [--depth=18] [--class_num=100] [--epoch=200] [--lambda_KD=0.5]
**depth** indicates the number of layers in resnet. 

**class_num** decides which dataset will be used (cifar10/100). 

**epoch** indicates how many epoches will be utilized to train this model.

**lambda_KD** is a hyper-parameter for balancing distillation loss and cross entropy loss.

## Dynamatic inference

    python inference.py [--depth=18]
Only a pre-trained ResNet18 model is prepared now, stored in **model** folder. inference.py will use it to inference, and print its accuracy and acceleration ratio. By adjusting the thresholds in line30 in inference.py, you can get different accuracy and acceleration results.
