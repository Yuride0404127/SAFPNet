# SAFPNet
## Super-pixel Auxiliary Feature Prompt Network for No-Service Rail Surface Defect Detection
This project provides the code and results for SAFPNet: 'Super-pixel Auxiliary Feature Prompt Network for No-Service Rail Surface Defect Detection'

# Requirements
### Python >=3.6
### Pytorch >= 1.7.1
### Cuda = 10.2
### For specific environmental details, see *requirements.txt*

# Architecture and Details
## The model's structure is as follows：
![SAFPNet](https://github.com/Yuride0404127/SAFPNet/Picture/SAFPNet.jpg)

## The pseudo code for our training algorithm is shown below:
![Algorithm](https://github.com/Yuride0404127/SAFPNet/Picture/Algorithm.png)

# Results
## Results in RSDD
![Table1](https://github.com/Yuride0404127/SAFPNet/Picture/Table1.png)

## Results in RGBD-SOD
![Table2](https://github.com/Yuride0404127/SAFPNet/Picture/Table2.png)

## PR-surves
![PR-surves](https://github.com/Yuride0404127/SAFPNet/Picture/PR-surves.png)

# Training & Testing

modify the train_root in RGBT_dataprocessing_CNet.py according to your own data path.

- Train the SAFPNet:
  - `python train_RGBT_prompt.py`
- Test the SAFPNet:
 - `python test_RGBT_prompt.py`

# Saliency Maps
- RSDD [baiduNetdsk]() / [Google drive]()
- SOD  [baiduNetdsk]() / [Google drive]()

# Pretraining Models
- RSDD [baiduNetdsk]() / [Google drive]()
- SOD  [baiduNetdsk]() / [Google drive]()