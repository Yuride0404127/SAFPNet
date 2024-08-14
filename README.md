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
![SAFPNet](https://github.com/Yuride0404127/SAFPNet/blob/main/Picture/SAFPNet.jpg)

## The pseudo code for our training algorithm is shown below:
![Algorithm](https://github.com/Yuride0404127/SAFPNet/blob/main/Picture/Algorithm.png)

# Results
## Results in RSDD
![Table1](https://github.com/Yuride0404127/SAFPNet/blob/main/Picture/Table1.png)

## Results in RGBD-SOD
![Table2](https://github.com/Yuride0404127/SAFPNet/blob/main/Picture/Table2.png)

## PR-surves
![PR-curves](https://github.com/Yuride0404127/SAFPNet/blob/main/Picture/PR-curves.bmp)

# Training & Testing

modify the train_root in RGBT_dataprocessing_CNet.py according to your own data path.

- Train the SAFPNet:
  - `python train_RGBT_prompt.py`
- Test the SAFPNet:
  - `python test_RGBT_prompt.py`

# Saliency Maps
- SDD [BaiduNetdsk](https://pan.baidu.com/s/1ESfIKXiljzuWCqkMyCmiPQ?pwd=mjog) / [Google drive](https://drive.google.com/file/d/1ieNKmPJMTiNQovxt78WPMkFivugNHswu/view?usp=drive_link)
- SOD [BaiduNetdsk](https://pan.baidu.com/s/1M91MLQ6m5z3dHyMZIv4B8g?pwd=4d0y) / [Google drive](https://drive.google.com/file/d/19ziiHF2qlFYlqDimwZBF5UqSEIJH1U-l/view?usp=drive_link)

# Pretraining Models
- SDD [BaiduNetdsk](https://pan.baidu.com/s/125VlikXk4j4i-SJkaPp9ow?pwd=moi2) / [Google drive](https://drive.google.com/file/d/1fpcz-gvAbNEW6qYsvOpPZkEOdl1mQCDf/view?usp=drive_link)
- SOD  [BaiduNetdsk](https://pan.baidu.com/s/1_TJLyGS1uB0UehAwJsReSA?pwd=3gn3) / [Google drive](https://drive.google.com/file/d/1aBfPRgVwzn4kWjtRBcKR3nGNlYHMeGGl/view?usp=drive_link)