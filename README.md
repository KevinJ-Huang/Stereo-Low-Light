# Low-Light Stereo Image Enhancement (TMM 2022)

Jie Huang, Xueyang Fu, Zeyu Xiao, Feng Zhao, and Zhiwei Xiong(*)

*Corresponding Author

University of Science and Technology of China (USTC)

## Introduction

This repository is the **official implementation** of the paper, "Low-Light Stereo Image Enhancement", where more implementation details are presented.

### 0. Hyper-Parameters setting

Overall, most parameters can be set in options/train/train_Enhance_Holopix.yml or options/train/train_Enhance_Midddlebury.yml

### 1. Dataset Preparation

Create a .txt file to put the path of the dataset using 

```python
python create_txt.py
```

### 2. Training

```python
python train.py --opt options/train/train_Enhance_Middlebury.yml or train_Enhance_Holopix.yml
```


### 3. Inference

```python
python eval.py 
```

## Dataset (coming soon)

Holopix50k dataset  https://drive.google.com/file/d/1rP72ioPkm-Nvv8fBU-PT8IlICf51pDVa/view?usp=sharing

train/test folder(ground truth)

test_real folder(no ground truth)

Middlebury dataset  https://drive.google.com/file/d/1q3FbFszkCDDVJHC4yN8lPnp11Pn-tWkD/view?usp=sharing

## Ours results (coming soon)

Result on Holopix50k dataset https://drive.google.com/file/d/10RPvw7hMBQk-FXCdXULYCzNJG6SR4UvF/view?usp=sharing

Result on Middlebiry dataset https://drive.google.com/file/d/13j7h3lL1wV7vaV0l0O3B-Ra_jnQ8vPgt/view?usp=sharing

Result on real samples of Holopix50k dataset https://drive.google.com/file/d/1XAXd_tz_XHh_vz8KSTdg43F16RG7SNJL/view?usp=sharing

## Contact

If you have any problem with the released code, please do not hesitate to contact me by email (hj0117@mail.ustc.edu.cn).

## Cite

```
@ARTICLE{9720943,
  author={Huang, Jie and Fu, Xueyang and Xiao, Zeyu and Zhao, Feng and Xiong, Zhiwei},
  journal={IEEE Transactions on Multimedia}, 
  title={Low-Light Stereo Image Enhancement}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMM.2022.3154152}}
```
