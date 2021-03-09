# DDA

## Paper

Code release for "Dynamic Domain Adaptation for Efficient Inference" (CVPR2021)

Our work proposes a Dynamic Domain Adaptation (DDA) framework to solve the problem of efficient inference in the context of domain adaptation.

## Dependencies
The code runs with Python3 and requires Pytorch of version 1.3.1 or higher. Please `pip install` the following packages:
- `numpy`
- `torch`
- `heaq`
- `math`
- `random`
- `datetime`

## Pre-trained models

Pre-trained models for backbone MSDNet can be downloaded [here](https://github.com/BIT-DA/DDA/releases) and change the `--pretrain_path` argument.

## Training
VisDA 2017 dataset can be found [here](https://github.com/VisionLearningGroup/taskcv-2017-public) in the classification track.

Run the following command in shell:

```
visda-2017 anytime

python train_dda.py --gpu_id id --dset visda --s_dset_path ../data/visda-2017/train_list.txt --t_dset_path ../data/visda-2017/validation_list.txt --test_dset_path ../data/visda-2017/validation_list.txt --pattern anytime

visda-2017 budgeted batch

python train_dda.py --gpu_id id --dset visda --s_dset_path ../data/visda-2017/train_list.txt --t_dset_path ../data/visda-2017/validation_list.txt --test_dset_path ../data/visda-2017/validation_list.txt --pattern budget
```

- Change `--base 4 --step 4` to `--base 7 --step  7` to run  DDA(step-7), and change the pretrain model path.

- See `train_dda.py` for details. 
*****

DomainNet dataset can be found [here](http://ai.bu.edu/M3SDA/)

Run the following command in shell:

```
DomainNet anytime

python train_dda.py --gpu_id id --dset domainnet --s_dset_path ../data/domainnet/clipart_train.txt --t_dset_path ../data/domainnet/infograph_train.txt --test_dset_path ../data/domainnet/infograph_test.txt --pattern anytime

DomainNet budgeted batch

python train_dda.py --gpu_id id --dset domainnet --s_dset_path ../data/domainnet/clipart_train.txt --t_dset_path ../data/domainnet/infograph_train.txt --test_dset_path ../data/domainnet/infograph_test.txt --pattern budget
```

Same options are available as in visda-2017.

## Acknowledgements
Some codes in this project are adapted from [CDAN](https://github.com/thuml/CDAN) and [MSDNet](https://github.com/kalviny/MSDNet-PyTorch). We thank them for their excellent projects.

## Citation
If you find this code useful for your research, please cite our paper:
```
@inproceedings{li2019joint,
author = {Shuang Li and Jinming Zhang and Wenxuan Ma and Chi Harold Liu and Wei Li},
title = {Dynamic Domain Adaptation for Efficient Inference},
year = {2021},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
pages = {-},
numpages = {9}
}
```

## Contact

If you have any problem about our code, feel free to contact
- jm-zhang@bit.edu.cn
- wenxuanma@bit.edu.cn

or describe your problem in Issues.
