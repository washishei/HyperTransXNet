# HyperTransXNet

A Python tool to perform deep learning experiments on various hyperspectral datasets.

![https://www.onera.fr/en/research/information-processing-and-systems-domain](https://www.onera.fr/sites/default/files/logo-onera.png|height=60)

![https://www-obelix.irisa.fr/](https://www.irisa.fr/sites/all/themes/irisa_theme/logoIRISA-web.png|width=60)

## Reference

## Requirements

It is based on the [PyTorch](http://pytorch.org/) deep learning and GPU computing framework and use the [Visdom](https://github.com/facebookresearch/visdom) visualization server.

## Setup

The easiest way to install this code is to create a [Python virtual environment](https://docs.python.org/3/tutorial/venv.html) and to install dependencies using:
`pip install -r requirements.txt`

(on Windows you should use `pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`)

In addition to the aforementioned libraries, the Mamba library should be incorporated. The relevant resources can be accessed via: [https://github.com/state-spaces/mamba]

## Hyperspectral datasets

Three publicly available hyperspectral datasets were utilized in this study, originally stored in the Datasets folder. Due to their substantial file sizes, the datasets have been hosted on a cloud platform with an access link provided. Users may directly retrieve and examine the data through this hyperlink.
This folder contains three datasets:

  * Indian Pines
  * Houston2013
  * WHU-Hi-LongKou

The folder has the following structure:

Datasets
    IndianPines
        Indian_pines_corrected.mat
        Indian_pines_gt.mat
    Houston2013
        gt.mat
        HSI.mat
        PaviaU.mat
        PaviaU_gt.mat
    WH
        WHU-Hi-LongKou
            Indian_pines_corrected.mat
            Indian_pines_gt.mat
            PaviaU_gt.mat
            PaviaU.mat

### Adding a new dataset

Adding a custom dataset can be done by modifying the `custom_datasets.py` file. Developers should add a new entry to the `CUSTOM_DATASETS_CONFIG` variable and define a specific data loader for their use case.

## Models

Currently, this tool implements several SVM variants from the [scikit-learn](http://scikit-learn.org/stable/) library and many state-of-the-art deep networks implemented in PyTorch.
  * 2D CNN ([Hyperspectral image classification with deep learning models, Yang et al, IEEE Transactions on Geoscience and Remote Sensing 2018](https://ieeexplore.ieee.org/abstract/document/8340197))
  * 3D CNN ([Synergistic 2D/3D convolutional neural network for hyperspectral image classification, Yang et al, Remote Sensing 2020](https://www.mdpi.com/2072-4292/12/12/2033))
  * HybridSN ([HybridSN: Exploring 3-D-2-D CNN feature hierarchy for hyperspectral image classification, with Application to Face Recognition, Roy et al, IEEE Geoscience and Remote Sensing Letters 2019](https://ieeexplore.ieee.org/abstract/document/8736016))
  * HiT ([Hyperspectral image transformer classification networks, Yang et al, IEEE Transactions on Geoscience and Remote Sensing 2022](https://ieeexplore.ieee.org/abstract/document/9766028))
  * MorphFormer ([Spectral-spatial morphological attention transformer for hyperspectral image classificationn, Roy et al, IEEE Transactions on Geoscience and Remote Sensing 2023](https://ieeexplore.ieee.org/abstract/document/10036472))
  * SSFTT ([Spectral-spatial feature tokenization transformer for hyperspectral image classification, Sun et al, IEEE Transactions on Geoscience and Remote Sensing 2022](https://ieeexplore.ieee.org/abstract/document/9684381))
  * MambaHSI ([MambaHSI: Spatial-spectral mamba for hyperspectral image classification, Li et al., IEEE Transactions on Geoscience and Remote Sensing 2024](https://ieeexplore.ieee.org/abstract/document/10604894))
  * HyperTransXNet
### Adding a new model

Adding a custom deep network can be done by modifying the `models.py` file. This implies creating a new class for the custom deep network and altering the `get_model` function.

## Usage

Start a Visdom server:
`python -m visdom.server`
and go to [`http://localhost:8097`](http://localhost:8097) to see the visualizations (or [`http://localhost:9999`](http://localhost:9999) if you use Docker).

Then, run the script `main.py`.

The most useful arguments are:
  * `--model` to specify the model (e.g. 'hit', 'conv2d', 'conv3d'),
  * `--dataset` to specify which dataset to use (e.g. 'IndianPines', 'Houston2013', 'WH'),
  * the `--cuda` switch to run the neural nets on GPU. The tool fallbacks on CPU if this switch is not specified.

There are more parameters that can be used to control more finely the behaviour of the tool. See `python main.py -h` for more information.

Examples:
  * `python main.py --model hit --dataset IndianPines --training_sample 0.3`
    This runs a grid search on HiT on the Indian Pines dataset, using 30% of the samples for training and the rest for testing. Results are displayed in the visdom panel.
  * `python main.py --model conv2d --dataset Houston2013 --training_sample 0.5 --patch_size 7 --epoch 50 --cuda`
    This runs on GPU the 2D CNN from Yang et al. on the Houston 2013 dataset with a patch size of 7, using 50% of the samples for training and optimizing for 50 epochs.
  * `python main.py --model hyperT --dataset WH --training_sample 0.5 --patch_size 15 --epoch 100 --cuda 0`
    This runs on GPU the HyperTransXNet on the WHU-Hi-LongKou dataset with a patch size of 15, using 50% of the samples for training and optimization for 100 epochs.

[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/nshaud)
