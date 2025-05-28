# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init

# utils
import math
import os
import datetime
import numpy as np
import joblib

## time
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib.colors import ListedColormap

from tqdm import tqdm
from sklearn.decomposition import PCA
from utils import grouper, sliding_window, count_sliding_window, camel_to_snake, padding_image
from torch.optim.lr_scheduler import StepLR
from linformer import Linformer
from vit_pytorch.vit import ViT
from functools import partial
#from vit_pytorch.efficient import ViT
from vit_pytorch.local_vit import LocalViT
from vit_pytorch.t2t import T2TViT
from vit_pytorch.rvt import RvT
from vit_pytorch.pit import PiT
from vit_pytorch.deepvit import DeepViT
from vit_pytorch.cross_vit import CrossViT
from vit_pytorch.levit import LeViT
from vit_pytorch.cvt import CvT
from vit_pytorch.tnt import TNT
from vit_pytorch.vip import VisionPermutator, WeightedPermuteMLP
from vit_pytorch.hit import HiT, ConvPermuteMLP
from vit_pytorch.cait import CaiT
from vit_pytorch.ccvt import CCvT
from RCNN import RCNN
from conv2d import TESTEtAl
from conv3d import C3DEtAl
from yang import Yangnew
from Involution3 import I2DEtAl, I3DEtAl
from vit_pytorch.vipp import ViP
from vit_pytorch.dwt import DWT, ConvPermute
from vit_pytorch.swin_transformer import SwinTransformer
from vit_pytorch.coat import CoaT
from QCNN.eca_net import QConvNet
from vit_pytorch.qtn import QtN
from vit_pytorch.qltn import QLTN
from UNet_add import UNet, UNet_SIIS
from vit_pytorch.sequencer import Sequencer2D, Sequencer3D, SequencerConv3D
from vit_pytorch.AVT_new import AvT
from vit_pytorch.TAVT import TAvT
from vit_pytorch.TAVTB import  TAVTB
from vit_pytorch.TAVTT import TAVTT
from vit_pytorch.TAVTBA import TAVTBA
from vit_pytorch.SSFTTnet import SSFTTnet
from vit_pytorch.morphFormer import morphFormer
from UMMA.UMMA_main import UMMA

# from vit_pytorch.ummformer import ummformer
# from vit_pytorch.conformer_new import Conformer
from vit_pytorch.TIM import TiM
def get_model(name, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    device = kwargs.setdefault("device", torch.device("cpu"))
    n_classes = kwargs["n_classes"]
    n_bands = kwargs["n_bands"]
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs["ignored_labels"])] = 0.0
    weights = weights.to(device)
    weights = kwargs.setdefault("weights", weights)

    if name == "nn":
        kwargs.setdefault("patch_size", 1)
        center_pixel = True
        model = Baseline(n_bands, n_classes, kwargs.setdefault("dropout", False))
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == 'yang':
        kwargs.setdefault('patch_size', 15)
        center_pixel = True
        kwargs.setdefault('epoch', 100)
        # "The RNN was trained with the Adadelta algorithm [...] We made use of a
        # fairly  high  learning  rate  of  1.0  instead  of  the  relatively  low
        # default of  0.002 to  train the  network"
        lr = kwargs.setdefault('lr', 0.1)
        model = Yangnew(n_bands, n_classes, patch_size=15)
        # For Adadelta, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault("batch_size", 100)
    elif name == "slstm":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = Sequencer2D(in_channels=n_bands, num_classes=n_classes, depth=[3, 3, 5, 3], embed_dims=[1, 4, 8, 16], hidden_dims=[48, 96, 96, 96], expansion_factor=3)
        lr = kwargs.setdefault("learning_rate", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "slstm3":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = Sequencer3D(in_channels=n_bands, num_classes=n_classes, depth=[3, 3, 5, 3], embed_dims=[1, 4, 8, 16], hidden_dims=[48, 96, 96, 96], expansion_factor=3)
        lr = kwargs.setdefault("learning_rate", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "slstm3c":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = SequencerConv3D(in_channels=n_bands, num_classes=n_classes, depth=[3, 3, 5, 3], embed_dims=[1, 4, 8, 16], hidden_dims=[48, 96, 96, 96], expansion_factor=3)
        lr = kwargs.setdefault("learning_rate", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == 'morphformer':
        patch_size = kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = morphFormer(16, n_bands, n_classes, True)
        lr = kwargs.setdefault('learning_rate', 5e-3)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault('epoch', 100)
        kwargs.setdefault('batch_size', 100)
    elif name == 'ssftt': ##需要在main.py中对数据集进行PCA，在datasets.py中修改：data = np.zeros([30, h, w])
        patch_size = kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = SSFTTnet(num_classes=n_classes)
        lr = kwargs.setdefault("learning_rate", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        kwargs.setdefault("batch_size", 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == 'unet':
        kwargs.setdefault('patch_size', 16)
        center_pixel = True
        kwargs.setdefault('epoch', 100)
        # "The RNN was trained with the Adadelta algorithm [...] We made use of a
        # fairly  high  learning  rate  of  1.0  instead  of  the  relatively  low
        # default of  0.002 to  train the  network"
        lr = kwargs.setdefault('lr', 0.01)
        model = UNet(n_bands, n_classes)
        # For Adadelta, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault("batch_size", 100)
    elif name == 'unetn':
        kwargs.setdefault('patch_size', 16)
        center_pixel = True
        kwargs.setdefault('epoch', 100)
        # "The RNN was trained with the Adadelta algorithm [...] We made use of a
        # fairly  high  learning  rate  of  1.0  instead  of  the  relatively  low
        # default of  0.002 to  train the  network"
        lr = kwargs.setdefault('lr', 0.01)
        model = UNet(in_channels=n_bands, num_classes=n_classes)
        # For Adadelta, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault("batch_size", 100)
    elif name == 'umma':
        kwargs.setdefault('patch_size', 16)
        center_pixel = True
        kwargs.setdefault('epoch', 100)
        # "The RNN was trained with the Adadelta algorithm [...] We made use of a
        # fairly  high  learning  rate  of  1.0  instead  of  the  relatively  low
        # default of  0.002 to  train the  network"
        lr = kwargs.setdefault('lr', 0.01)
        model = UMMA(padding_val=0, input_dim=225, embd_dim=512, n_classes = n_classes,device_umma=device)
        # For Adadelta, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault("batch_size", 100)
    elif name == 'qcnn':
        kwargs.setdefault('patch_size', 15)
        center_pixel = True
        kwargs.setdefault('epoch', 100)
        # "The RNN was trained with the Adadelta algorithm [...] We made use of a
        # fairly  high  learning  rate  of  1.0  instead  of  the  relatively  low
        # default of  0.002 to  train the  network"
        lr = kwargs.setdefault('lr', 0.06)
        model = QConvNet(input_channel=n_bands, patch_size=15, n_classes=n_classes)
        # For Adadelta, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault("batch_size", 100)
    elif name == 'qtn':
        kwargs.setdefault('patch_size', 15)
        center_pixel = True
        kwargs.setdefault('epoch', 100)
        # "The RNN was trained with the Adadelta algorithm [...] We made use of a
        # fairly  high  learning  rate  of  1.0  instead  of  the  relatively  low
        # default of  0.002 to  train the  network"
        lr = kwargs.setdefault('lr', 0.5)
        model = QtN(dim=1024, image_size=15, patch_size=3, depth=6, heads=16, dropout=0.1, emb_dropout=0.1, num_classes=n_classes, channels=n_bands,)
        # For Adadelta, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault("batch_size", 100)
    elif name == 'qltn':
        kwargs.setdefault('patch_size', 15)
        center_pixel = True
        kwargs.setdefault('epoch', 100)
        # "The RNN was trained with the Adadelta algorithm [...] We made use of a
        # fairly  high  learning  rate  of  1.0  instead  of  the  relatively  low
        # default of  0.002 to  train the  network"
        lr = kwargs.setdefault('lr', 0.75)
        model = QLTN(img_size=15, in_chans=n_bands, num_classes=n_classes, embed_dims=[16, 64, 128, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2],)
        # For Adadelta, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault("batch_size", 100)
    elif name == "swint":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = SwinTransformer(
            hidden_dim=96,
            layers=(2, 2, 6, 2),
            heads=(3, 6, 12, 24),
            channels=n_bands,
            num_classes=n_classes,
            head_dim=32,
            window_size=2,
            downscaling_factors=(2, 1, 2, 1),
            relative_pos_embedding=True
            )
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    # elif name == "conformer":
    #     kwargs["supervision"] = "con"
    #     kwargs.setdefault("patch_size", 15)
    #     center_pixel = True
    #     model = Conformer(patch_size=16, channel_ratio=4, embed_dim=384, depth=12,
    #                   num_heads=6, mlp_ratio=4, qkv_bias=True, num_classes=n_classes, in_chans=n_bands)
    #     lr = kwargs.setdefault("learning_rate", 0.001)
    #     optimizer = optim.Adam(model.parameters(), lr=lr)
    #     criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    #     kwargs.setdefault("epoch", 100)
    #     kwargs.setdefault("batch_size", 100)
    elif name == "vit":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        # efficient_transformer = Linformer(
        #     dim=128,
        #     seq_len=25 + 1,  # 7x7 patches + 1 cls-token
        #     depth=12,
        #     heads=8,
        #     k=64
        # )
        # model = ViT(
        #     dim=128,
        #     image_size=15,
        #     patch_size=3,
        #     num_classes=n_classes,
        #     transformer=efficient_transformer,
        #     channels=n_bands,
        # )
        model = ViT(dim=128, image_size=15, patch_size=3, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1, num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0003)
        ### lr for efficient transformer
        #lr = kwargs.setdefault("learning_rate", 0.005)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "lvt":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = LocalViT(dim=128, image_size=15, patch_size=3, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1, num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "coat":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = CoaT(patch_size=4, embed_dims=[152, 216, 216, 216], serial_depths=[2, 2, 2, 2], parallel_depth=6, num_heads=8, mlp_ratios=[4, 4, 4, 4], img_size=64, num_classes=n_classes, in_chans=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "t2t":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = T2TViT(dim=512, image_size=15, depth=4, heads=8, mlp_dim=512, t2t_layers=((7, 2), (3, 2), (3, 2)), dropout=0.1, emb_dropout=0.1, num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "rvt":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = RvT(dim=128, image_size=15, patch_size=3, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1, num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0003)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "pit":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = PiT(dim=256, image_size=15, patch_size=3, depth=(3, 3, 3, 3), heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1, num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "dit":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = DeepViT(dim=1024, image_size=15, patch_size=3, depth=4, heads=8, mlp_dim=2048, dropout=0.1, emb_dropout=0.1, num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "crosvit":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = CrossViT(image_size=15, num_classes=n_classes, channels=n_bands, sm_dim=192, lg_dim=384, )
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "levit":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = LeViT(dim=(256, 384, 512), image_size=224, stages=3,  depth=4, heads=(4, 6, 8), mlp_mult=2, dropout=0.1, num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "cvt":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = CvT(num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "tim":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = TiM(num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "tavt":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = TAvT(num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "tavtb":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = TAVTB(num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "tavtba":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = TAVTBA(num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "tavtt":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = TAVTT(num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "avt":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = AvT(num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "ccvt":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = CCvT(num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "cait":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = CaiT(image_size=15, patch_size=3, num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "tnt":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = TNT(image_size=15, patch_dim=512, pixel_dim=24, patch_size=3, pixel_size=3, depth=6, heads=16, attn_dropout=0.1, ff_dropout=0.1, num_classes=n_classes, channels=n_bands,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "vip":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        layers = [4, 3, 14, 3]
        transitions = [False, True, False, False]
        segment_dim = [8, 8, 4, 4]
        mlp_ratios = [3, 3, 3, 3]
        embed_dims = [256, 256, 512, 512]
        model = VisionPermutator(layers, img_size=15, in_chans=n_bands, num_classes=n_classes, embed_dims=embed_dims, patch_size=3, transitions=transitions,
        segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=WeightedPermuteMLP,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "vipp":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = ViP(pretrained=False, in_chans=n_bands,inplanes=64, num_classes=n_classes, num_chs=(96, 192, 384, 768), patch_sizes=[1, 1, 1, 1], num_heads=[3, 6, 12, 24],
                     num_enc_heads=[1, 3, 6, 12], num_parts=[64, 64, 64, 64], num_layers=[1, 1, 3, 1], ffn_exp=3,
                     has_last_encoder=True, drop_path=0.1,)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "dwt":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        layers = [4, 3, 14, 3]
        #layers = [4, 3, 14]
        transitions = [False, True, False, False]
        #transitions = [False, True, False]
        segment_dim = [8, 8, 4, 4]
        #segment_dim = [8, 8, 4]
        mlp_ratios = [3, 3, 3, 3]
        #mlp_ratios = [3, 3, 3]
        embed_dims = [256, 256, 512, 512]
        #embed_dims = [256, 256, 512]
        model = DWT(layers, img_size=15, in_chans=n_bands, num_classes=n_classes, embed_dims=embed_dims, patch_size=3, transitions=transitions,
        segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=ConvPermute,)
        lr = kwargs.setdefault("learning_rate", 0.0001)## for KSC 0.000003
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 64)
    elif name == "hit":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        layers = [4, 3, 14, 3]
        transitions = [False, True, False, False]
        segment_dim = [8, 8, 4, 4]
        mlp_ratios = [3, 3, 3, 3]
        embed_dims = [288, 288, 256, 256]## for IN 368, for GRSS 256, for PU 168, for KSC 320 for XA 480
        model = HiT(layers, img_size=15, in_chans=n_bands, num_classes=n_classes, embed_dims=embed_dims, patch_size=3, transitions=transitions,
        segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=ConvPermuteMLP,)
        lr = kwargs.setdefault("learning_rate", 0.0001)## for KSC 0.000003
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "rcnn":
        patch_size = kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = RCNN(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        kwargs.setdefault("batch_size", 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "conv2d":
        patch_size = kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = TESTEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        kwargs.setdefault("batch_size", 50)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "conv3d":
        patch_size = kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = C3DEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        kwargs.setdefault("batch_size", 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "i2d":
        patch_size = kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = I2DEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.01)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        kwargs.setdefault("batch_size", 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "i3d":
        patch_size = kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = I3DEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.01)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        kwargs.setdefault("batch_size", 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "hamida":
        patch_size = kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = HamidaEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.01)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        kwargs.setdefault("batch_size", 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "lee":
        kwargs.setdefault("epoch", 200)
        patch_size = kwargs.setdefault("patch_size", 5)
        center_pixel = False
        model = LeeEtAl(n_bands, n_classes)
        lr = kwargs.setdefault("learning_rate", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "chen":
        patch_size = kwargs.setdefault("patch_size", 27)
        center_pixel = True
        model = ChenEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.003)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 400)
        kwargs.setdefault("batch_size", 100)
    elif name == "li":
        patch_size = kwargs.setdefault("patch_size", 5)
        center_pixel = True
        model = LiEtAl(n_bands, n_classes, n_planes=16, patch_size=patch_size)
        lr = kwargs.setdefault("learning_rate", 0.01)
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
        )
        epoch = kwargs.setdefault("epoch", 200)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        # kwargs.setdefault('scheduler', optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch // 2, (5 * epoch) // 6], gamma=0.1))
    elif name == "hu":
        kwargs.setdefault("patch_size", 1)
        center_pixel = True
        model = HuEtAl(n_bands, n_classes)
        # From what I infer from the paper (Eq.7 and Algorithm 1), it is standard SGD with lr = 0.01
        lr = kwargs.setdefault("learning_rate", 0.01)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "he":
        # We train our model by AdaGrad [18] algorithm, in which
        # the base learning rate is 0.01. In addition, we set the batch
        # as 40, weight decay as 0.01 for all the layers
        # The input of our network is the HSI 3D patch in the size of 7×7×Band
        kwargs.setdefault("patch_size", 7)
        kwargs.setdefault("batch_size", 40)
        lr = kwargs.setdefault("learning_rate", 0.01)
        center_pixel = True
        model = HeEtAl(n_bands, n_classes, patch_size=kwargs["patch_size"])
        # For Adagrad, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "luo":
        # All  the  experiments  are  settled  by  the  learning  rate  of  0.1,
        # the  decay  term  of  0.09  and  batch  size  of  100.
        kwargs.setdefault("patch_size", 3)
        kwargs.setdefault("batch_size", 100)
        lr = kwargs.setdefault("learning_rate", 0.1)
        center_pixel = True
        model = LuoEtAl(n_bands, n_classes, patch_size=kwargs["patch_size"])
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.09)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    elif name == "sharma":
        # We train our S-CNN from scratch using stochastic gradient descent with
        # momentum set to 0.9, weight decay of 0.0005, and with a batch size
        # of 60.  We initialize an equal learning rate for all trainable layers
        # to 0.05, which is manually decreased by a factor of 10 when the validation
        # error stopped decreasing. Prior to the termination the learning rate was
        # reduced two times at 15th and 25th epoch. [...]
        # We trained the network for 30 epochs
        kwargs.setdefault("batch_size", 60)
        epoch = kwargs.setdefault("epoch", 30)
        lr = kwargs.setdefault("lr", 0.05)
        center_pixel = True
        # We assume patch_size = 64
        kwargs.setdefault("patch_size", 64)
        model = SharmaEtAl(n_bands, n_classes, patch_size=kwargs["patch_size"])
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault(
            "scheduler",
            optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[epoch // 2, (5 * epoch) // 6], gamma=0.1
            ),
        )
    elif name == "hybridsn":
        kwargs.setdefault("patch_size", 15)
        center_pixel = True
        model = HybridEtAl(n_bands, n_classes, patch_size=15)
        lr = kwargs.setdefault("learning_rate", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)
        kwargs.setdefault("batch_size", 100)
    elif name == "liu":
        kwargs["supervision"] = "semi"
        # "The learning rate is set to 0.001 empirically. The number of epochs is set to be 40."
        kwargs.setdefault("epoch", 40)
        lr = kwargs.setdefault("lr", 0.001)
        center_pixel = True
        patch_size = kwargs.setdefault("patch_size", 9)
        model = LiuEtAl(n_bands, n_classes, patch_size)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        # "The unsupervised cost is the squared error of the difference"
        criterion = (
            nn.CrossEntropyLoss(weight=kwargs["weights"]),
            lambda rec, data: F.mse_loss(
                rec, data[:, :, :, patch_size // 2, patch_size // 2].squeeze()
            ),
        )
    elif name == "boulch":
        kwargs["supervision"] = "semi"
        kwargs.setdefault("patch_size", 1)
        kwargs.setdefault("epoch", 100)
        lr = kwargs.setdefault("lr", 0.001)
        center_pixel = True
        model = BoulchEtAl(n_bands, n_classes)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = (
            nn.CrossEntropyLoss(weight=kwargs["weights"]),
            lambda rec, data: F.mse_loss(rec, data.squeeze()),
        )
    elif name == "mou":
        kwargs.setdefault("patch_size", 1)
        center_pixel = True
        kwargs.setdefault("epoch", 100)
        # "The RNN was trained with the Adadelta algorithm [...] We made use of a
        # fairly  high  learning  rate  of  1.0  instead  of  the  relatively  low
        # default of  0.002 to  train the  network"
        lr = kwargs.setdefault("lr", 1.0)
        model = MouEtAl(n_bands, n_classes)
        # For Adadelta, we need to load the model on GPU before creating the optimizer
        model = model.to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
    else:
        raise KeyError("{} model is unknown.".format(name))

    model = model.to(device)
    epoch = kwargs.setdefault("epoch", 100)
    kwargs.setdefault(
        "scheduler",
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=epoch // 4, verbose=True
        ),
    )
    # kwargs.setdefault('scheduler', None)
    kwargs.setdefault("batch_size", 100)
    kwargs.setdefault("supervision", "full")
    kwargs.setdefault("flip_augmentation", False)
    kwargs.setdefault("radiation_augmentation", False)
    kwargs.setdefault("mixture_augmentation", False)
    kwargs["center_pixel"] = center_pixel
    return model, optimizer, criterion, kwargs


class Baseline(nn.Module):
    """
    Baseline network
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, dropout=False):
        super(Baseline, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc4(x)
        return x


class HybridEtAl(nn.Module):
    """
    HybridSN: Exploring 3D-2D CNN Feature Hierarchy for Hyperspectral Image Classification
    Roy, Swalpa Kumar and Krishna, Gopal and Dubey, Shiv Ram and Chaudhuri, Bidyut B
    IEEE Geoscience and Remote Sensing Letters 2020
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=15):
        super(HybridEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        # the 8-neighbor pixels [...] are fed into the Conv1 convolved by n1 kernels
        # and s1 stride. Conv1 results are feature vectors each with height of and
        # the width is 1. After reshape layer, the feature vectors becomes an image-like
        # 2-dimension data.
        # Conv2 has 64 kernels size of 3x3, with stride s2.
        # After that, the 64 results are drawn into a vector as the input of the fully
        # connected layer FC1 which has n4 nodes.
        # In the four datasets, the kernel height nk1 is 24 and stride s1, s2 is 9 and 1
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, (7, 3, 3), padding=0, stride=(1, 1, 1)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, (5, 3, 3), padding=0, stride=(1, 1, 1)),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(16, 32, (3, 3, 3), padding=0, stride=(1, 1, 1)),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(6016, 64, (3, 3), padding=0, stride=(1, 1)),
            nn.ReLU()
        ) ## 6016 for IN, 576 for Grss ##2912 for PU3

        # self.features_size = self._get_final_flattened_size()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(3136, 256)
        self.dropout1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=0.4)
        self.classfier = nn.Linear(128, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        b, c, s, h, w = x.size()
        x = x.view(b, c*s, h, w)
        # print(x.shape)
        x = self.conv4(x)
        x = self.flatten(x)
        # print(x.shape)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.classfier(x)

        return x

class HuEtAl(nn.Module):
    """
    Deep Convolutional Neural Networks for Hyperspectral Image Classification
    Wei Hu, Yangyu Huang, Li Wei, Fan Zhang and Hengchao Li
    Journal of Sensors, Volume 2015 (2015)
    https://www.hindawi.com/journals/js/2015/258619/
    """

    @staticmethod
    def weight_init(m):
        # [All the trainable parameters in our CNN should be initialized to
        # be a random value between −0.05 and 0.05.]
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
        return x.numel()

    def __init__(self, input_channels, n_classes, kernel_size=None, pool_size=None):
        super(HuEtAl, self).__init__()
        if kernel_size is None:
            # [In our experiments, k1 is better to be [ceil](n1/9)]
            kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
            # The authors recommand that k2's value is chosen so that the pooled features have 30~40 values
            # ceil(kernel_size/5) gives the same values as in the paper so let's assume it's okay
            pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels

        # [The first hidden convolution layer C1 filters the n1 x 1 input data with 20 kernels of size k1 x 1]
        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        # [n4 is set to be 100]
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        # [In our design architecture, we choose the hyperbolic tangent function tanh(u)]
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.tanh(self.pool(x))
        x = x.view(-1, self.features_size)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class HamidaEtAl(nn.Module):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=5, dilation=1):
        super(HamidaEtAl, self).__init__()
        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=1
            )
        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0
            )
        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # equal to 2 in order to reduce the spectral dimension
        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Conv3d(
            20, 35, (3, 3, 3), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)
        )
        self.pool2 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )
        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)
        )
        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )

        # self.dropout = nn.Dropout(p=0.5)

        self.features_size = self._get_final_flattened_size()
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes.
        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        # x = self.dropout(x)
        x = self.fc(x)
        return x


class LeeEtAl(nn.Module):
    """
    CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon
    IGARSS 2016
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, in_channels, n_classes):
        super(LeeEtAl, self).__init__()
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)
        self.conv_3x3 = nn.Conv3d(
            1, 128, (in_channels, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )
        self.conv_1x1 = nn.Conv3d(
            1, 128, (in_channels, 1, 1), stride=(1, 1, 1), padding=0
        )

        # We use two modules from the residual learning approach
        # Residual block 1
        self.conv1 = nn.Conv2d(256, 128, (1, 1))
        self.conv2 = nn.Conv2d(128, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, (1, 1))

        # Residual block 2
        self.conv4 = nn.Conv2d(128, 128, (1, 1))
        self.conv5 = nn.Conv2d(128, 128, (1, 1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(128, 128, (1, 1))
        self.conv7 = nn.Conv2d(128, 128, (1, 1))
        self.conv8 = nn.Conv2d(128, n_classes, (1, 1))

        self.lrn1 = nn.LocalResponseNorm(256)
        self.lrn2 = nn.LocalResponseNorm(128)

        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def forward(self, x):
        # Inception module
        x_3x3 = self.conv_3x3(x)
        x_1x1 = self.conv_1x1(x)
        x = torch.cat([x_3x3, x_1x1], dim=1)
        # Remove the third dimension of the tensor
        x = torch.squeeze(x)

        # Local Response Normalization
        x = F.relu(self.lrn1(x))

        # First convolution
        x = self.conv1(x)

        # Local Response Normalization
        x = F.relu(self.lrn2(x))

        # First residual block
        x_res = F.relu(self.conv2(x))
        x_res = self.conv3(x_res)
        x = F.relu(x + x_res)

        # Second residual block
        x_res = F.relu(self.conv4(x))
        x_res = self.conv5(x_res)
        x = F.relu(x + x_res)

        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        x = self.conv8(x)
        return x


class ChenEtAl(nn.Module):
    """
    DEEP FEATURE EXTRACTION AND CLASSIFICATION OF HYPERSPECTRAL IMAGES BASED ON
                        CONVOLUTIONAL NEURAL NETWORKS
    Yushi Chen, Hanlu Jiang, Chunyang Li, Xiuping Jia and Pedram Ghamisi
    IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2017
    """

    @staticmethod
    def weight_init(m):
        # In the beginning, the weights are randomly initialized
        # with standard deviation 0.001
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.normal_(m.weight, std=0.001)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=27, n_planes=32):
        super(ChenEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, n_planes, (32, 4, 4))
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.conv3 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = x.view(-1, self.features_size)
        x = self.fc(x)
        return x


class LiEtAl(nn.Module):
    """
    SPECTRAL–SPATIAL CLASSIFICATION OF HYPERSPECTRAL IMAGERY
            WITH 3D CONVOLUTIONAL NEURAL NETWORK
    Ying Li, Haokui Zhang and Qiang Shen
    MDPI Remote Sensing, 2017
    http://www.mdpi.com/2072-4292/9/1/67
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0)

    def __init__(self, input_channels, n_classes, n_planes=2, patch_size=5):
        super(LiEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        # The proposed 3D-CNN model has two 3D convolution layers (C1 and C2)
        # and a fully-connected layer (F1)
        # we fix the spatial size of the 3D convolution kernels to 3 × 3
        # while only slightly varying the spectral depth of the kernels
        # for the Pavia University and Indian Pines scenes, those in C1 and C2
        # were set to seven and three, respectively
        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3), padding=(1, 0, 0))
        # the number of kernels in the second convolution layer is set to be
        # twice as many as that in the first convolution layer
        self.conv2 = nn.Conv3d(n_planes, 2 * n_planes, (3, 3, 3), padding=(1, 0, 0))
        # self.dropout = nn.Dropout(p=0.5)
        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            x = self.conv2(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        # x = self.dropout(x)
        x = self.fc(x)
        return x


class HeEtAl(nn.Module):
    """
    MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL
    IMAGE CLASSIFICATION
    Mingyi He, Bo Li, Huahui Chen
    IEEE International Conference on Image Processing (ICIP) 2017
    https://ieeexplore.ieee.org/document/8297014/
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=7):
        super(HeEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, 16, (11, 3, 3), stride=(3, 1, 1))
        self.conv2_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0))
        self.conv2_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.conv2_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.conv2_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))
        self.conv3_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0))
        self.conv3_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.conv3_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.conv3_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))
        self.conv4 = nn.Conv3d(16, 16, (3, 2, 2))
        self.pooling = nn.MaxPool2d((3, 2, 2), stride=(3, 2, 2))
        # the ratio of dropout is 0.6 in our experiments
        self.dropout = nn.Dropout(p=0.6)

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            x2_1 = self.conv2_1(x)
            x2_2 = self.conv2_2(x)
            x2_3 = self.conv2_3(x)
            x2_4 = self.conv2_4(x)
            x = x2_1 + x2_2 + x2_3 + x2_4
            x3_1 = self.conv3_1(x)
            x3_2 = self.conv3_2(x)
            x3_3 = self.conv3_3(x)
            x3_4 = self.conv3_4(x)
            x = x3_1 + x3_2 + x3_3 + x3_4
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)
        x = x2_1 + x2_2 + x2_3 + x2_4
        x = F.relu(x)
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.conv3_3(x)
        x3_4 = self.conv3_4(x)
        x = x3_1 + x3_2 + x3_3 + x3_4
        x = F.relu(x)
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class LuoEtAl(nn.Module):
    """
    HSI-CNN: A Novel Convolution Neural Network for Hyperspectral Image
    Yanan Luo, Jie Zou, Chengfei Yao, Tao Li, Gang Bai
    International Conference on Pattern Recognition 2018
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=3, n_planes=90):
        super(LuoEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.n_planes = n_planes

        # the 8-neighbor pixels [...] are fed into the Conv1 convolved by n1 kernels
        # and s1 stride. Conv1 results are feature vectors each with height of and
        # the width is 1. After reshape layer, the feature vectors becomes an image-like
        # 2-dimension data.
        # Conv2 has 64 kernels size of 3x3, with stride s2.
        # After that, the 64 results are drawn into a vector as the input of the fully
        # connected layer FC1 which has n4 nodes.
        # In the four datasets, the kernel height nk1 is 24 and stride s1, s2 is 9 and 1
        self.conv1 = nn.Conv3d(1, 90, (24, 3, 3), padding=0, stride=(9, 1, 1))
        self.conv2 = nn.Conv2d(1, 64, (3, 3), stride=(1, 1))

        self.features_size = self._get_final_flattened_size()

        self.fc1 = nn.Linear(self.features_size, 1024)
        self.fc2 = nn.Linear(1024, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            b = x.size(0)
            x = x.view(b, 1, -1, self.n_planes)
            x = self.conv2(x)
            _, c, w, h = x.size()
        return c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        b = x.size(0)
        x = x.view(b, 1, -1, self.n_planes)
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SharmaEtAl(nn.Module):
    """
    HYPERSPECTRAL CNN FOR IMAGE CLASSIFICATION & BAND SELECTION, WITH APPLICATION
    TO FACE RECOGNITION
    Vivek Sharma, Ali Diba, Tinne Tuytelaars, Luc Van Gool
    Technical Report, KU Leuven/ETH Zürich
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=64):
        super(SharmaEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        # An input image of size 263x263 pixels is fed to conv1
        # with 96 kernels of size 6x6x96 with a stride of 2 pixels
        self.conv1 = nn.Conv3d(1, 96, (input_channels, 6, 6), stride=(1, 2, 2))
        self.conv1_bn = nn.BatchNorm3d(96)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        #  256 kernels of size 3x3x256 with a stride of 2 pixels
        self.conv2 = nn.Conv3d(1, 256, (96, 3, 3), stride=(1, 2, 2))
        self.conv2_bn = nn.BatchNorm3d(256)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        # 512 kernels of size 3x3x512 with a stride of 1 pixel
        self.conv3 = nn.Conv3d(1, 512, (256, 3, 3), stride=(1, 1, 1))
        # Considering those large kernel values, I assume they actually merge the
        # 3D tensors at each step

        self.features_size = self._get_final_flattened_size()

        # The fc1 has 1024 outputs, where dropout was applied after
        # fc1 with a rate of 0.5
        self.fc1 = nn.Linear(self.features_size, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = F.relu(self.conv1_bn(self.conv1(x)))
            x = self.pool1(x)
            print(x.size())
            b, t, c, w, h = x.size()
            x = x.view(b, 1, t * c, w, h)
            x = F.relu(self.conv2_bn(self.conv2(x)))
            x = self.pool2(x)
            print(x.size())
            b, t, c, w, h = x.size()
            x = x.view(b, 1, t * c, w, h)
            x = F.relu(self.conv3(x))
            print(x.size())
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool1(x)
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t * c, w, h)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool2(x)
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t * c, w, h)
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.features_size)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LiuEtAl(nn.Module):
    """
    A semi-supervised convolutional neural network for hyperspectral image classification
    Bing Liu, Xuchu Yu, Pengqiang Zhang, Xiong Tan, Anzhu Yu, Zhixiang Xue
    Remote Sensing Letters, 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=9):
        super(LiuEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.aux_loss_weight = 1

        # "W1 is a 3x3xB1 kernel [...] B1 is the number of the output bands for the convolutional
        # "and pooling layer" -> actually 3x3 2D convolutions with B1 outputs
        # "the value of B1 is set to be 80"
        self.conv1 = nn.Conv2d(input_channels, 80, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv1_bn = nn.BatchNorm2d(80)

        self.features_sizes = self._get_sizes()

        self.fc_enc = nn.Linear(self.features_sizes[2], n_classes)

        # Decoder
        self.fc1_dec = nn.Linear(self.features_sizes[2], self.features_sizes[2])
        self.fc1_dec_bn = nn.BatchNorm1d(self.features_sizes[2])
        self.fc2_dec = nn.Linear(self.features_sizes[2], self.features_sizes[1])
        self.fc2_dec_bn = nn.BatchNorm1d(self.features_sizes[1])
        self.fc3_dec = nn.Linear(self.features_sizes[1], self.features_sizes[0])
        self.fc3_dec_bn = nn.BatchNorm1d(self.features_sizes[0])
        self.fc4_dec = nn.Linear(self.features_sizes[0], input_channels)

        self.apply(self.weight_init)

    def _get_sizes(self):
        x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size))
        x = F.relu(self.conv1_bn(self.conv1(x)))
        _, c, w, h = x.size()
        size0 = c * w * h

        x = self.pool1(x)
        _, c, w, h = x.size()
        size1 = c * w * h

        x = self.conv1_bn(x)
        _, c, w, h = x.size()
        size2 = c * w * h

        return size0, size1, size2

    def forward(self, x):
        x = x.squeeze()
        x_conv1 = self.conv1_bn(self.conv1(x))
        x = x_conv1
        x_pool1 = self.pool1(x)
        x = x_pool1
        x_enc = F.relu(x).view(-1, self.features_sizes[2])
        x = x_enc

        x_classif = self.fc_enc(x)

        # x = F.relu(self.fc1_dec_bn(self.fc1_dec(x) + x_enc))
        x = F.relu(self.fc1_dec(x))
        x = F.relu(
            self.fc2_dec_bn(self.fc2_dec(x) + x_pool1.view(-1, self.features_sizes[1]))
        )
        x = F.relu(
            self.fc3_dec_bn(self.fc3_dec(x) + x_conv1.view(-1, self.features_sizes[0]))
        )
        x = self.fc4_dec(x)
        return x_classif, x


class BoulchEtAl(nn.Module):
    """
    Autoencodeurs pour la visualisation d'images hyperspectrales
    A.Boulch, N. Audebert, D. Dubucq
    GRETSI 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, planes=15):
        super(BoulchEtAl, self).__init__()
        self.input_channels = input_channels
        self.aux_loss_weight = 0.1

        encoder_modules = []
        n = input_channels
        with torch.no_grad():
            x = torch.zeros((10, 1, self.input_channels))
            print(x.size())
            while n > 1:
                print("---------- {} ---------".format(n))
                if n == input_channels:
                    p1, p2 = 1, 2 * planes
                elif n == input_channels // 2:
                    p1, p2 = 2 * planes, planes
                else:
                    p1, p2 = planes, planes
                encoder_modules.append(nn.Conv1d(p1, p2, 3, padding=1))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.MaxPool1d(2))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.ReLU(inplace=True))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.BatchNorm1d(p2))
                x = encoder_modules[-1](x)
                print(x.size())
                n = n // 2

            encoder_modules.append(nn.Conv1d(planes, 3, 3, padding=1))
        encoder_modules.append(nn.Tanh())
        self.encoder = nn.Sequential(*encoder_modules)
        self.features_sizes = self._get_sizes()

        self.classifier = nn.Linear(self.features_sizes, n_classes)
        self.regressor = nn.Linear(self.features_sizes, input_channels)
        self.apply(self.weight_init)

    def _get_sizes(self):
        with torch.no_grad():
            x = torch.zeros((10, 1, self.input_channels))
            x = self.encoder(x)
            _, c, w = x.size()
        return c * w

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.view(-1, self.features_sizes)
        x_classif = self.classifier(x)
        x = self.regressor(x)
        return x_classif, x


class MouEtAl(nn.Module):
    """
    Deep recurrent neural networks for hyperspectral image classification
    Lichao Mou, Pedram Ghamisi, Xiao Xang Zhu
    https://ieeexplore.ieee.org/document/7914752/
    """

    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with a uniform distribution, and the values of these weight matrices and bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU)):
            init.uniform_(m.weight.data, -0.1, 0.1)
            init.uniform_(m.bias.data, -0.1, 0.1)

    def __init__(self, input_channels, n_classes):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(MouEtAl, self).__init__()
        self.input_channels = input_channels
        self.gru = nn.GRU(1, 64, 1, bidirectional=False)  # TODO: try to change this ?
        self.gru_bn = nn.BatchNorm1d(64 * input_channels)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(64 * input_channels, n_classes)

    def forward(self, x):
        x = x.squeeze()
        x = x.unsqueeze(0)
        # x is in 1, N, C but we expect C, N, 1 for GRU layer
        x = x.permute(2, 1, 0)
        x = self.gru(x)[0]
        # x is in C, N, 64, we permute back
        x = x.permute(1, 2, 0).contiguous()
        x = x.view(x.size(0), -1)
        x = self.gru_bn(x)
        x = self.tanh(x)
        x = self.fc(x)
        return x


def train(
    net,
    optimizer,
    criterion,
    data_loader,
    epoch,
    scheduler=None,
    display_iter=100,
    device=torch.device("cpu"),
    display=None,
    val_loader=None,
    supervision="full",
):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """

    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)

    save_epoch = epoch // 20 if epoch > 20 else 1

    losses = np.zeros(100000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    gamma = 0.7
    scheduler = StepLR(optimizer, step_size=10, gamma=gamma)
    loss_win, val_win = None, None
    val_accuracies = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        # Set the network to training mode
        net.train()
        avg_loss = 0.0

        # Run the training loop for one epoch
        for batch_idx, (data, target) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            # Load the data into the GPU if required
            # print(target)
            data, target = data.to(device), target.to(device)
            # print(target.shape)

            optimizer.zero_grad()
            if supervision == "full":
                output = net(data)
                # print(output.shape)
                loss = criterion(output[0], target) + criterion(output[1], target)
            elif supervision == "semi":
                outs = net(data)
                output, rec = outs
                loss = criterion[0](output, target) + net.aux_loss_weight * criterion[
                    1
                ](rec, data)
            elif supervision == "con":
                output, outs = net(data)
                loss1 = criterion(output, target)
                loss2 = criterion(outs, target)
                loss = loss1 + loss2
            else:
                raise ValueError(
                    'supervision mode "{}" is unknown.'.format(supervision)
                )
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100) : iter_ + 1])

            if display_iter and iter_ % display_iter == 0:
                string = "Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}"
                string = string.format(
                    e,
                    epoch,
                    batch_idx * len(data),
                    len(data) * len(data_loader),
                    100.0 * batch_idx / len(data_loader),
                    mean_losses[iter_],
                )
                update = None if loss_win is None else "append"
                loss_win = display.line(
                    X=np.arange(iter_ - display_iter, iter_),
                    Y=mean_losses[iter_ - display_iter : iter_],
                    win=loss_win,
                    update=update,
                    opts={
                        "title": "Training loss",
                        "xlabel": "Iterations",
                        "ylabel": "Loss",
                    },
                )
                tqdm.write(string)

                if len(val_accuracies) > 0:
                    val_win = display.line(
                        Y=np.array(val_accuracies),
                        X=np.arange(len(val_accuracies)),
                        win=val_win,
                        opts={
                            "title": "Validation accuracy",
                            "xlabel": "Epochs",
                            "ylabel": "Accuracy",
                        },
                    )
            iter_ += 1
            del (data, target, loss, output)
            # end.record()
            # torch.cuda.synchronize()
            # print("The training time is:***********************", start.elapsed_time(end))

        # Update the scheduler
        avg_loss /= len(data_loader)
        if val_loader is not None:
            val_acc = val(net, val_loader, device=device, supervision=supervision)
            # val_acc = test(net, val_loader, hyperparams=hyperparams)
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        # Save the weights
        if e % save_epoch == 0:
            save_model(
                net,
                camel_to_snake(str(net.__class__.__name__)),
                data_loader.dataset.name,
                epoch=e,
                metric=abs(metric),
            )
    end.record()
    torch.cuda.synchronize()
    print("The training time is:***********************", start.elapsed_time(end)/1000)

def t_SNE(
    net,
    data_loader,
    device=torch.device(0),
    num_class = 15
):
    net.to(device)
    features = []
    labels = []
    net.eval()
    print("Start Catch the features and labels")
    for batch_idx, (data, target) in tqdm(
        enumerate(data_loader), total=len(data_loader)
    ):
        data, target = data.to(device), target.to(device)
        activation = {}
        def hook_fn(net, data, output):
            activation['fc_1_output'] = output
        target_layer = net.to_latent#ViT
        # target_layer = net.to_cls_token#SSFTT
        # target_layer = net.classifier  # DCTN,HiT,SS_TMNet
        # target_layer = net.feature
        # target_layer = net.classifier#cnn2D
        # target_layer = net.out_finetune ## cnn3d
        # target_layer = net.ca#morphformer
        # target_layer = net.fc2#hybridsn
        hook = target_layer.register_forward_hook(hook_fn)
        # _, _, output = net(data)
        output = net(data)
        # if isinstance(output, tuple):
        #     output = output[0]
        fc_1_output = activation['fc_1_output']
        # fc_1_output = fc_1_output.mean(1)  # DCTN HiT SS-TMNet
        # fc_1_output = fc_1_output.reshape(fc_1_output.shape[0],-1)#morphformer
        # print(fc_1_output.shape)
        features.append(fc_1_output.cpu().detach().numpy())
        labels.append(target.cpu().detach().numpy())
        ##这里设置钩子函数（data，traget）可以捕获到标签值
##t-SNE可视化
    print("End Catch the features and labels")
    features = np.vstack(features)
    labels = np.concatenate(labels)
    print("features", features.shape)
    print("labels", labels.shape)
    # 使用 PCA 拟合并转换数据
    # pca = PCA(n_components=64)  # 降维到 50 维
    # reduced_data = pca.fit_transform(features)
    # print("样本_PCA:", reduced_data.shape)

    # num_clusters = num_class#change  #Indian:16 PaviaU:9 Houston:15
    palette = {}  # {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", num_class)):
        palette[k] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
    custom_cmap = ListedColormap(np.array(list(palette.values())) / 255.0)
    embeddings = TSNE(n_components=2).fit_transform(#perplexity=400, learning_rate=50, n_jobs=8
        features)  # init='pca', random_state=0,
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]

    # plt.figure(figsize=(5, 5), dpi=100)
    # 使用聚类结果为数据点着色
    plt.scatter(vis_x, vis_y, c=labels, cmap=custom_cmap, marker='.')
    plt.colorbar(ticks=range(1, num_class + 1))
    plt.clim(0.5, num_class + 0.5)
    # 获取x轴和y轴的最大和最小值
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    # 设置x轴的范围，确保最大值不超过100且最小值不低于-100
    if x_max < 100:
        x_max = 100
    if x_min > -100:
        x_min = -100

    # 设置y轴的范围，确保最大值不超过100且最小值不低于-100
    if y_max < 100:
        y_max = 100
    if y_min > -100:
        y_min = -100

    # 设置x轴和y轴的坐标范围
    plt.xlim(x_min, x_max)  # 设置x轴坐标范围
    plt.ylim(y_min, y_max)
    # plt.xlim(-100, 100)  # 设置x轴坐标范围为(-100, 100)
    # plt.ylim(-100, 100)  # 设置y轴坐标范围为(-100, 100)
    plt.xticks(fontsize=12)  # 设置x轴刻度标签的字体大小为12
    plt.yticks(fontsize=12)  # 设置y轴刻度标签的字体大小为12
    # plt.axis('off')
    # fig.axes.get_xaxis().set_visible(False)
    # fig.axes.get_yaxis().set_visible(False)
    plt.savefig('E:/UM/BF/DeepHyperX-Transformer/t_sne/IndianPines/act_tsne', dpi=600)
    plt.show()
    exit()

def train3(#Backup t-sne
    net,
    optimizer,
    criterion,
    data_loader,
    epoch,
    scheduler=None,
    display_iter=100,
    device=torch.device("cpu"),
    display=None,
    val_loader=None,
    supervision="full",
):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """

    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)

    save_epoch = epoch // 1 if epoch > 20 else 1

    losses = np.zeros(100000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    gamma = 0.7
    scheduler = StepLR(optimizer, step_size=10, gamma=gamma)
    loss_win, val_win = None, None
    val_accuracies = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    features = []
    labels = []
    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        # Set the network to training mode
        net.train()
        avg_loss = 0.0

        # Run the training loop for one epoch
        for batch_idx, (data, target) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            # print()
            # print("*****",batch_idx,"*****")
            # print()
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            if supervision == "full":
                if e==epoch:# and batch_idx!=123:
                    activation = {}
                    def hook_fn(net, data, output):
                        activation['fc_1_output'] = output
                    target_layer = net.norm  # DCTN
                    # target_layer = net.fc_1#cnn2D
                    hook = target_layer.register_forward_hook(hook_fn)
                    output = net(data)
                    fc_1_output = activation['fc_1_output']
                    fc_1_output = fc_1_output.mean(1)  # DCTN
                    features.append(fc_1_output.cpu().detach().numpy())
                    labels.append(target.cpu().detach().numpy())
                    ##这里设置钩子函数（data，traget）可以捕获到标签值
                else:
                    output = net(data)
                loss = criterion(output, target)
            elif supervision == "semi":
                outs = net(data)
                output, rec = outs
                loss = criterion[0](output, target) + net.aux_loss_weight * criterion[
                    1
                ](rec, data)
            else:
                raise ValueError(
                    'supervision mode "{}" is unknown.'.format(supervision)
                )
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100) : iter_ + 1])

            if display_iter and iter_ % display_iter == 0:
                string = "Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.8f}"#{:.6f}
                string = string.format(
                    e,
                    epoch,
                    batch_idx * len(data),
                    len(data) * len(data_loader),
                    100.0 * batch_idx / len(data_loader),
                    mean_losses[iter_],
                )
                update = None if loss_win is None else "append"
                loss_win = display.line(
                    X=np.arange(iter_ - display_iter, iter_),
                    Y=mean_losses[iter_ - display_iter : iter_],
                    win=loss_win,
                    update=update,
                    opts={
                        "title": "Training loss",
                        "xlabel": "Iterations",
                        "ylabel": "Loss",
                    },
                )
                tqdm.write(string)

                if len(val_accuracies) > 0:
                    val_win = display.line(
                        Y=np.array(val_accuracies),
                        X=np.arange(len(val_accuracies)),
                        win=val_win,
                        opts={
                            "title": "Validation accuracy",
                            "xlabel": "Epochs",
                            "ylabel": "Accuracy",
                        },
                    )
            iter_ += 1
            del (data, target, loss, output)

        # Update the scheduler
        avg_loss /= len(data_loader)
        if val_loader is not None:
            val_acc = val(net, val_loader, device=device, supervision=supervision)
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)

        elif scheduler is not None:
            scheduler.step()

        # Save the weights
        # if e % save_epoch == 0:
        #     save_model(
        #         net,
        #         camel_to_snake(str(net.__class__.__name__)),
        #         data_loader.dataset.name,
        #         epoch=e,
        #         metric=abs(metric),
        #     )
    end.record()
    torch.cuda.synchronize()
    print("The training time is:***********************", start.elapsed_time(end) / 1000)
    ##t-SNE可视化
    features = np.vstack(features)
    labels = np.concatenate(labels)
    print("features", features.shape)
    print("labels", labels.shape)
    pca = PCA(n_components=64)  # 降维到 50 维
    # 使用 PCA 拟合并转换数据
    reduced_data = pca.fit_transform(features)
    print("样本_PCA:", reduced_data.shape)
    num_clusters = 16 - 1 #change
    cmap = plt.cm.get_cmap("tab20", num_clusters)
    embeddings = TSNE(n_components=2, perplexity=400, learning_rate=50, n_jobs=8).fit_transform(
        reduced_data)  # init='pca', random_state=0,
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]

    plt.figure(figsize=(5, 5), dpi=100)
    # 使用聚类结果为数据点着色
    fig = plt.scatter(vis_x, vis_y, c=labels, cmap=cmap, marker='.')
    plt.colorbar(ticks=range(1, num_clusters+1))
    plt.clim(0.5, num_clusters+0.5)  # 设置颜色映射的范围
    # plt.axis('off')
    # fig.axes.get_xaxis().set_visible(False)
    # fig.axes.get_yaxis().set_visible(False)
    plt.savefig('t_sne', dpi=100)
    plt.show()

def save_model(model, model_name, dataset_name, **kwargs):
    model_dir = "./checkpoints/" + model_name + "/" + dataset_name + "/"
    """
    Using strftime in case it triggers exceptions on windows 10 system
    """
    time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        filename = time_str + "_epoch{epoch}_{metric:.2f}".format(
            **kwargs
        )
        tqdm.write("Saving neural network weights in {}".format(filename))
        torch.save(model.state_dict(), model_dir + filename + ".pth")
    else:
        filename = time_str
        tqdm.write("Saving model params in {}".format(filename))
        joblib.dump(model, model_dir + filename + ".pkl")


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    newX = np.asarray(np.copy(newX), dtype="float32")
    return newX, pca

def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams["patch_size"]
    center_pixel = hyperparams["center_pixel"]
    batch_size, device = hyperparams["batch_size"], hyperparams["device"]
    n_classes = hyperparams["n_classes"]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    kwargs = {
        "step": hyperparams["test_stride"],
        "window_size": (patch_size, patch_size),
    }
    # print(probs.shape)
    # img = padding_image(img, patch_size=[15, 15], constant_values=0)
    probs = np.zeros(img.shape[:2] + (n_classes,))
    # img, pca = applyPCA(img, numComponents=4)
    # img = np.asarray(np.copy(img), dtype="float32")

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(
        grouper(batch_size, sliding_window(img, **kwargs)),
        total=(iterations),
        desc="Inference on the image",
    ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                #b, d, h, w = data.shape
                #data_z = np.zeros([b, 256, h, w])
                # data = data[:, 0:4, :, :]
                #
                # data = np.asarray(np.copy(data), dtype="float32")
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            out = net(data)
            output = out[1]
            if isinstance(output, tuple):
                output = output[0]
            output = output.to("cpu")
            # print(output.shape)
            # out = output[:, :, 16 // 2, 16 // 2]
            # print(out.shape)

            if patch_size == 1 or center_pixel:
                output = output.numpy()
                # out = out.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
                # out = np.transpose(out.numpy(), (0, 2, 3, 1))
            # for (x, y, w, h), out in zip(indices, output):
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                   # probs[x + w // 2, y + h // 2] += out
                   #  probs[x, y] += out
                   #  out = out.transpose((1, 2, 0))
                    probs[x: x + w, y: y + h] += out
                else:
                    probs[x : x + w, y : y + h] += out
    end.record()
    torch.cuda.synchronize()
    print("The testing time is:***********************", start.elapsed_time(end)/1000)
    return probs


def val(net, data_loader, device="cpu", supervision="full"):
    # TODO : fix me using metrics()
    accuracy, total = 0.0, 0.0
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            if supervision == "full":
                output = net(data)
            elif supervision == "semi":
                outs = net(data)
                output, rec = outs
            elif supervision == "con":
                output, outs = net(data)
                # output = (outputs + outputs) /2
            _, output = torch.max(output[1], dim=1)
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    return accuracy / total
