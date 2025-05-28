# -*- coding: utf-8 -*-
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
dctn = cv2.imread('G:/submitting paper/ACTN_files/ACT_results/WHL/visdom_image.jpg')
# input = cv2.imread('G:/submitting paper/ACTN_files/ACT_results/WHL/longkou/input.png')
# sstmnet = cv2.imread('E:/DWT/results_visual/xa/svm/visdom_image.png')
# mou = cv2.imread('E:/DWT/results_visual/xa/mou/visdom_image.png')
# knn = cv2.imread('E:/DWT/results_visual/xa/knn/visdom_image.png')
# rnn = cv2.imread('E:/DWT/results_visual/xa/knn/visdom_image.png')
# vit = cv2.imread('E:/DWT/results_visual/xa/vit/visdom_image.png')
# dvit = cv2.imread('E:/DWT/results_visual/xa/dit/visdom_image.png')
# cait = cv2.imread('E:/DWT/results_visual/xa/cait/visdom_image.png')
# cvt = cv2.imread('E:/DWT/results_visual/xa/cvt/visdom_image.png')
# ours = cv2.imread('E:/DWT/results_visual/xa/dwt/visdom_image.png')

# gt = cnn_2d[:,3756:7508]

# h_svm = svm[:,0:3756]
# h_mou = mou[:,0:3756]
# h_knn = knn[:,0:3756]
# h_rnn = rnn[:,0:3756]
# h_vit = vit[:,0:3756]
# h_dvit = dvit[:,0:3756]
# h_cait = cait[:,0:3756]
# h_cvt = cvt[:,0:3756]
# h_ours = ours[:,0:3756]
# print(input.shape)
h_dctn = dctn[:,0:403]

cv2.imwrite("G:/submitting paper/ACTN_files/ACT_results/WHL/longkou/dctn.png", h_dctn)
# cv2.imwrite("E:/DWT/figures/xa/2d.png", cnn_2d[:,0:3756])
# cv2.imwrite("E:/DWT/figures/xa/3d.png", cnn_3d[:,0:3756])
# cv2.imwrite("E:/DWT/figures/xa/svm.png", h_svm)
# cv2.imwrite("E:/DWT/figures/xa/mou.png", h_mou)
# cv2.imwrite("E:/DWT/figures/xa/knn.png", h_knn)
# cv2.imwrite("E:/DWT/figures/xa/rnn.png", h_rnn)
# cv2.imwrite("E:/DWT/figures/xa/vit.png", h_vit)
# cv2.imwrite("E:/DWT/figures/xa/dit.png", h_dvit)
# cv2.imwrite("E:/DWT/figures/xa/cait.png", h_cait)
# cv2.imwrite("E:/DWT/figures/xa/cvt.png", h_cvt)
# cv2.imwrite("E:/DWT/figures/xa/ours.png", h_ours)


