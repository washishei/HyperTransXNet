# -*- coding: utf-8 -*-
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
img = cv2.imread('D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss/input.jpg')
gt = cv2.imread('D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss/gt.png')
cnn_2d = cv2.imread('D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss/2d.png')
cnn_3d = cv2.imread('D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss/3d.png')
mou = cv2.imread('D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss/mou.png')
rnn = cv2.imread('D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss/rcnn.png')
he = cv2.imread('D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss/he.png')
vit = cv2.imread('D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss/vit.png')
dit = cv2.imread('D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss/dit.png')
t2t = cv2.imread('D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss/t2t.png')
levit = cv2.imread('D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss/lvt.png')
rvt = cv2.imread('D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss/rvt.png')
hit = cv2.imread('D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss/ours.png')
bou = cv2.imread('D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss/bou.png')
# cnn_3d = cv2.imread('E:/DWT/results_visual/grss/conv3d/visdom_image.png')
# svm = cv2.imread('E:/DWT/results_visual/grss/svm/visdom_image.png')
# mou = cv2.imread('E:/DWT/results_visual/grss/mou/visdom_image.png')
# knn = cv2.imread('E:/DWT/results_visual/grss/knn/visdom_image.png')
# rnn = cv2.imread('E:/DWT/results_visual/grss/knn/visdom_image.png')
# vit = cv2.imread('E:/DWT/results_visual/grss/vit/visdom_image.png')
# dvit = cv2.imread('E:/DWT/results_visual/grss/dit/visdom_image.png')
# cait = cv2.imread('E:/DWT/results_visual/grss/cait/visdom_image.png')
# cvt = cv2.imread('E:/DWT/results_visual/grss/cvt/visdom_image.png')
# ours = cv2.imread('E:/DWT/results_visual/grss/dwt/visdom_image.png')

a1 = img.shape[0]//2
b1 = img.shape[1]//2
a = a1- 128
b = a1 +128
c = b1 -128
d = b1 +128
gt = gt[a:b, c:d]
h_he = he[a:b, c:d]
h_mou = mou[a:b, c:d]
h_rvt = rvt[a:b, c:d]
h_rnn = rnn[a:b, c:d]
h_vit = vit[a:b, c:d]
h_dit = dit[a:b, c:d]
h_t2t = t2t[a:b, c:d]
h_lvt = levit[a:b, c:d]
h_ours = hit[a:b, c:d]
h_bou = bou[a:b, c:d]
input = img[a:b, c:d]


cv2.imwrite("D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss-new/input.png", input)
cv2.imwrite("D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss-new/gt.png", gt)
cv2.imwrite("D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss-new/2d.png", cnn_2d[a:b, c:d])
cv2.imwrite("D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss-new/3d.png", cnn_3d[a:b, c:d])
cv2.imwrite("D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss-new/he.png", h_he)
cv2.imwrite("D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss-new/mou.png", h_mou)
cv2.imwrite("D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss-new/rvt.png", h_rvt)
cv2.imwrite("D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss-new/rcnn.png", h_rnn)
cv2.imwrite("D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss-new/vit.png", h_vit)
cv2.imwrite("D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss-new/dit.png", h_dit)
cv2.imwrite("D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss-new/t2t.png", h_t2t)
cv2.imwrite("D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss-new/lvt.png", h_lvt)
cv2.imwrite("D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss-new/bou.png", h_bou)
cv2.imwrite("D:/UM/understanding transformer in hyperspectral image classification/figures_new/grss-new/hit.png", h_ours)
