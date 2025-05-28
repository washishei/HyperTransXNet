# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import cv2
input_image = cv2.imread('E:/transformer/in/input.jpg')
ground_truth = cv2.imread('E:/transformer/in/gt.png')
cnn_2d = cv2.imread('E:/transformer/in/2d.png')
cnn_3d = cv2.imread('E:/transformer/in/3d.png')
he = cv2.imread('E:/transformer/in/he.png')
mou = cv2.imread('E:/transformer/in/mou.png')
boulch = cv2.imread('E:/transformer/in/bou.png')
rnn = cv2.imread('E:/transformer/in/rcnn.png')
vit = cv2.imread('E:/transformer/in/vit.png')
dvit = cv2.imread('E:/transformer/in/dit.png')
t2t = cv2.imread('E:/transformer/in/t2t.png')
lvt = cv2.imread('E:/transformer/in/lvt.png')
rvt = cv2.imread('E:/transformer/in/rvt.png')
ours = cv2.imread('E:/transformer/in/hit.png')

h, w = cnn_2d.shape[0], cnn_2d.shape[1]
print(cnn_2d.shape)
h_x = 9
h_y = h-9
w_x = 8
w_y = w - 7
print(ground_truth.shape)

input = input_image[h_x:h_y, w_x:w_y]
gt = ground_truth[h_x:h_y, w_x+1:(w_y-1)]
h_mou = mou[h_x:h_y, w_x:w_y]
h_bou = boulch[h_x:h_y, w_x:w_y]
cnn2d = cnn_2d[h_x:h_y, w_x:w_y]
h_rcnn = rnn[h_x:h_y, w_x:w_y]
cnn3d = cnn_3d[h_x:h_y, w_x:w_y]
h_he = he[h_x:h_y, w_x:w_y]
h_vit = vit[h_x:h_y, w_x:w_y]
h_dvit = dvit[h_x:h_y, w_x:w_y]
h_t2t = t2t[h_x:h_y, w_x:w_y]
h_lvt = lvt[h_x:h_y, w_x:w_y]
h_rvt = rvt[h_x:h_y, w_x:w_y]
h_ours = ours[h_x:h_y, w_x:w_y]

cv2.imwrite("E:/transformer/result_update/in/input.png", input)
cv2.imwrite("E:/transformer/result_update/in/gt.png", gt)
cv2.imwrite("E:/transformer/result_update/in/2d.png", cnn2d)
cv2.imwrite("E:/transformer/result_update/in/3d.png", cnn3d)
cv2.imwrite("E:/transformer/result_update/in/mou.png", h_mou)
cv2.imwrite("E:/transformer/result_update/in/bou.png", h_bou)
cv2.imwrite("E:/transformer/result_update/in/rcnn.png", h_rcnn)
cv2.imwrite("E:/transformer/result_update/in/vit.png", h_vit)
cv2.imwrite("E:/transformer/result_update/in/dit.png", h_dvit)
cv2.imwrite("E:/transformer/result_update/in/t2t.png", h_t2t)
cv2.imwrite("E:/transformer/result_update/in/lvt.png", h_lvt)
cv2.imwrite("E:/transformer/result_update/in/rvt.png", h_rvt)
cv2.imwrite("E:/transformer/result_update/in/hit.png", h_ours)

