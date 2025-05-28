# -*- codpug: utf-8 -*-
import matplotlib.pyplot as plt
import cv2
input_image = cv2.imread('E:/transformer/pu/input.jpg')
ground_truth = cv2.imread('E:/transformer/pu/gt.png')
cnn_2d = cv2.imread('E:/transformer/pu/2d.png')
cnn_3d = cv2.imread('E:/transformer/pu/3d.png')
he = cv2.imread('E:/transformer/pu/he.png')
mou = cv2.imread('E:/transformer/pu/mou.png')
boulch = cv2.imread('E:/transformer/pu/bou.png')
rnn = cv2.imread('E:/transformer/pu/rcnn.png')
vit = cv2.imread('E:/transformer/pu/vit.png')
dvit = cv2.imread('E:/transformer/pu/dit.png')
t2t = cv2.imread('E:/transformer/pu/t2t.png')
lvt = cv2.imread('E:/transformer/pu/lvt.png')
rvt = cv2.imread('E:/transformer/pu/rvt.png')
ours = cv2.imread('E:/transformer/pu/hit.png')

h, w = cnn_2d.shape[0], cnn_2d.shape[1]
print(cnn_2d.shape)
h_x = 9
h_y = h-9
w_x = 9
w_y = w - 9


puput = input_image[h_x:h_y, w_x:w_y]
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

cv2.imwrite("E:/transformer/result_update/pu/puput.png", puput)
cv2.imwrite("E:/transformer/result_update/pu/gt.png", gt)
cv2.imwrite("E:/transformer/result_update/pu/2d.png", cnn2d)
cv2.imwrite("E:/transformer/result_update/pu/3d.png", cnn3d)
cv2.imwrite("E:/transformer/result_update/pu/mou.png", h_mou)
cv2.imwrite("E:/transformer/result_update/pu/bou.png", h_bou)
cv2.imwrite("E:/transformer/result_update/pu/rcnn.png", h_rcnn)
cv2.imwrite("E:/transformer/result_update/pu/vit.png", h_vit)
cv2.imwrite("E:/transformer/result_update/pu/dit.png", h_dvit)
cv2.imwrite("E:/transformer/result_update/pu/t2t.png", h_t2t)
cv2.imwrite("E:/transformer/result_update/pu/lvt.png", h_lvt)
cv2.imwrite("E:/transformer/result_update/pu/rvt.png", h_rvt)
cv2.imwrite("E:/transformer/result_update/pu/hit.png", h_ours)

