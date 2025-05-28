import matplotlib.pyplot as plt
import numpy as np

x = [9, 11, 13, 15, 17, 19, 21]
# y = [83.08,
# 88.48,
# 89.86,
# 90.52,
# 90.76]
#
# y1 = [91.71,
# 92.12,
# 92.20,
# 92.26,
# 92.31]
# y2 = [97.53,
# 98.83,
# 99.12,
# 99.32,
# 99.52]
# y3 = [96.36,
# 97.97,
# 98.57,
# 98.58,
# 98.68]
y=[70.16, 71.18,	71.8,	72.13,	72.36,	73.01,	73.84]
y1=[86.9,	87.57,	87.89,	88.24, 88.82,	89.19, 90.69]
y2=[78.24,	79.09,	81.9,	82.37,	82.8,	84.03,	84.98]
y3=[87.78,	87.99,	88.44,	88.67,	89.56,	90.65,	91.28]
plt.xlabel("The scale size of the cropped image")
plt.ylabel("OA (%)")
plt.plot(x,y, label='Indian Pines', marker='o' )
plt.plot(x,y1, label='PaviaU', marker='s',)
plt.plot(x,y2, label='Houston2013',marker='^', )
plt.plot(x,y3, label='Xiongan', marker='v', )
plt.legend()
# plt.savefig('E:/SSL/adassl.png')
plt.show()


# conf=(29.354, 98.03)
# vit=(13.001, 91.61)
# dit=(27.661, 90.08)
# cvt=(111.726, 97.26)
# lvt=(21.234, 93.00)
# swin=(18.044,95.98)
# focal=(74.401,87.42)
# cnn2=(9.235, 95.50)
# cnn3=(15.396, 91.68)
# sycnn=(69.374, 95.65)
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# x = [29.354, 13.001, 27.661, 111.726, 21.234, 18.044, 74.401, 9.235, 15.396,69.374]
# y = [98.03, 91.61, 90.08, 97.26, 93.00, 95.98, 87.42, 95.50, 91.68, 95.65]
# n = np.arange(10)
#
# fig, ax = plt.subplots()
# ax.scatter(x, y, c='r', s=150)
# n=["Ours", "ViT", "Deep ViT", "CvT", "LeViT", "Swin Transformer", "Focal Transformer", "2D-CNN", "3D-CNN", "SyCNN"]
# for i, txt in enumerate(n):
#     ax.annotate(txt, (x[i], y[i]))
# plt.xlabel('Times (ms)')
# plt.ylabel('OA (%)')
# plt.legend()
# plt.savefig('E:/combine convolution transformer/efficiency.jpg')
# plt.show()