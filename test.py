# import plotly.express as px
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# sns.axes_style('whitegrid');
# from scipy.io import loadmat
#
# # def read_HSI():
# #   X = loadmat('PaviaU.mat')['paviaU']
# #   y = loadmat('PaviaU_gt.mat')['paviaU_gt']
# #   print(f"X shape: {X.shape}\ny shape: {y.shape}")
# #   return X, y
# #
# # X, y = read_HSI()
import os
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
import matplotlib.patches as patches
# import matplotlib.colors as colors
import math

# fig = plt.figure()
# ax = fig.add_subplot(111)
# colormap = [[0, 0, 0], [255, 0, 0], [38, 115, 0], [255, 0, 197],
#                     [111, 74, 0], [85, 255, 0], [0, 112, 255], [255, 255, 190]]
# classes = ['background', 'build', 'Tree', 'grass', 'Soil',
#                    'mango', 'Water', 'Farm']
#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import scipy.io as io
# import matplotlib as mpl
# from PIL import Image
# import numpy as np
# import matplotlib.patches as mpatches
#
# t = 1
# # 自定义colormap
# def colormap():
#     return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#000000', '#FF0000', '#00FF00', '#FF00FF','#836FFF', '#528B8B', '#0000CD', '#FFFF00'], 256)
# cmap = { 0: [0,0,0, t], 1:[255/255, 0, 0, t], 2:[0/255, 255/255, 0, t], 3:[255/255, 0, 255/255, t],
#                     4:[131/255, 111/255, 255/255, t], 5:[82/255, 139/255, 139/255, t], 6:[0, 0/255, 255/255, t], 7:[255/255, 255/255, 0/255, t]}
# # cmap = {1:[219/255,94/255,86/255,t],2:[211/255,219/255,86/255,t],3:[86/255,219/255,94/255,t],4:[86/255,211/255,219/255,t],5:[94/255,86/255,219/255,t]}
# plt.figure(figsize=(20, 10))
#
# labels = { 0:'background', 1:'build', 2:'Tree', 3:'grass', 4:'Soil',
#                    5:'mango', 6:'Water', 7:'Farm'}
# # labels = {1:'ground',2:'building',3:'forest',4:'river',5:'road'}
# patches =[mpatches.Patch(color=cmap[i],label=labels[i]) for i in cmap]
#
# img = cv2.imread('E:/CWJ/pred.png')
# # img = cnn_2d = cv2.imread('E:/CWJ/gt.png')
# plt.imshow(img, cmap=colormap())
# plt.legend(handles=patches, loc=4, borderaxespad=0.)
# plt.axis("off")
# plt.savefig('E:/CWJ/pred.jpg', bbox_inches='tight', pad_inches=0)
# plt.show()
import numpy as np
import seaborn as sns
LABEL_VALUES = 9
palette = {0: (0, 0, 0)}
for k, color in enumerate(sns.color_palette("hls", LABEL_VALUES - 1)):
    palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))

print(palette)