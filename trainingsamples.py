import matplotlib.pyplot as plt
import numpy as np

x1 = [1, 2, 3, 4]
x2 = [0.5, 1, 1.5, 2]
x3 = [1, 2, 10, 20,]
x4 = [5, 9, 10, 20,]

y1 = [96.55, 97.55,	98.10, 98.42]
y2 = [98.74, 99.01,	99.17, 99.28]
y3 = [91.11, 92.83, 95.40, 95.74,]
y4 = [98.71, 98.95, 99.09, 99.11,]
plt.xlabel("The input size ")
plt.ylabel("OA (%)")
plt.plot(x1,y1, label='Houston2013', marker='o' )
plt.plot(x2,y2, label='WHL', marker='s',)
plt.plot(x3,y3, label='IndianPines',marker='^', )
plt.plot(x4,y4, label='PaviaU', marker='v', )
plt.legend()
plt.show()