import numpy as np
import matplotlib.pyplot as plt

x = [29.354, 13.001, 27.661, 111.726, 21.234, 18.044, 74.401, 9.235, 15.396,69.374]
y = [98.03, 91.61, 90.08, 97.26, 93.00, 95.98, 87.42, 95.50, 91.68, 95.65]
n = np.arange(10)

fig, ax = plt.subplots()
ax.scatter(x, y, c='r', s=150)
n = ["Ours", "ViT", "Deep ViT", "CvT", "LeViT", "Swin Transformer", "Focal Transformer", "2D-CNN", "3D-CNN", "SyCNN"]
for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))
plt.xlabel('The number of parameters)')
plt.ylabel('OA (%)')
# plt.legend()
plt.savefig('G:/submitting paper/ACT_results/efficiency.png')
plt.show()