# coding=utf-8
# ================================================================
#
#   File name   : optim_duibi.py
#   Author      : Faye
#   Created date: 2022/8/26 17:30
#   Description :
#
# ================================================================

import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 超参数
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

# 生成假数据
# torch.unsqueeze() 的作用是将一维变二维，torch只能处理二维的数据
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)  # x data (tensor), shape(100, 1)
# 0.2 * torch.rand(x.size())增加噪点
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

# 定义数据库
dataset = Data.TensorDataset(x, y)

# 定义数据加载器
loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


# 定义pytorch网络
class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        y = self.predict(x)
        return y


# 定义不同的优化器网络
net_LBFGS = Net(1, 10, 1)

# 选择不同的优化方法
opt_LBFGS = torch.optim.LBFGS(net_LBFGS.parameters(), lr=LR, max_iter=20)

nets = [net_LBFGS]
optimizers = [opt_LBFGS]

# 选择损失函数
loss_func = torch.nn.MSELoss()

# 不同方法的loss
loss_LBFGS = []

# 保存所有loss
losses = [loss_LBFGS]

# 执行训练
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader):
        var_x = Variable(batch_x)
        var_y = Variable(batch_y)
        for net, optimizer, loss_history in zip(nets, optimizers, losses):
            if isinstance(optimizer, torch.optim.LBFGS):
                def closure():
                    y_pred = net(var_x)
                    loss = loss_func(y_pred, var_y)
                    optimizer.zero_grad()
                    loss.backward()
                    return loss


                loss = optimizer.step(closure)
            else:
                # 对x进行预测
                prediction = net(var_x)
                # 计算损失
                loss = loss_func(prediction, var_y)
                # 每次迭代清空上一次的梯度
                optimizer.zero_grad()
                # 反向传播
                loss.backward()
                # 更新梯度
                optimizer.step()
            # 保存loss记录
            loss_history.append(loss.data)

# 画图
labels = ['LBFGS']
for i, loss_history in enumerate(losses):
    plt.plot(loss_history, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()
