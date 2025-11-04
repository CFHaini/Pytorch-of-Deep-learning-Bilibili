import numpy as np
import torch
import matplotlib.pyplot as plt

'''
# @Collector and Speaker : little snow
# @Owner : 考研数学计算机之路 2024-2025 LLM微调实战项目

# 相关前备知识：
    torch.from_numpy()和numpy()函数，用于在PyTorch张量和NumPy数组间转换：
        https://blog.csdn.net/qq_41813454/article/details/129838551

# 代码重点解读：
    torch.nn.Sigmoid() 也被看作是网络的一层，而不是简单的函数使用 
    永远记住，神经网络的参数w和b才是网络需要学习的对象：
        权重 w 是一个可学习的参数，决定了输入特征如何影响输出。
        偏置 b 是另一个可学习的参数，用于调整输出。
    你在代码中看不到w和b，是因为他们是torch.nn.Linear层自动创建和管理的。

    你可以通过 named_parameters() 来查看这些权重和偏置：
    for name, param in model.named_parameters():
        if param.requires_grad:  # 只查看可学习的参数
            print(name, param.data)
'''


# 定义一个神经网络模型类，继承自 torch.nn.Module
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 定义四个线性层，分别为 8 -> 6, 6 -> 4, 4 -> 2, 2 -> 1 的变换
        self.linear1 = torch.nn.Linear(8, 6)  # 第一层：输入8维，输出6维
        self.linear2 = torch.nn.Linear(6, 4)  # 第二层：输入6维，输出4维
        self.linear3 = torch.nn.Linear(4, 2)  # 第三层：输入4维，输出2维
        self.linear4 = torch.nn.Linear(2, 1)  # 第四层：输入2维，输出1维
        self.sigmoid = torch.nn.Sigmoid()  # 定义Sigmoid激活函数

    def forward(self, x):
        # 前向传播过程，依次通过每一层并应用Sigmoid激活函数
        x = self.sigmoid(self.linear1(x))  # 第一层输出
        x = self.sigmoid(self.linear2(x))  # 第二层输出
        x = self.sigmoid(self.linear3(x))  # 第三层输出  
        x = self.sigmoid(self.linear4(x))  # 第四层输出，即预测值 y_hat
        return x  # 返回最终输出

# 实例化模型
model = Model()

# 构建损失函数和优化器
# 使用二元交叉熵损失函数，reduction='mean'表示返回平均损失
criterion = torch.nn.BCELoss(reduction='mean')
# 使用随机梯度下降（SGD）作为优化器，学习率为0.1
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 训练循环，进行前向传播、反向传播和参数更新
# 注意此处设置跑99万9999轮，如果大家只是测试的话，跑100000轮就够了
for epoch in range(999999):
    # 前向传播，计算预测值
    y_pred = model(x_data)
    # 计算损失
    loss = criterion(y_pred, y_data)
    # print(epoch, loss.item())  # 打印当前迭代次数和损失值

    # 清零梯度
    optimizer.zero_grad()
    # 反向传播，计算梯度
    loss.backward()
    # 更新模型参数
    optimizer.step()

    # 每10000次迭代打印一次损失和准确率
    if epoch % 10000 == 9999:
        # 将预测值转换为标签，阈值为0.5
        y_pred_label = torch.where(y_pred >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))

        # 计算准确率
        acc = torch.eq(y_pred_label, y_data).sum().item() / y_data.size(0)
        print("loss = ", loss.item(), "acc = ", acc)  # 打印损失和准确率
