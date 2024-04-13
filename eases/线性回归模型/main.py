# 导入PyTorch及相关库
import torch  # PyTorch的主要库，提供了张量对象和多种操作
import torch.nn as nn  # nn模块定义了一系列神经网络层和损失函数
import torch.optim as optim  # optim模块提供了多种优化算法
from torch.utils.data import DataLoader, TensorDataset  # 数据加载器和数据集的工具
import matplotlib.pyplot as plt  # 用于绘图的库

# 生成数据集
x = torch.randn(100, 1) * 10  # 随机生成100个正态分布的样本作为输入特征
y = 2 * x + 3 + torch.randn(100, 1) * 2  # 根据线性关系生成输出值，并添加噪声

# 将数据封装进 DataLoader
dataset = TensorDataset(x, y)  # 创建一个包含输入和输出的数据集
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)  # 创建一个数据加载器，用于批量处理和打乱数据


# 定义线性回归模型类
class LinearRegressionModel(nn.Module):  # 继承nn.Module，自定义模型
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 定义一个线性层，输入和输出特征都是1维

    def forward(self, x):
        return self.linear(x)  # 定义前向传播函数


model = LinearRegressionModel()  # 实例化模型

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 使用均方误差作为损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 使用随机梯度下降优化器，学习率设置为0.01

# 训练模型
num_epochs = 60  # 设置训练的轮数
for epoch in range(num_epochs):
    for inputs, targets in data_loader:
        outputs = model(inputs)  # 计算模型的输出
        loss = criterion(outputs, targets)  # 计算损失

        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')  # 每10轮输出一次损失值

# 绘制数据点和拟合的线
predicted = model(x).detach()  # 获取模型预测结果，并从计算图中分离出来
plt.scatter(x.numpy(), y.numpy(), label='Original data')  # 绘制原始数据点
plt.plot(x.numpy(), predicted.numpy(), label='Fitted line', color='red')  # 绘制拟合线
plt.legend()  # 显示图例
plt.show()  # 显示图表
