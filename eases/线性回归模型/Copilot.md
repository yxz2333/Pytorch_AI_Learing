## Copilot

使用 PyTorch 编写一个简单的线性回归模型并实现交互，您可以按照以下步骤进行：

- 导入必要的库：

```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
```

- 生成数据集：
```
# 生成数据集

x = torch.randn(100, 1) _ 10 # 输入特征
y = 2 _ x + 3 + torch.randn(100, 1) \* 2 # 输出值，其中包含一些噪声
```
- 创建数据加载器：
```
# 将数据封装进 DataLoader

data_set = TensorDataset(x, y)
data_loader = DataLoader(data_set, batch_size=10, shuffle=True)
```
- 定义模型：
```
# 定义线性回归模型

class LinearRegressionModel(nn.Module):
def **init**(self):
super(LinearRegressionModel, self).**init**()
self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()
```
- 设置损失函数和优化器：
```
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```
- 训练模型：
```
# 训练模型

num_epochs = 60
for epoch in range(num_epochs):
for inputs, targets in data_loader: # 前向传播
outputs = model(inputs)
loss = criterion(outputs, targets)

        # 后向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
```
- 可视化结果：
```
# 绘制数据点和拟合的线

predicted = model(x).detach()
plt.scatter(x.numpy(), y.numpy(), label='Original data')
plt.plot(x.numpy(), predicted.numpy(), label='Fitted line', color='red')
plt.legend()
plt.show()
```
- 交云端互：
    - 您可以通过接收用户输入来交互式地使用模型。 
    - 例如，您可以让用户输入新的 x 值，并使用模型来预测对应的 y 值。
    这个例子展示了如何使用 PyTorch 构建和训练一个简单的线性回归模型，并通过可视化来展示模型的拟合效果。您可以根据需要调整模型结构、损失函数、优化器和训练参数。
