## 张量（Tensor）
**张量（Tensor）** 是一个数学概念，它是一个可以表示在多个向量、标量和其他张量之间的线性关系的多线性函数。

在机器学习和深度学习中，张量通常指的是一个**多维数组** ，**它是神经网络中数据的基本单位**。例如，一个 0 维张量是一个单独的数值（标量），一个 1 维张量是一个数组（向量），一个 2 维张量是一个矩阵，而更高维度的张量可以看作是**矩阵的推广**。


***

<br>

## torch.randn()
<code>randn(\*size)</code> **用于生成一个张量**，**该张量填充有从标准正态分布（均值为 0，标准差为 1）中抽取的随机数**。

这个函数**在神经网络的权重初始化中非常有用**，因为它可以帮助避免权重值过大或过小的情况。也可以用来**向数据中添加一些随机生成的噪声**。

具体来说，<code>torch.randn(\*size)</code> 的 size 参数定义了输出张量的形状。

例如，<code>torch.randn(2, 3)</code> 会生成一个 2x3 的张量，其元素都是从标准正态分布中随机抽取的。

***

<br>

## torch.utils.data.TensorDataset() 
<code>TensorDataset(data_tensor, target_tensor)</code>  **即创建包装数据和目标张量的数据集。**

- data_tensor：（Tensor），包含了样本的数据。
- target_tensor：（Tensor），包含了样本的目标（答案、标签）。

***
<br>

## torch.utils.data.DataLoader()
<code>DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)</code>

- 参数：
    - dataset (Dataset) – 加载数据的数据集。
    - batch_size (int, optional) – 每个 batch 加载多少个样本(默认: 1)。
    - shuffle (bool, optional) – 设置为 True 时会在每个 epoch 重新打乱数据(默认: False)。
    - sampler (Sampler, optional) – 定义从数据集中提取样本的策略。如果指定，则忽略 shuffle 参数。
    - num_workers (int, optional) – 用多少个子进程加载数据。0 表示数据将在主进程中加载(默认: 0)
    - collate_fn (callable, optional) – ~~官网没写注解~~
    - pin_memory (bool, optional) – ~~官网没写注解~~
    - drop_last (bool, optional) – 如果数据集大小不能被 batch size 整除，则设置为 True 后可删除最后一个不完整的 batch。如果设为 False 并且数据集的大小不能被 batch size 整除，则最后一个 batch 将更小。(默认: False)

~~参数太多了看不懂，还是来看例子吧。~~

#### 例子：

1. <code>DataLoader(dataset, batch_size=10, shuffle=True)</code>，即对 TensorDataset
生成的数据集 dataset 进行加载:
    - <code>batch_size</code>: 

        - 批量加载：<code>batch_size=10</code> 指定了每个批次加载的样本数量。在这个例子中，每次迭代将返回 10 个样本的数据和标签。
    - <code>shuffle</code>:

        - 洗牌：<code>shuffle=True</code> 表示在每个 epoch 开始时，数据将被打乱。这有助于模型学习时的泛化能力，因为它确保了模型不会记住批次中样本的顺序。
<br>
2.
    - [ ] 后续补充更多例子

***

<br>

