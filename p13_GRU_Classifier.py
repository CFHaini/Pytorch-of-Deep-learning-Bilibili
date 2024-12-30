import torch 
from torch.utils.data import Dataset  # 从torch.utils.data导入Dataset类
from torch.utils.data import DataLoader  # 从torch.utils.data导入DataLoader类
import gzip  # 导入gzip库，用于处理gzip压缩文件
import csv  # 导入csv库，用于读取csv文件
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图
import numpy as np 
import time  
import math

'''
# @Collector and Speaker : little snow
# @Owner : 考研数学计算机之路 2024-2025 LLM微调实战项目
'''

class NameDataset(Dataset):
    def __init__(self, is_train_set):
        # 根据是否是训练集选择文件名
        filename = './names_train.csv.gz' if is_train_set else './names_test.csv.gz'
        with gzip.open(filename, 'rt') as f:    # 以只读模式打开gzip文件
            reader = csv.reader(f)  # 创建csv读取器
            rows = list(reader)  # 读取所有行
        self.names = [row[0] for row in rows]  # 提取名字
        self.len = len(self.names)  # 记录名字的数量
        self.countries = [row[1] for row in rows]  # 提取对应的国家

        # 获取国家列表并创建国家字典
        self.country_list = list(sorted(set(self.countries)))  # 去重并排序国家
        self.country_dict = self.getCountryDict()  # 创建国家字典
        self.country_num = len(self.country_list)  # 记录国家数量

    def __getitem__(self, index):  # 根据索引获取名字和对应国家的索引
        return self.names[index], self.country_dict[self.countries[index]]

    def __len__(self):
        return self.len  # 返回数据集的长度

    def getCountryDict(self):
        # 创建国家字典，将国家名映射到索引
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list, 0):
            country_dict[country_name] = idx
        return country_dict

    def idx2country(self, index):
        return self.country_list[index]  # 根据索引返回国家名

    def getCountriesNum(self):
        return self.country_num  # 返回国家数量


HIDDEN_SIZE = 100  # 隐藏层大小
BATCH_SIZE = 256  # 批处理大小
N_LAYER = 2  # GRU层数
N_EPOCHS = 25  # 训练的epoch数量
# ASCII 字符集包含了 128 个字符（从 0 到 127），包括英文字母、数字、标点符号和一些控制字符
N_CHARS = 128  # GRU中的输入大小，控制嵌入层的形状

# 创建训练和测试数据集及数据加载器
trainSet = NameDataset(is_train_set=True)  # 创建训练集
trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True)  # 创建训练数据加载器
testSet = NameDataset(is_train_set=False)  # 创建测试集
testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=False)  # 创建测试数据加载器

N_COUNTRY = trainSet.getCountriesNum()  # 获取国家数量


class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()  # 调用父类构造函数
        self.hidden_size = hidden_size  # 隐藏层大小
        self.n_layers = n_layers  # GRU层数
        self.n_directions = 2 if bidirectional else 1  # 使用双向GRU

        # 嵌入层（𝑠𝑒𝑞𝐿𝑒𝑛, 𝑏𝑎𝑡𝑐ℎ𝑆𝑖𝑧𝑒） --> (𝑠𝑒𝑞𝐿𝑒𝑛, 𝑏𝑎𝑡𝑐ℎ𝑆𝑖𝑧𝑒, hidden_size)
        self.embedding = torch.nn.Embedding(input_size, hidden_size)  # 嵌入层
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)  # GRU层
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)  # 全连接层

    def _init_hidden(self, batch_size):
        # 初始化隐藏状态
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size) 
        return hidden

    def forward(self, input, seq_lengths):
        # 前向传播
        # input shape : B x S -> S x B
        input = input.t()  # 转置输入
        batch_size = input.size(1)  # 获取批处理大小
        hidden = self._init_hidden(batch_size)  # 初始化隐藏状态
        embedding = self.embedding(input)  # 获取嵌入表示

        # pack them up
        gru_input = torch.nn.utils.rnn.pack_padded_sequence(embedding, seq_lengths)  # 打包序列
        output, hidden = self.gru(gru_input, hidden)  # GRU前向传播
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)  # 双向GRU拼接隐藏状态
        else:
            hidden_cat = hidden[-1]  # 单向GRU的隐藏状态
        fc_output = self.fc(hidden_cat)  # 全连接层输出
        return fc_output  # 返回输出
    
# 这四个传参分别对应类的初始化中的 input_size, hidden_size, output_size, n_layers
classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)  # 创建分类器

criterion = torch.nn.CrossEntropyLoss()  # 定义损失函数
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)  # 定义优化器


def name2list(name):
    # 将名字转换为字符的ASCII值列表
    arr = [ord(c) for c in name]  # 获取每个字符的ASCII值
    return arr, len(arr)  # 返回ASCII值列表和长度


def make_tensors(names, countries):
    # 将名字和国家转换为张量
    sequences_and_lengths = [name2list(name) for name in names]  # 获取名字的ASCII值和长度
    name_sequences = [s1[0] for s1 in sequences_and_lengths]  # 提取ASCII值列表
    seq_lengths = torch.LongTensor([s1[1] for s1 in sequences_and_lengths])  # 提取长度
    countries = countries.long()  # 转换国家为长整型

    # 创建名字的张量，形状为 BatchSize * seqLen
    # 他这里补零的方式先将所有的0 Tensor给初始化出来，然后在每行前面填充每个名字
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()  # 初始化张量
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)  # 填充名字的ASCII值

    # 按长度排序以使用pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)  # 按长度降序排列
    seq_tensor = seq_tensor[perm_idx]  # 根据排序索引重新排列名字张量
    countries = countries[perm_idx]  # 根据排序索引重新排列国家张量

    # 返回排序后的名字张量、长度张量和国家张量
    return seq_tensor, seq_lengths, countries


def trainModel():
    # 训练模型
    def time_since(since):
        # 计算经过的时间
        s = time.time() - since
        m = math.floor(s / 60)  # 计算分钟
        s -= m * 60  # 计算秒
        return '%dm %ds' % (m, s)  # 返回格式化的时间字符串

    total_loss = 0  # 初始化总损失
    for i, (names, countries) in enumerate(trainLoader, 1):  # 遍历训练数据
        inputs, seq_lengths, target = make_tensors(names, countries)  # 制作张量

        output = classifier(inputs, seq_lengths)  # 前向传播
        loss = criterion(output, target)  # 计算损失
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        total_loss += loss.item()  # 累加损失
        if i % 10 == 0:  # 每10个batch打印一次信息
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
            print(f'[{i * len(inputs)}/{len(trainSet)}] ', end='')
            print(f'loss={total_loss / (i * len(inputs))}')  # 打印当前损失
    return total_loss  # 返回总损失


def testModel():
    # 测试模型
    correct = 0  # 初始化正确预测数量
    total = len(testSet)  # 测试集总样本数
    print("evaluating trained model ... ")  # 打印评估信息
    with torch.no_grad():  # 不计算梯度
        for i, (names, countries) in enumerate(testLoader):  # 遍历测试数据
            inputs, seq_lengths, target = make_tensors(names, countries)  # 制作张量
            output = classifier(inputs, seq_lengths)  # 前向传播
            pred = output.max(dim=1, keepdim=True)[1]  # 获取预测结果
            correct += pred.eq(target.view_as(pred)).sum().item()  # 统计正确预测数量

        percent = '%.2f' % (100 * correct / total)  # 计算准确率
        print(f'Test set: Accuracy {correct}/{total} {percent}%')  # 打印准确率

    return correct / total  # 返回准确率


start = time.time()  # 记录开始时间
print("Training for %d epochs..." % N_EPOCHS)  # 打印训练信息
acc_list = []  # 初始化准确率列表
for epoch in range(1, N_EPOCHS + 1):  # 遍历每个epoch
    trainModel()  # 训练模型
    acc = testModel()  # 测试模型
    acc_list.append(acc)  # 记录准确率


epoch = np.arange(1, len(acc_list) + 1)  # 创建epoch数组
acc_list = np.array(acc_list)  # 转换为numpy数组
plt.plot(epoch, acc_list)  # 绘制准确率曲线
plt.xlabel('Epoch')  # 设置x轴标签
plt.ylabel('Accuracy')  # 设置y轴标签
plt.grid()  # 显示网格
plt.savefig('picture/rnn_classifier_accuracy_plot.png')  # 保存准确率图像
print(f"训练完成，训练指标图像已保存在 picture/rnn_classifier_accuracy_plot.png")  # 打印完成信息
plt.show()  # 显示图像
