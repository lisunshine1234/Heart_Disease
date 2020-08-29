# Generative Adversarial Networks (GAN) example in PyTorch.
# See related blog post at https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.sch4xgsa9
from classify.set import *
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

set = set4[set4[:, -1] == 1][:, range(columns_count4 - 1)]
column_len = len(set[0, :])
row_len = len(set)
print(column_len)
print(row_len)
# Model params
g_input_size = (columns_count4 - 1) * len(set)  # Random noise dimension coming into generator, per output vector
g_hidden_size = 128  # Generator complexity
g_output_size = 100  # size of generated output vector
d_input_size = columns_count4 - 1  # Minibatch size - cardinality of distributions
d_hidden_size = 128  # Discriminator complexity
d_output_size = 1  # Single dimension for 'real' vs. 'fake'

d_learning_rate = 2e-4  # 2e-4
g_learning_rate = 2e-4
optim_betas = (0.9, 0.999)
num_epochs = 1000
print_interval = 200
d_steps = 1  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
g_steps = 1


# ##### DATA: Target data and generator input data
def get_distribution_sampler():
    set1 = set[np.newaxis, :]
    return torch.Tensor(set1)


def get_generator_input_sampler():
    return torch.rand(13, 126, 1)  # Uniform-dist data into generator, _NOT_ Gaussian


def extract(v):
    return v.data.storage().tolist()


def stats(d):
    return [np.mean(d), np.std(d)]


# 超过2万训练回合, 平均 G 的输出过度 4.0, 但然后回来在一个相当稳定, 正确的范围 (左)。同样, 标准偏差最初下落在错误方向, 但然后上升到期望1.25 范围 (正确), 匹配 R。
# ##### MODELS: Generator model and discriminator model
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        # self.gen = nn.Sequential(
        #     nn.Conv1d(13, 13, 1),  # in_channels, out_channels, kernel_size
        #     # nn.ReLU(True),
        #     # nn.Conv1d(128, 128, 1),
        #     # nn.ReLU(True),
        #     # nn.Conv1d(128, columns_count4 * len(set), 1),
        #     nn.Sigmoid()    #fc
        # )
        self.conv1 = nn.Conv1d(13, 128, 3)  # 定义conv1函数的是图像卷积函数：输入为图像（1个频道，即灰度图）,输出为 6张特征图, 卷积核为5x5正方形
        self.conv2 = nn.Conv1d(128, 256, 3)  # 定义conv2函数的是图像卷积函数：输入为6张特征图,输出为16张特征图, 卷积核为5x5正方形
        self.fc1 = nn.Linear(256 * 3 * 3, 120)  # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将16*5*5个节点连接到120个节点上。
        self.fc2 = nn.Linear(120, 84)# 定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将120个节点连接到84个节点上。
        self.fc3 = nn.Linear(84, 13)# 定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b，并将84个节点连接到10个节点上。

    def forward(self, x):
        # x = self.gen(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # 输入x经过卷积conv1之后，经过激活函数ReLU（原来这个词是激活函数的意思），使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 输入x经过卷积conv2之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = x.view(-1, self.num_flat_features(x))  # view函数将张量x变形成一维的向量形式，总特征数并不改变，为接下来的全连接作准备。
        x = F.relu(self.fc1(x))  # 输入x经过全连接1，再经过ReLU激活函数，然后更新x
        x = F.relu(self.fc2(x))  # 输入x经过全连接2，再经过ReLU激活函数，然后更新x
        x = self.fc3(x)  # 输入x经过全连接3，然后更新x
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv1d(13, 13, 1),  # in_channels, out_channels, kernel_size
            # nn.LeakyReLU(0.2),
            # nn.Conv1d(128, 128, 1),
            # nn.LeakyReLU(0.2),
            # nn.Conv1d(128, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        return x


d_sampler = get_distribution_sampler()
g_sampler = get_generator_input_sampler()

G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)

# 计算损失函数
criterion = nn.BCELoss()

# torch.optim是一个实现了多种优化算法的包
# Adam利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。它的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳
d_optimizer = torch.optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = torch.optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)

d_loss_set = []
g_loss_set = []
d_real_loss_set = []
d_fake_loss_set = []
real_scores_set = []
fake_scores_set = []

for epoch in range(num_epochs):
    for d_index in range(d_steps):
        # 计算real数据的损失
        d_real_data = Variable(d_sampler)
        d_real_out = D(d_real_data)
        d_real_loss = criterion(d_real_out, Variable(torch.ones(13)))  # ones = true
        real_scores = d_real_out  # closer to 1 means better

        #  计算fake数据的损失
        d_gen_input = Variable(g_sampler)
        d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
        d_fake_out = D(d_fake_data.t())
        d_fake_loss = criterion(d_fake_out, Variable(torch.zeros(13)))  # zeros = fake
        fake_scores = d_fake_out  # closer to 0 means better

        d_loss = d_real_loss + d_fake_loss
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        gen_input = Variable(g_sampler)
        g_fake_data = G(gen_input)
        dg_fake_out = D(g_fake_data.t())
        g_loss = criterion(dg_fake_out, Variable(torch.ones(13)))  # we want to fool, so pretend it's all genuine

        g_optimizer.zero_grad()  # 将上次迭代计算的梯度值清0
        g_loss.backward()  # 反向传播，计算梯度值
        g_optimizer.step()  # 更新权值参数

    if (epoch) % print_interval == 0:
        print('Epoch [{}/{}], d_loss: {}, g_loss: {} ''D real: {}, D fake: {}'.format(epoch, num_epochs, d_loss.data[0], g_loss.data[0],
                                                                                      real_scores.data.mean(), fake_scores.data.mean()))

    d_loss_set.append(d_loss)
    g_loss_set.append(g_loss)
    d_real_loss_set.append(d_real_loss)
    d_fake_loss_set.append(d_fake_loss)
    real_scores_set.append(real_scores)
    fake_scores_set.append(fake_scores)

    # if epoch == num_epochs - 1:
    #     with open(data_path + 'g_fake_data.csv', 'w', newline='') as file_csv:
    #         write_csv = csv.writer(file_csv)
    #         write_csv.writerows(g_fake_data.data.numpy().T)
    #         file_csv.close()
    #     with open(data_path + 'd_real_out.csv', 'w', newline='') as file_csv:
    #         write_csv = csv.writer(file_csv)
    #         write_csv.writerows(d_real_out.data.numpy().T)
    #         file_csv.close()
    #     with open(data_path + 'g_fake_data.csv', 'w', newline='') as file_csv:
    #         write_csv = csv.writer(file_csv)
    #         write_csv.writerows(g_fake_data.data.numpy().T)
    #         file_csv.close()
    #     with open(data_path + 'dg_fake_out.csv', 'w', newline='') as file_csv:
    #         write_csv = csv.writer(file_csv)
    #         write_csv.writerows(dg_fake_out.data.numpy().T)
    #         file_csv.close()

    # with open(data_path + 'gan_real.csv', 'w', newline='') as file_csv:
    #     write_csv = csv.writer(file_csv)
    #     write_csv.writerows(d_real_data)
    # with open(data_path + 'gan_fake.csv', 'w', newline='') as file_csv:
    #     write_csv = csv.writer(file_csv)
    #     write_csv.writerows(d_fake_data)

plt.figure('d_loss')
plt.scatter(range(num_epochs), d_loss_set, s=10)
plt.figure('g_loss')
plt.scatter(range(num_epochs), g_loss_set, s=10)
plt.figure('d_real_loss')
plt.scatter(range(num_epochs), d_real_loss_set, s=10)
plt.figure('d_fake_loss')
plt.scatter(range(num_epochs), d_fake_loss_set, s=10)

plt.show()
