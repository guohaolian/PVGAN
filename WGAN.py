import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据集
data = pd.read_csv('data/weather_data_with_missing.csv')

# 处理时间戳
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# 处理缺失值：Power列单独处理
power_data = data[['Power']].copy()
data = data.drop(columns=['Power'])

# 对非Power列进行缺失值填充，使用SimpleImputer
imputer = SimpleImputer(strategy='mean')
data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns, index=data.index)

# 归一化数据（除Power列）
scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data_filled), columns=data_filled.columns, index=data_filled.index)

# 将Power列和其他特征一起处理
power_data_scaled = scaler.fit_transform(power_data)

# 将数据分割为训练集和测试集
train_data, test_data = train_test_split(data_scaled, test_size=0.2, shuffle=False)

# 转换为Tensor
train_tensor = torch.tensor(train_data.values, dtype=torch.float32).to(device)
test_tensor = torch.tensor(test_data.values, dtype=torch.float32).to(device)

# 创建数据加载器
batch_size = 64
train_dataset = TensorDataset(train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义生成器：自编码器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(train_data.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, train_data.shape[1]),
            nn.Tanh()  # 输出层的激活函数，确保输出在[-1, 1]之间
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 定义判别器：循环神经网络
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.rnn = nn.LSTM(input_size=train_data.shape[1], hidden_size=64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出概率值
        )

    def forward(self, x):
        rnn_out, (hn, cn) = self.rnn(x)
        output = self.fc(hn[-1])  # 使用最后一个时间步的隐藏状态
        return output

# 创建生成器和判别器
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 训练模型
epochs = 50
for epoch in range(epochs):
    for batch in train_loader:
        real_data = batch[0].to(device)  # 获取真实数据并移动到设备上
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, 1).to(device)  # 真实数据标签
        fake_labels = torch.zeros(batch_size, 1).to(device)  # 生成数据标签

        # 训练判别器
        optimizer_d.zero_grad()

        # 计算判别器在真实数据上的损失
        real_output = discriminator(real_data.unsqueeze(1))  # RNN需要三维输入
        real_loss = criterion(real_output, real_labels)

        # 生成假数据
        fake_data = generator(real_data)
        fake_output = discriminator(fake_data.detach().unsqueeze(1))  # 不计算生成器的梯度
        fake_loss = criterion(fake_output, fake_labels)

        # 总体判别器损失
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d.step()

        # 训练生成器
        optimizer_g.zero_grad()

        # 判别器认为假数据是真实的
        fake_output = discriminator(fake_data.unsqueeze(1))
        g_loss = criterion(fake_output, real_labels)
        g_loss.backward()
        optimizer_g.step()

    print(f'Epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

# 使用生成器进行缺失值补插
def fill_missing_values(data_with_missing):
    missing_data = torch.tensor(data_with_missing, dtype=torch.float32).to(device)
    filled_data = generator(missing_data)
    return filled_data.cpu().detach().numpy()

# 补充Power列
def fill_power_column(power_data):
    power_data_filled = scaler.inverse_transform(power_data)  # 逆归一化
    return power_data_filled

# 预测缺失值
filled_data = fill_missing_values(test_tensor.cpu().numpy())

# 恢复到原始数据范围（反归一化）
filled_data = scaler.inverse_transform(filled_data)

# 填充Power列
filled_power_data = fill_power_column(power_data_scaled)

# 完成最终的填充
final_filled_data = pd.DataFrame(filled_data, columns=data.columns, index=test_data.index)
final_filled_data['Power'] = filled_power_data

# 输出填充后的数据
final_filled_data.to_csv('filled_data.csv', index=True)
