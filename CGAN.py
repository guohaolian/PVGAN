import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

# 数据加载与预处理
data = pd.read_csv("data/weather_data_with_missing.csv")
data_features = data.drop(columns=['timestamp'], errors='ignore')  # 去除非数值列
scalers = {}  # 保存每列的归一化器

# 对所有列单独归一化
normalized_data = []
for col in data_features.columns:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scalers[col] = scaler
    normalized_data.append(scaler.fit_transform(data_features[[col]]).flatten())
data_normalized = np.array(normalized_data).T

# 缺失值掩码与已知值分离
mask = ~np.isnan(data_normalized)  # 缺失值为 False，已知值为 True
data_known = np.where(mask, data_normalized, 0)  # 用 0 替换缺失值

# 转换为 PyTorch 张量
data_known = torch.tensor(data_known, dtype=torch.float32)
mask = torch.tensor(mask, dtype=torch.float32)

# 检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义超参数
input_dim = data_known.shape[1]  # 输入特征维度
latent_dim = 64  # 噪声向量维度
batch_size = 64
epochs = 10000
learning_rate_D= 0.0001
learning_rate_G = 0.0002

# 定义生成器
class Generator(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + input_dim, 128),  # 噪声 + 条件输入
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # 输出范围为 [0, 1]
        )

    def forward(self, noise, condition):
        x = torch.cat((noise, condition), dim=1)
        return self.model(x)


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, 256),  # 数据 + 条件输入
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出范围为 [0, 1]
        )

    def forward(self, data, condition):
        x = torch.cat((data, condition), dim=1)
        return self.model(x)


# 初始化生成器和判别器
generator = Generator(input_dim, latent_dim).to(device)
discriminator = Discriminator(input_dim).to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate_G, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate_D, betas=(0.5, 0.999))

# 用于记录损失的列表
d_loss_list = []
g_loss_list = []

# 训练过程
# 改进后的训练循环
for epoch in range(epochs):
    for _ in range(2):  # 训练判别器两次
        idx = np.random.randint(0, data_known.shape[0], batch_size)
        real_data = data_known[idx].to(device)
        real_condition = real_data.to(device)
        real_labels = torch.full((batch_size, 1), 0.9, device=device)  # 平滑真实标签
        fake_labels = torch.full((batch_size, 1), 0.1, device=device)  # 平滑假标签

        # 生成假样本
        noise = torch.randn(batch_size, latent_dim).to(device)
        fake_data = generator(noise, real_condition)

        # 判别器损失
        real_loss = criterion(discriminator(real_data, real_condition), real_labels)
        fake_loss = criterion(discriminator(fake_data.detach(), real_condition), fake_labels)
        d_loss = real_loss + fake_loss

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

    # 训练生成器
    for _ in range(1):  # 训练生成器一次
        noise = torch.randn(batch_size, latent_dim).to(device)
        fake_data = generator(noise, real_condition)
        g_loss = criterion(discriminator(fake_data, real_condition), real_labels)  # 希望生成器生成的数据判别为真

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    # 记录损失
    d_loss_list.append(d_loss.item())
    g_loss_list.append(g_loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}: [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")


# 使用生成器补全缺失值
noise = torch.randn(data_known.shape[0], latent_dim).to(device)
data_known_device = data_known.to(device)
generated_data = generator(noise, data_known_device).detach().cpu().numpy()

# 替换缺失值
filled_data = np.where(mask.numpy(), data_normalized, generated_data)
filled_data = pd.DataFrame(filled_data, columns=data_features.columns)

# 反归一化
for col in data_features.columns:
    filled_data[col] = scalers[col].inverse_transform(filled_data[[col]])

# 保存补全后的数据
filled_data['timestamp'] = data['timestamp']
filled_data.to_csv("CGAN/filled_data_01.csv", index=False)
print("缺失值已补全，并保存为 'CGAN/filled_data.csv'.")

# 可视化损失并保存
plt.figure(figsize=(10, 6))
plt.plot(range(len(d_loss_list)), d_loss_list, label='Discriminator Loss')
plt.plot(range(len(g_loss_list)), g_loss_list, label='Generator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss of Discriminator and Generator')
plt.legend()
plt.grid(True)
plt.savefig("CGAN/loss_visualization_01.png")  # 保存图像
plt.show()

print("训练损失曲线已保存为 'CGAN/loss_visualization.png'")
