import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 加载生成数据
generated_data_path = "WGAN-GP/filled_data.csv"
generated_data = pd.read_csv(generated_data_path)

# 加载真实数据
real_data_path = "data/weather_data.csv"
real_data = pd.read_csv(real_data_path)

# 确保生成数据和真实数据的列一致
real_data_features = real_data.drop(columns=['timestamp'], errors='ignore')  # 去除时间戳列
generated_data_features = generated_data.drop(columns=['timestamp'], errors='ignore')

# 确保两者的列顺序一致
assert list(real_data_features.columns) == list(generated_data_features.columns), "列顺序不一致，请检查数据！"

# 归一化真实数据和生成数据到相同范围
scaler = MinMaxScaler(feature_range=(0, 1))
real_data_normalized = scaler.fit_transform(real_data_features)
generated_data_normalized = scaler.transform(generated_data_features)

# 合并真实数据和生成数据
data_combined = np.concatenate([real_data_normalized, generated_data_normalized])
labels_combined = np.array([0] * real_data_normalized.shape[0] + [1] * generated_data_normalized.shape[0])  # 0 表示真实数据，1 表示生成数据

# 使用 t-SNE 降维
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
data_embedded = tsne.fit_transform(data_combined)

# 可视化
plt.figure(figsize=(10, 10))
plt.scatter(data_embedded[labels_combined == 0, 0], data_embedded[labels_combined == 0, 1], label='Real Data', alpha=0.6, s=10,color='blue')
plt.scatter(data_embedded[labels_combined == 1, 0], data_embedded[labels_combined == 1, 1], label='Generated Data', alpha=0.1, s=10,color='red')
plt.title('t-SNE Visualization of Real and Generated Data')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.grid(True)
plt.savefig("WGAN-GP/tsne_comparison.png")  # 保存图像
plt.show()

print("t-SNE 可视化已完成，并保存为 'tsne_comparison.png'")



# plt.figure(figsize=(15, 6))
#
# # 子图 1：真实数据
# plt.subplot(1, 2, 1)
# plt.scatter(data_embedded[labels_combined == 0, 0], data_embedded[labels_combined == 0, 1],
#             label='Real Data', alpha=0.5, s=10, color='blue')
# plt.title('Real Data t-SNE')
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.legend()
# plt.grid(True)
#
# # 子图 2：生成数据
# plt.subplot(1, 2, 2)
# plt.scatter(data_embedded[labels_combined == 1, 0], data_embedded[labels_combined == 1, 1],
#             label='Generated Data', alpha=0.5, s=10, color='yellow')
# plt.title('Generated Data t-SNE')
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.legend()
# plt.grid(True)
#
# plt.tight_layout()
# plt.savefig("GAN/tsne_comparison_separate.png")  # 保存图像
# plt.show()
#
# print("单独显示的 t-SNE 可视化已完成，并保存为 'tsne_comparison_separate.png'")
