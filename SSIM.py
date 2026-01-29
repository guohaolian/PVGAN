from skimage.metrics import structural_similarity as ssim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 加载真实完整数据和补全数据
real_data_path = "data/weather_data.csv"  # 真实完整数据路径
filled_data_path = "WGAN-GP/filled_data.csv"  # 补全数据路径

real_data = pd.read_csv(real_data_path).drop(columns=['timestamp'], errors='ignore')
filled_data = pd.read_csv(filled_data_path).drop(columns=['timestamp'], errors='ignore')

# 确保两者的列顺序一致
assert list(real_data.columns) == list(filled_data.columns), "列顺序不一致，请检查数据！"

# 对真实数据和补全数据归一化到 [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
real_data_normalized = scaler.fit_transform(real_data)
filled_data_normalized = scaler.transform(filled_data)

# 逐列计算 SSIM
ssim_scores = []
for i, column in enumerate(real_data.columns):
    real_column = real_data_normalized[:, i]
    filled_column = filled_data_normalized[:, i]
    score = ssim(real_column, filled_column, data_range=1.0)  # SSIM 的值在 [0, 1] 范围
    ssim_scores.append((column, score))

# 转为 DataFrame 显示
ssim_results = pd.DataFrame(ssim_scores, columns=['Feature', 'SSIM'])

# 输出 SSIM 结果
print("SSIM Scores for Each Feature:")
print(ssim_results)

# 保存 SSIM 结果
ssim_results.to_csv("WGAN-GP/ssim_results.csv", index=False)
print("SSIM 结果已保存为 'ssim_results.csv'")
