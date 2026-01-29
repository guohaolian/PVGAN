import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# 加载生成数据
generated_data_path = "WGAN-GP/filled_data.csv"
generated_data = pd.read_csv(generated_data_path)

# 加载真实完整数据集
real_data_path = "data/weather_data.csv"
real_data = pd.read_csv(real_data_path)

# 确保生成数据和真实数据的列一致
real_data_features = real_data.drop(columns=['timestamp'], errors='ignore')  # 去除时间戳列
generated_data_features = generated_data.drop(columns=['timestamp'], errors='ignore')

# 确保两者的列顺序一致
assert list(real_data_features.columns) == list(generated_data_features.columns), "列顺序不一致，请检查数据！"

# 归一化真实数据和生成数据到相同范围
scaler = MinMaxScaler(feature_range=(0, 1))  # 归一化范围与生成器一致
real_data_normalized = scaler.fit_transform(real_data_features)
generated_data_normalized = scaler.transform(generated_data_features)

# 计算全局重构误差
mse = mean_squared_error(real_data_normalized.flatten(), generated_data_normalized.flatten())
mae = mean_absolute_error(real_data_normalized.flatten(), generated_data_normalized.flatten())
# 输出全局重构误差
print(f"Global Mean Squared Error (MSE): {mse}")
print(f"Global Mean Absolute Error (MAE): {mae}")

# 输出全局重构误差
global_results = f"Global Mean Squared Error (MSE): {mse}\nGlobal Mean Absolute Error (MAE): {mae}\n"

# 逐特征计算误差
feature_errors = {}
feature_results = "\nPer Feature Errors:\n"
for i, feature_name in enumerate(real_data_features.columns):
    feature_mse = mean_squared_error(real_data_normalized[:, i], generated_data_normalized[:, i])
    feature_mae = mean_absolute_error(real_data_normalized[:, i], generated_data_normalized[:, i])
    feature_errors[feature_name] = {"MSE": feature_mse, "MAE": feature_mae}
    feature_results += f"Feature: {feature_name} -> MSE: {feature_mse}, MAE: {feature_mae}\n"


print("\nPer Feature Errors:")
for feature, errors in feature_errors.items():
    print(f"Feature: {feature} -> MSE: {errors['MSE']}, MAE: {errors['MAE']}")
# 保存结果到文件
output_file = "WGAN-GP/evaluation_results.txt"
with open(output_file, "w") as f:
    f.write(global_results)
    f.write(feature_results)

print(f"评估结果已保存到 {output_file}")
