import pandas as pd
import numpy as np

# 读取数据集
file_path = 'data/weather_data.csv'
df = pd.read_csv(file_path)

# 删除比例
delete_ratio = 0.4  # 删除40%数据

# 设置随机种子，确保每次运行结果一致
np.random.seed(42)

# 获取不包含'timestamp'的所有列数据
data_to_delete = df.drop(columns=['timestamp'])

# 计算总单元格数和需要删除的单元格数
total_cells = data_to_delete.size
cells_to_delete = int(total_cells * delete_ratio)

# 获取所有的单元格位置
cell_positions = [(row, col) for row in range(data_to_delete.shape[0]) for col in range(data_to_delete.shape[1])]

# 随机选择需要删除的单元格位置
cells_to_delete_positions = np.random.choice(range(len(cell_positions)), size=cells_to_delete, replace=False)

# 将选中的单元格位置的值设置为None，表示清除内容
for idx in cells_to_delete_positions:
    row, col = cell_positions[idx]
    df.iloc[row, col + 1] = None  # 加1是因为timestamp列在前面



# 如果需要将结果保存为新的CSV文件，可以使用：
df.to_csv('data/weather_data_with_missing_20.csv', index=False)
