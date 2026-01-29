# windowkmeans_v2.py 参数说明（可调参速查）

> 目标：把 `windowkmeans_v2.py` 里所有可配置参数给出“作用 + 默认值 + 什么时候调 + 推荐范围 + 示例”。
> 
> 说明：如你在用的是 `windowkmeans_v1.py`，部分默认值（如 `--horizon`、`--output-dir`）与 v2 不同。

---

## 1) 输入输出（IO）

### `--data-path` (str)
- 默认：`data/91-Site_DKA-M9_B-Phase.csv`
- 作用：输入 CSV 路径。
- 要求：CSV 至少包含 `timestamp`（可解析为 datetime）、`target-col` 与 `feature-cols`。
- 示例：
  - `--data-path data/weather_data.csv`

### `--output-dir` (str)
- 默认：`./Plot_v2`
- 作用：所有输出（png 图、csv 指标、预测序列）保存目录。
- 输出文件（常见）：
  - `test_metrics.csv`
  - `test_predictions_flat.csv`
  - `test_pred_vs_true_*.png`
  - `24h_PVPower.png`
  - `forecast_24h_predictions.csv`

### （可选）调试 / 中间产物

#### `--print-fixed-P` (flag)
- 默认：False
- 作用：在控制台打印“全量历史统计”的固定转移矩阵 `P_fixed`（用于对照滚动 `P_t`）。

#### `--save-markov` (flag)
- 默认：False
- 作用：将“每个时间点的滚动转移矩阵”序列 `P_t` 保存到输出目录，便于复盘/绘图。

#### `--markov-save-format` (str)
- 默认：`csv`
- 可选：`csv`, `npz`, `both`
- 作用：控制 `P_t` 的保存格式（仅在 `--save-markov` 启用时生效）。

---

## 2) 列配置（数据列名）

### `--target-col` (str)
- 默认：`Power`
- 作用：预测目标列名（LSTM 的 y）。
- 示例：
  - `--target-col pv_power`

### `--feature-cols` (list[str])
- 默认：
  - `Wind_Speed`
  - `Weather_Temperature_Celsius`
  - `Weather_Relative_Humidity`
  - `Global_Horizontal_Radiation`
  - `Diffuse_Horizontal_Radiation`
  - `Radiation_Global_Tilted`
- 作用：
  1) KMeans 聚类输入特征（窗口内聚类）
  2) LSTM 输入特征（序列 x）
- 调参建议：
  - 想让“状态”更像天气：可只用辐照 + 湿度（必要时加温度）
  - 想让模型预测更准：可加入你认为对功率有解释力的气象变量
- 示例：只用辐照做聚类/建模
  - `--feature-cols Global_Horizontal_Radiation Diffuse_Horizontal_Radiation Radiation_Global_Tilted`

> 注意：如需“聚类用一组特征、LSTM 用另一组特征”，可以继续扩展参数（新增 `--cluster-cols`）。

---

## 3) 滑动窗口与天气状态标注（KMeans + cluster->S/C/R）

### `--window-days` (int)
- 默认：7
- 作用：每个窗口包含多少天的数据，用于 KMeans 并产出一个窗口标签。
- 经验：
  - 7~14 天常见；太大窗口会把多种天气混在一起、转移矩阵更平滑

### `--step-days` (int)
- 默认：1
- 作用：窗口滑动步长（按天）。
- 经验：
  - 1：每天一个窗口标签（更细）
  - >1：更稀疏、转移矩阵更平滑，但细节少

### `--kmeans-clusters` (int)
- 默认：3
- 作用：KMeans 簇数。
- 说明：当前脚本默认按 3 类映射到 S/C/R。若改为 !=3，需要你扩展映射逻辑。

### `--kmeans-random-state` (int)
- 默认：42
- 作用：KMeans 随机种子，保证可复现。

### `--cluster-label-mode` (str)
- 默认：`score_day`
- 可选：`ghi`, `ghi_day`, `power_day`, `score_day`
- 作用：把 KMeans 的 cluster 排序后映射到 S/C/R。

#### 为什么需要 day 版本？
- 夜间的 GHI/功率几乎都很小，如果直接用全窗口的 GHI 均值排序，可能把“夜间占比大”的 cluster 误判为 R（雨）。
- `*_day` 会先用“白天掩码”过滤夜间，再统计 cluster 得分。

#### 各模式含义
- `ghi`：用全窗口 GHI mean 排序（容易受夜间影响）
- `ghi_day`：仅白天样本的 GHI mean 排序
- `power_day`：仅白天样本的 Power mean 排序（更贴近预测任务）
- `score_day`：白天样本下，多指标线性打分排序（推荐）

### `--day-ghi-thr` (float)
- 默认：10.0
- 作用：白天掩码阈值，`GHI >= thr` 视为白天。
- 建议范围：5 ~ 100（取决于数据单位/采样间隔）。
- 经验：
  - 5~20：更宽松
  - 50~100：更严格（更接近真正日照时段）

---

## 4) `score_day` 多指标打分权重（线性组合）

score_day 的核心：

`score = w_ghi*z(GHI) + w_dhi*z(DHI) + w_rh*z(RH) + w_ws*z(WindSpeed) + w_p*z(Power)`

其中 `z(*)` 是“在 cluster 维度做 z-score”（窗口内标准化）。

### `--score-w-ghi` (float)
- 默认：+1.0
- 作用：更偏向高辐照 => 更像 S。

### `--score-w-rh` (float)
- 默认：-1.0
- 作用：更偏向低湿度 => 更像 S；高湿度惩罚。

### `--score-w-dhi` (float)
- 默认：-0.3
- 作用：DHI 通常在多云/散射占比高时提高（具体要看你的站点/传感器），给负权用于偏向“直射强”更像晴。
- 调参建议：如果你发现多云反而 DHI 更高但你想把它归为 C，可尝试把它调得更负或更正（看数据分布）。

### `--score-w-ws` (float)
- 默认：0.0
- 作用：风速权重。一般不建议一开始就加大（先 0）。

### `--score-w-p` (float)
- 默认：+0.2
- 作用：把“发电强”作为一点辅助信号（更任务相关）。
- 注意：Power 也会受到设备/遮挡影响，权重不要过大。

---

## 5) 序列建模参数（lookback / horizon）

### `--lookback` (int)
- 默认：72
- 作用：输入序列长度（过去多少个点作为 x）。
- 你的数据若是 5min 采样：72 = 6 小时。
- 经验：
  - 太小：无法覆盖足够历史
  - 太大：训练更慢、对小数据更容易过拟合

### `--horizon` (int)
- 默认：288
- 作用：预测步长（输出长度）。
- 你的数据若 5min：288 = 24 小时。
- 注意：如果你的采样间隔不是 5min，需要按采样频率换算（例如 15min 采样，24h 应为 96）。

---

## 6) 训练/测试切分与 DataLoader

### `--test-ratio` (float)
- 默认：0.2
- 作用：按时间切分，后 20% 做测试。

### `--batch-size` (int)
- 默认：32
- 作用：训练 batch。

### `--test-batch-size` (int)
- 默认：64
- 作用：测试 batch（脚本会强制 `shuffle=False` 以保证时间对齐）。

---

## 7) LSTM 模型结构

### `--hidden` (int)
- 默认：64
- 作用：LSTM hidden size。
- 经验：32/64/128 常见。

### `--num-layers` (int)
- 默认：3
- 作用：LSTM 层数。
- 经验：2~3 常见，过深不一定更好。

### `--dropout` (float)
- 默认：0.2
- 作用：LSTM dropout（仅当 num_layers>1 生效）。

---

## 8) 训练超参数

### `--epochs` (int)
- 默认：100
- 作用：每个标签模型训练轮数。

### `--lr` (float)
- 默认：1e-3
- 作用：Adam 学习率。

---

## 9) Markov / 集成（固定或滚动 P_t）

### `--pi` (3 floats)
- 默认：`[1,0,0]`
- 作用：初始状态分布（S/C/R）。
- 说明：脚本用 Markov 矩阵递推得到权重，再对三模型输出加权。

### `--markov-window` (int)
- 默认：30
- 作用：估计时间变化 Markov 矩阵 P_t 的滚动窗口长度（按 window_labels 个数）。
- 规则：
  - 0：使用全量历史（退化成固定 P）
  - >0：只用最近 N 个 window_labels 估计当前 P_t

#### 推荐怎么设置（经验法）
`markov-window` 的本质是在“稳定(更平滑) vs 灵敏(更跟随近期)”之间取舍：

- **数据比较平稳、你更看重鲁棒性**：
  - 推荐：`0`（固定 P）或 `60~200`
- **天气/季节变化明显、你希望权重能跟随近期状态变化**：
  - 推荐：`15~60`

更具体一点的选法（按你的窗口定义换算）：
- `window_labels` 的间隔约等于 `step-days` 天。
- 因此：
  - `markov-window=30` 约等于用最近 `30 * step-days` 天的“窗口状态序列”来估计 P_t。
  - 如果 `step-days=1`，那就是最近 30 天。
  - 如果 `step-days=3`，那就是最近 90 天。

#### 一个简单可用的默认推荐
- `step-days=1`：优先从 `--markov-window 30` 开始试
- `step-days>1`：可以用 `--markov-window 10~30`（因为每一步覆盖更多天）

#### 注意事项 / 边界
- `markov-window` 太小（比如 <10）会导致转移统计很稀疏、P_t 抖动大；建议同时把 `--markov-min-transitions` 设大一点（如 5~20），让稀疏时自动回退到固定 P。
- `markov-window` 太大（接近全量）会退化成固定 P，不能反映近期变化。

#### 示例
```bash
# 近 1 个月（step-days=1）
python windowkmeans_v2.py --markov-window 30

# 近 2 个月（step-days=1）
python windowkmeans_v2.py --markov-window 60

# 稀疏滑窗（每 3 天一个 window label），近 30*3=90 天
python windowkmeans_v2.py --step-days 3 --markov-window 30
```

### `--markov-min-transitions` (int)
- 默认：1
- 作用：滚动窗口内转移数太少时，退回固定 P（更稳定）。

---

## 10) 可视化

### `--max-plot-points` (int)
- 默认：800
- 作用：把 (N,horizon) 展平绘图时的最大点数（否则图太大）。

### （v2）测试集只画 24h 窗口（推荐）

#### `--plot-window-hours` (float)
- 默认：24.0
- 作用：测试集可视化仅截取指定小时数的片段绘图。
- 规则：
  - `>0`：只画该时长（例如 24h）
  - `<=0`：画整个测试集（不推荐，图过大且不直观）

#### `--plot-window-mode` (str)
- 默认：`end`
- 可选：`end`, `start`, `random`
- 作用：当只画窗口时，窗口从测试集哪里取。

#### `--plot-window-start` (str)
- 默认：None
- 作用：手动指定窗口起点时间（例如 `2023-07-01 00:00:00`）。
- 说明：如果提供该参数，会优先于 `--plot-window-mode`。

#### `--plot-window-seed` (int)
- 默认：42
- 作用：`plot-window-mode=random` 时用于可复现抽取。

### `--no-show` (flag)
- 默认：False（不传则会弹出窗口）
- 作用：不弹窗，只保存 png。

---

## 11) 推荐的起手命令（示例）

### (A) 推荐：白天 + 多指标 score_day
```bash
python windowkmeans_v2.py \
  --cluster-label-mode score_day \
  --day-ghi-thr 20 \
  --score-w-ghi 1.0 --score-w-rh -1.2 --score-w-dhi -0.2 --score-w-p 0.2 \
  --markov-window 30 \
  --epochs 50 \
  --save-markov --markov-save-format csv \
  --no-show
```

### (B) 更任务导向：按白天 Power 映射
```bash
python windowkmeans_v2.py --cluster-label-mode power_day --day-ghi-thr 20 --no-show
```

### (C) 画测试集末尾 24h 对比图（默认就画 24h）
```bash
python windowkmeans_v2.py --plot-window-mode end --plot-window-hours 24 --no-show
```

---

## 12) 常见坑位

1. **窗口标签覆盖范围**：窗口标签数量约为 `(n_days-window_days)//step_days + 1`。
2. **horizon 与“24h”换算**：v2 默认 `horizon=288` 对应 5min 采样的 24h；如果采样间隔变化，需要你自己换算。
3. **类别样本不足**：某个 label 的数据行数不足 `lookback+horizon` 时，`PVSeqDataset` 长度为 0，会跳过训练/评估。
4. **score_day 权重**：不要把 `--score-w-p` 调太大，否则状态会被“设备异常导致低功率”影响。

---

## windowkmeans_v2.py（新版、带滚动 Markov P_t + 时变权重集成）

新版脚本的运行逻辑与参数说明见：`README_windowkmeans_v2.md`

- 重点特性：
  - 白天掩码/日照时段（避免夜间 GHI 过小导致误归类）
  - 多指标打分映射 cluster→S/C/R（`--cluster-label-mode score_day`）
  - 滚动更新 Markov 转移矩阵 `P_t`（`--markov-window`）
  - 测试阶段按样本时间滚动更新权重并可视化
  - 推理阶段“多步递推权重 + 多步加权输出”并保存 24h CSV
