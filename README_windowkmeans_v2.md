# windowkmeans_v2.py 说明文档（运行逻辑 + 参数指南）

本文档解释脚本 `windowkmeans_v2.py` 的整体运行逻辑、关键模块、参数含义，以及输出文件/图的含义，方便你后续自己调参和扩展。

> 目的：
> - 用 **滑动窗口 KMeans** 给时间段打上天气状态标签（S/C/R）
> - 分状态训练 3 个 LSTM（S、C、R 三个“专家模型”）
> - 用 **Markov 转移矩阵** 建模天气状态随时间的变化，并在测试/推理阶段按权重对三模型输出做 **加权集成**
> - 输出评估指标、预测 CSV、可视化图，以及可选保存所有滚动转移矩阵 `P_t`

---

## 1. 输入/输出契约（你需要准备什么）

### 1.1 输入数据格式
脚本默认读取：`--data-path data/91-Site_DKA-M9_B-Phase.csv`

要求 CSV 中至少包含：
- `timestamp`：可被 `pandas.to_datetime` 解析的时间戳列
- `Power`（可通过 `--target-col` 修改）：预测目标
- 特征列（可通过 `--feature-cols` 修改），默认 6 个：
  - `Wind_Speed`
  - `Weather_Temperature_Celsius`
  - `Weather_Relative_Humidity`
  - `Global_Horizontal_Radiation` (GHI)
  - `Diffuse_Horizontal_Radiation` (DHI)
  - `Radiation_Global_Tilted`

脚本内部会额外生成：
- `date = timestamp.dt.date`
- `window_label`：每一天（或每个时间点）归属的天气状态标签 `S/C/R`

### 1.2 输出目录
由 `--output-dir` 指定（默认 `./Plot_v2`），脚本会在里面写入：
- 测试评估指标 CSV
- 测试集预测序列 CSV（flatten）
- 推理阶段 24h 预测 CSV
- 多张 PNG 图
- （可选）保存所有滚动 Markov 转移矩阵 `P_t` 的文件

---

## 2. 总体流程（main 发生了什么）

`main()` 的执行顺序可以概括为：

1. **解析参数**：`args = parse_args()`
2. **初始化全局配置**：`init_globals(args)`（设置 `device`、`feature_cols`、`target_col`、`lookback`、`horizon`）
3. **读取并排序数据**：`df = load_data(args.data_path)`
4. **滑窗聚类 → 生成窗口标签**：`window_labels = compute_window_labels(df, args)`
5. **构造固定 Markov 矩阵**：`P_fixed = build_markov(window_labels)`（用于对比与回退）
6. **构造滚动 Markov 序列**：`Ps = build_markov_timevarying(window_labels, ...)` 得到 `P_t`
7. （可选）**保存所有 P_t 到文件**：`--save-markov`
8. **由 P_t 递推得到权重序列**：`pi_series = build_pi_series(Ps, pi0)`
9. **建立 day→window 索引映射**：`day_to_win = _compute_day_to_window_index(...)`
10. **把标签赋回 df**：`df = assign_window_labels_to_df(df, window_labels)`
11. **按时间切分 train/test，并按 S/C/R 分桶成 Dataset/Loader**：`build_datasets_and_loaders()`
12. **构建 3 个模型并训练**：`build_models()` + `train_models()`
13. **评估 + 可视化 + 保存 CSV**：`evaluate_and_plot()`
14. **推理：对最新时刻做一次 24h 预测并输出 CSV**：`forecast_one()`

---

## 3. 数据与标签：滑动窗口 KMeans 是怎么做的

### 3.1 关键概念：窗口（window_labels）
`compute_window_labels()` 会先取所有日期：
- `unique_days = sorted(df['date'].unique())`

然后按如下方式滑窗：
- 每个窗口覆盖 `--window-days` 天（默认 7 天）
- 窗口起点每次滑动 `--step-days` 天（默认 1 天）

因此窗口数量大约是：

```
N_windows ≈ (N_days - window_days) / step_days + 1
```

你之前看到“窗口数量是 660”，通常意味着数据里大约有 666~667 天（日粒度），因为默认 `window_days=7`。

### 3.2 每个窗口内做 KMeans 的输入特征
聚类输入就是 `--feature-cols` 指定的列（默认 6 个气象/辐照特征）：
- Wind_Speed
- Temperature
- Relative Humidity
- GHI
- DHI
- Global Tilted

每个窗口内：
1. 取出该窗口对应的所有行 `window_df`
2. `StandardScaler().fit_transform(X)` 标准化
3. `KMeans(n_clusters=--kmeans-clusters)` 聚类

注意：这是 **每个窗口单独 fit 一次 KMeans**（不是全局 fit 一次），因此不同窗口的 cluster id（0/1/2）并不天然对应同一个物理含义。

### 3.3 cluster → S/C/R 的映射（解决“GHI夜间很小会像雨天”）
因为夜间 GHI 本来就接近 0，单纯按“全时段平均 GHI”会把“夜间多”的窗口误判为差天气。

脚本提供了 `--cluster-label-mode` 来做 cluster→S/C/R 的映射：

- `ghi`：按整个窗口的 cluster 平均 GHI 排序（最亮=晴 S）
- `ghi_day`：只用白天样本（day mask）计算平均 GHI
- `power_day`：只用白天样本计算平均 Power
- `score_day`（默认、推荐）：白天样本上做 **多指标线性打分** 再排序

#### 白天掩码（day mask）
使用 `GHI >= --day-ghi-thr`（默认 10 W/m^2）认为是白天。
- 若窗口里完全没有白天（极端情况），则回退使用全窗口。

#### score_day 的打分公式
在白天样本上，对每个 cluster 统计均值并做窗口内 z-score，然后线性组合：

```
score = w_ghi*z(GHI) + w_dhi*z(DHI) + w_rh*z(RH) + w_ws*z(WS) + w_p*z(Power)
```

默认权重：
- `--score-w-ghi=+1.0`：GHI 越高越像晴
- `--score-w-rh=-1.0`：湿度越高越像差天气
- `--score-w-dhi=-0.3`：DHI 在某些天气下会升高，这里给负权重（可调）
- `--score-w-ws=0.0`：默认不使用风速
- `--score-w-p=+0.2`：功率高通常意味着更晴（少量正权重）

映射方式：
- 将 cluster 按 `score` 从高到低排序
- top1 → `S`（晴），top2 → `C`（多云），top3 → `R`（雨/差天气）

### 3.4 将 window_labels 分配回每一天（window_label）
`assign_window_labels_to_df()` 目前的逻辑是（示意）：

- for day, lab in zip(unique_days[:len(window_labels)], window_labels):
  - df.loc[df['date'] == day, 'window_label'] = lab

也就是：
- `window_labels[t]` 被赋给第 `t` 天

这是一种简单对齐方式，含义是“用窗口 t 的主导天气代表第 t 天”。

> 注意：如果你希望更严格地“每一天用覆盖它的窗口”或“用最近窗口起点”，需要改写这段映射规则。

---

## 4. Markov：固定 P vs 滚动 P_t（随时间更新）

### 4.1 固定转移矩阵 P_fixed
`build_markov(window_labels)`：
- 状态映射：`S=0, C=1, R=2`
- 统计相邻标签转移次数 N[i,j]
- +1 拉普拉斯平滑（避免 0 概率）
- 行归一化得到 `P_fixed`（每行和为 1）

这一矩阵是“全历史统计”的，**不随时间变化**。

### 4.2 滚动转移矩阵序列 P_t
`build_markov_timevarying(window_labels, window_size, min_transitions)`：
- 目标：对每个时间步 t 估计一个 `P_t`
- 如果 `--markov-window>0`：用 `labels[max(0,t-window+1):t+1]` 的局部历史估计
- 如果 `--markov-window=0`：用全量历史（等价固定）
- 若局部历史内转移数 `< --markov-min-transitions`：回退到 `P_fixed`（保证稳定）

返回：
- `Ps`: list[np.ndarray]，长度 = `len(window_labels)`
- `Ps[t]` 表示时刻 t 的转移矩阵估计（用于从 t 推到 t+1）

### 4.3 为什么控制台只打印最后一个 P_t？
脚本在 main 里：
- 为了避免刷屏，只展示最后一个 `P_last = Ps[-1]`
- 但如果加 `--save-markov`，会把全部 `P_t` 保存到文件（详见“输出”一节）

### 4.4 权重 pi(t) 的递推（并确保权重和为 1）
`build_pi_series(Ps, pi0)` 用递推：

```
pi_{t+1} = P_t @ pi_t
```

注意：
- `P_t` 是按“from-state 行归一”的行随机矩阵
- 这里用列向量递推 `P@pi`
- 递推每一步做 `sum=1` 的归一化，避免数值漂移导致权重和≠1

---

## 5. 三模型训练：S/C/R 分桶 + LSTM

### 5.1 Dataset 的核心切片
`PVSeqDataset(df, label)` 会筛选：
- `df[df['window_label']==label]`

每个样本：
- 输入 `X`: shape `(lookback, n_features)`
- 输出 `Y`: shape `(horizon,)`

其中：
- `--lookback` 默认 72（例如 5min 间隔对应 6h）
- `--horizon` 默认 288（5min 间隔对应 24h）

### 5.2 模型结构 PVLSTM
- 多层 LSTM（`--num-layers`，默认 3）
- hidden size：`--hidden`（默认 64）
- dropout：`--dropout`（默认 0.2）
- 取 LSTM 最后一个时间步输出 `out[:, -1, :]` 过 MLP 输出 horizon 维

### 5.3 训练与跳过逻辑
如果某个 label 的数据不足以构造样本（长度<=0），会跳过训练并打印：

```
跳过训练 X 模型：该类别样本不足 (lookback+horizon=...)
```

---

## 6. 测试评估：单模型 + 时变权重集成（并保存 CSV/图）

`evaluate_and_plot()` 做两类评估：

### 6.1 单模型评估
对每个 label 的测试集桶（S-test/C-test/R-test）：
- 收集 `(y_true, y_pred)`
- 计算指标：`MAE / RMSE / R2`
- 保存预测 flatten 到 CSV
- 画图 `test_pred_vs_true_{lab}.png`

### 6.2 集成评估（重点：即便测试集来自某一类，也会用三个模型）
你问过：
> 测试集数据来自 C 那预测为什么还会用三个模型？

原因：
- 这里的思想是“专家模型集成”：每个时刻真实天气可能并不严格等于桶标签，或者标签本身是窗口主导状态
- 以及你要求的“滚动 Markov + 滚动权重”本质是让输出对天气切换更鲁棒

因此，在 `tag=ensemble_timevarying` 时：
- 对同一个输入 X，同时算 `pred_S, pred_C, pred_R`
- 对每个样本按其时间对应的 `pi_series[t]` 生成权重 `w=[wS,wC,wR]`
- 输出：

```
pred = wS*pred_S + wC*pred_C + wR*pred_R
```

并计算 R2 等指标，画图：
- `test_pred_vs_true_ensemble_timevarying_on_C.png`

这张图的含义是：
- “在 C 类测试集桶上，用时变权重集成后的预测 vs 真实”

### 6.3 测试集只画 24 小时
脚本通过 `_select_24h_slice()` 将 flatten 后的序列截取一段（默认 24h 对应 `horizon` 步数）用于画图。
你可以用：
- `--plot-window-hours 24`（默认）
- `--plot-window-mode end/start/random`

---

## 7. 推理阶段：多步递推权重 + 多步加权输出（24h）

`forecast_one()` 会对“最新时刻”做一次 24h 预测：

1. 找到一个可用的 bucket 来取出最后一个样本 `x_now`
   - 优先用最新 `window_label`
   - 如果该类 bucket 为空，则回退找任意非空 bucket
2. 三模型输出 **整段 horizon** 的预测：`pred_S, pred_C, pred_R`（各为 H=288）
3. 取 `P_last`（main 中保存的最后一个滚动矩阵；若无则用 `P_fixed`）
4. 从 `pi_next` 作为起点，做 **多步递推** 得到未来每一步权重：

```
pi_{k+1} = P_last @ pi_k
w_series[k] = pi_k
```

5. **逐步加权输出**：

```
pred_ensemble[k] = wS[k]*pred_S[k] + wC[k]*pred_C[k] + wR[k]*pred_R[k]
```

6. 输出图：`24h_PVPower.png`
7. 输出 CSV：`forecast_24h_predictions.csv`

> 推理阶段为什么权重和可能不是 1？
> - 脚本里在递推时做了归一化（应该接近 1）
> - 若你看到明显不为 1，常见原因是：读取/后处理时的浮点格式显示，或某处绕过了归一化（可以检查输出 CSV 的 w_S+w_C+w_R）。

---

## 8. 输出文件说明（你会看到什么）

### 8.1 CSV
- `test_metrics.csv`
  - 每行：`tag`（single/ensemble_timevarying/ensemble_fixed）、`label`（S/C/R）、MAE、RMSE、R2、n_samples
- `test_predictions_flat.csv`
  - 展平后的序列，用于后处理或画更复杂的图
  - 列：`tag,label,t,y_true,y_pred`
- `forecast_24h_predictions.csv`
  - 推理阶段未来 24h
  - 列：
    - `k`：步数索引（0..H-1）
    - `w_S,w_C,w_R`：每一步的集成权重
    - `pred_S,pred_C,pred_R`：三模型各自输出
    - `pred_ensemble`：逐步加权后的最终预测

### 8.2 PNG
- `test_pred_vs_true_S.png` / `C.png` / `R.png`
- `test_pred_vs_true_ensemble_timevarying_on_S.png` / `_on_C.png` / `_on_R.png`
  - “on_X” 表示：在 X 类测试桶上评估/绘制
- `24h_PVPower.png`
  - 推理阶段 24h 的加权预测曲线

### 8.3 Markov 文件（可选）
当开启 `--save-markov`：
- `markov_transitions.csv`
  - 行格式：`t, from, to, p`
  - 表示第 t 个窗口对应的 `P_t[from,to]`
- `markov_transitions.npz`
  - `Ps`：shape `(T,3,3)` 的数组

控制台仍然只打印最后一个 `P_last`，但文件里会保存全量。

---

## 9. 参数说明（按模块整理，便于调参）

### 9.1 IO
- `--data-path`：输入 CSV 路径
- `--output-dir`：输出目录

### 9.2 特征与目标
- `--target-col`：预测目标列名（默认 `Power`）
- `--feature-cols`：输入特征列列表（默认 6 列）

### 9.3 聚类与窗口
- `--window-days`：每个滑窗覆盖多少天（默认 7）
- `--step-days`：窗口起点滑动步长（默认 1 天）
- `--kmeans-clusters`：聚类数（默认 3，对应 S/C/R）
- `--kmeans-random-state`：KMeans 随机种子

### 9.4 cluster→S/C/R 映射（重要）
- `--cluster-label-mode`：`ghi/ghi_day/power_day/score_day`（默认 `score_day`）
- `--day-ghi-thr`：白天阈值（默认 10 W/m^2）
- `--score-w-ghi/--score-w-dhi/--score-w-rh/--score-w-ws/--score-w-p`：score_day 的权重

调参建议：
- 夜间误判明显时：优先用 `score_day` 或至少 `ghi_day`
- 如果“多云/雨天”边界不稳：增大 `|w_rh|` 或降低 `w_p`

### 9.5 序列建模
- `--lookback`：输入窗口长度
- `--horizon`：预测长度（默认 288 即 24h@5min）

### 9.6 训练
- `--epochs`：训练轮数
- `--lr`：学习率
- `--batch-size`：训练 batch
- `--test-batch-size`：测试 batch（测试必须 `shuffle=False` 才能对齐时间）

### 9.7 模型结构
- `--hidden`：LSTM hidden size
- `--num-layers`：层数
- `--dropout`：dropout

### 9.8 Markov / 集成
- `--pi a b c`：初始状态分布（默认 `[1,0,0]`）
- `--markov-window`：滚动估计窗口长度（单位：window_labels 的步数）
  - `0` 表示使用全历史（退化成固定矩阵）
- `--markov-min-transitions`：局部窗口内最少转移数，太少就回退到固定矩阵

> `--markov-window` 推荐：
> - 数据很长且季节性明显：可尝试 14 / 30 / 60
> - 变化快、想更灵敏：更小（例如 7~14）但更不稳定
> - 你现在默认是 30，是一个比较稳妥的折中

### 9.9 画图与保存
- `--plot-window-hours`：测试图只画多少小时（默认 24）
- `--plot-window-mode`：从 end/start/random 截取
- `--max-plot-points`：画图最大点数（避免太密/太慢）
- `--no-show`：不弹窗

### 9.10 调试/导出 Markov
- `--print-fixed-P`：打印固定矩阵
- `--save-markov`：保存所有 `P_t`
- `--markov-save-format csv|npz|both`

---

## 10. 常见问题（FAQ）

### Q1: 报错 “Expected sequence length to be larger than 0 in RNN”
这通常表示输入 `x_now` 的形状是 `[B, 0, F]`，也就是序列长度为 0。
常见原因：某个 bucket（S/C/R）数据太少，导致 `PVSeqDataset.__len__()==0`。

脚本现在在推理阶段做了“候选 bucket 回退”以尽量避免该问题，但如果三类都不足 `lookback+horizon` 行，仍会报错。

### Q2: 为什么测试集来自 C 但还会用三个模型？
因为这是“专家集成”设计：
- 标签是窗口主导状态，不保证每个样本都纯粹属于该类；
- 天气会切换；
- Markov 权重提供更鲁棒的混合输出。

### Q3: 为什么控制台还显示“固定转移矩阵”？
- `P_fixed` 始终会被计算，用于回退与对比。
- 若你加了 `--print-fixed-P` 才会打印它。
- 控制台默认打印的是最后一个滚动矩阵 `P_last`。

### Q4: 为什么只展示最后一个 P_t？
为了不刷屏。你可以用 `--save-markov` 保存全部 `P_t` 到文件。

### Q5: 我的推理 CSV 里 w_S+w_C+w_R 为什么不是 1？
理论上每一步都做了归一化，和应当非常接近 1。
如果偏差明显：
- 检查是否用外部工具做了截断/格式化
- 或检查是否有修改过 `forecast_one()` 的 normalize

---

## 11. 典型运行后的控制台输出长什么样（示例）

不同数据会不同，但一般会包含这些段落：

- 设备：
  - `Using device: cuda` 或 `Using device: cpu`
- 窗口数量：
  - `窗口数量: 660`
- Markov：
  - `滚动转移矩阵已启用: markov_window=30, min_transitions=1; 展示最后一个 P_t (P_last)=...`
- Train/Test 样本数：
  - `[Train] S samples: ...` 等
- 训练日志：
  - `训练 S 模型` + `Epoch i, loss=...`
- 测试评估：
  - `[S] MAE=..., RMSE=..., R2=...`
  - `[ensemble_timevarying on C-test] MAE=..., RMSE=..., R2=...`
- CSV 输出：
  - `[CSV] 已保存指标: .../test_metrics.csv`
  - `[CSV] 已保存预测序列: .../test_predictions_flat.csv`
  - `[CSV] 已保存推理阶段(24h)预测: .../forecast_24h_predictions.csv`

---

## 12. 你后续可能想改的点（扩展方向）

- 更严谨的“窗口标签 → 每天标签”对齐方式（目前是简单 zip）
- 推理阶段使用“未来 P_t”而不是固定 `P_last`：
  - 如果你能先预测未来天气状态序列，可以用非齐次 Markov 或按预测状态选择转移。
- 将“聚类 + 映射”改为全局模型（比如 GMM/HMM）以减少跨窗口 cluster id 不一致问题。

---

如你愿意，我也可以：
- 把“窗口标签赋值策略”改成更合理的（例如每一天用最近窗口起点或多数投票），并验证输出一致性；
- 在 `forecast_24h_predictions.csv` 里额外输出 `w_sum` 方便排查权重和是否为 1。
