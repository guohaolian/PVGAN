# =====================================================
#  PyTorch 24h 光伏功率预测（OOM-safe 版本）
# =====================================================

import argparse
import os

from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ================== 0. 全局配置（供 Dataset / 评估辅助函数使用） ==================

# NOTE: 为了保持 PVSeqDataset 的实现简洁，这里仍使用少量 module-level 配置。
# 这些值会在 main() 一开始由 args 初始化。

device: Optional[torch.device] = None
feature_cols: Optional[list[str]] = None
target_col: Optional[str] = None
lookback: Optional[int] = None
horizon: Optional[int] = None


def time_split_df(df_in: pd.DataFrame, test_ratio: float = 0.2):
    """按时间切分（不打乱），返回 train_df, test_df。"""
    if test_ratio <= 0 or test_ratio >= 1:
        raise ValueError("test_ratio must be in (0, 1)")

    df_sorted = df_in.sort_values('timestamp').reset_index(drop=True)
    n = len(df_sorted)
    if n == 0:
        return df_sorted.copy(), df_sorted.copy()

    split = int(n * (1 - test_ratio))
    split = max(1, min(n - 1, split))
    return df_sorted.iloc[:split].copy(), df_sorted.iloc[split:].copy()


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """返回 MAE / RMSE / R2。输入 shape: (N, horizon) 或 (N,)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # R2 = 1 - SSE / SST
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    if ss_tot > 1e-12:
        r2 = 1.0 - np.divide(ss_res, ss_tot)
    else:
        r2 = np.nan

    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader):
    model.eval()
    ys, ps = [], []
    for X, Y in loader:
        X = X.to(device)
        pred = model(X).detach().cpu().numpy()
        ys.append(Y.numpy())
        ps.append(pred)

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    return calc_metrics(y_true, y_pred)


@torch.no_grad()
def evaluate_weighted_ensemble(pi_next_vec: np.ndarray, models: dict, loader: DataLoader):
    """按 pi_next_vec 对 S/C/R 三模型输出加权，并评估与真实 Y 的误差。"""
    for m in models.values():
        m.eval()

    ys, ps = [], []
    for X, Y in loader:
        X = X.to(device)

        pred_S = models['S'](X).detach().cpu().numpy()
        pred_C = models['C'](X).detach().cpu().numpy()
        pred_R = models['R'](X).detach().cpu().numpy()

        pred_final = (
            pi_next_vec[0] * pred_S +
            pi_next_vec[1] * pred_C +
            pi_next_vec[2] * pred_R
        )

        ys.append(Y.numpy())
        ps.append(pred_final)

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    return calc_metrics(y_true, y_pred)


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader):
    """收集整个 loader 的预测与真实，返回 (y_true, y_pred) shape: (N, horizon)。"""
    model.eval()
    ys, ps = [], []
    for X, Y in loader:
        X = X.to(device)
        pred = model(X).detach().cpu().numpy()
        ys.append(Y.numpy())
        ps.append(pred)
    return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)


@torch.no_grad()
def collect_weighted_predictions(pi_next_vec: np.ndarray, models: dict, loader: DataLoader):
    """固定权重（整段测试集同一 pi）加权集成预测。

    返回 (y_true, y_pred) 形状: (N, horizon)
    """
    for m in models.values():
        m.eval()

    ys, ps = [], []
    for X, Y in loader:
        X = X.to(device)

        pred_S = models['S'](X).detach().cpu().numpy()
        pred_C = models['C'](X).detach().cpu().numpy()
        pred_R = models['R'](X).detach().cpu().numpy()

        pred_final = (
            float(pi_next_vec[0]) * pred_S +
            float(pi_next_vec[1]) * pred_C +
            float(pi_next_vec[2]) * pred_R
        )

        ys.append(Y.numpy())
        ps.append(pred_final)

    return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)


def _compute_day_to_window_index(unique_days: list, window_days: int, step_days: int) -> dict:
    """把每个 day 映射到 window_labels 的索引（用于时间变化的 pi(t) 加权）。

    compute_window_labels():
      for i in range(0, n_days-window_days+1, step_days):  # i 是窗口起点 day_idx
         window_labels.append(label_of_window_i)

    一个具体的 day_idx=d，应该用哪个窗口索引 t 来代表它？这里采用“最近的窗口起点”策略：
      t = floor(d / step_days)
    再做边界裁剪到 [0, n_windows-1]。

    返回:
      dict[date] -> window_index (int). 若没有任何窗口，返回 -1。
    """
    n_days = len(unique_days)
    n_windows = max(0, (n_days - window_days) // step_days + 1)

    m = {}
    if n_windows <= 0:
        for day in unique_days:
            m[day] = -1
        return m

    for di, day in enumerate(unique_days):
        t = di // step_days
        if t < 0:
            t = 0
        if t >= n_windows:
            t = n_windows - 1
        m[day] = t

    return m


def build_pi_series(Ps: list[np.ndarray], pi0: np.ndarray):
    """根据一系列 P_t 滚动递推得到 pi(t)。

    递推: pi_{t+1} = P_t @ pi_t

    返回:
      pi_series: list[np.ndarray], len = len(Ps)
        其中 pi_series[t] 用作窗口索引 t 对应样本的权重（3,）。

    重要：
      - 本项目里 P 是按 "from-state 行归一" 得到的行随机矩阵（每行和为 1）。
      - 但我们递推用的是列向量公式 pi_{t+1}=P@pi_t。
      - 为避免数值上出现权重和≠1 的漂移，这里每一步都做一次归一化。
    """
    if len(Ps) == 0:
        return []

    def _normalize_pi(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).reshape(3)
        s = float(v.sum())
        if s > 0:
            return v / s
        return np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)

    cur = _normalize_pi(np.asarray(pi0, dtype=np.float64).reshape(3))
    series = [cur]

    # Ps[t] 用于从 t -> t+1
    for t in range(len(Ps) - 1):
        cur = _normalize_pi(Ps[t] @ cur)
        series.append(cur)

    return series


@torch.no_grad()
def collect_weighted_predictions_timevarying(
    models: dict,
    loader: DataLoader,
    pi_series: list[np.ndarray],
    sample_window_indices: np.ndarray,
):
    """随样本时间滚动更新权重的加权集成预测。

    参数:
      - pi_series[t]: (3,) 权重向量
      - sample_window_indices: (N,) 每个样本对应窗口索引 t（与 loader 顺序严格一致）

    返回:
      (y_true, y_pred) 形状: (N, horizon)
    """
    for m in models.values():
        m.eval()

    if len(pi_series) == 0:
        raise ValueError("pi_series is empty")

    idx_ptr = 0
    ys, ps = [], []

    for X, Y in loader:
        bsz = X.shape[0]
        X = X.to(device)

        pred_S = models['S'](X).detach().cpu().numpy()
        pred_C = models['C'](X).detach().cpu().numpy()
        pred_R = models['R'](X).detach().cpu().numpy()

        t_batch = sample_window_indices[idx_ptr: idx_ptr + bsz]
        idx_ptr += bsz

        # (B,3)
        w = np.stack([
            pi_series[int(max(0, min(len(pi_series) - 1, int(t))))] for t in t_batch
        ], axis=0).astype(np.float64)

        # (B,H)
        pred_final = (
            w[:, 0:1] * pred_S +
            w[:, 1:2] * pred_C +
            w[:, 2:3] * pred_R
        )

        ys.append(Y.numpy())
        ps.append(pred_final)

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)

    if y_true.shape[0] != sample_window_indices.shape[0]:
        raise RuntimeError(
            f"Mismatch: y_true N={y_true.shape[0]} but sample_window_indices N={sample_window_indices.shape[0]}. "
            "Ensure loader matches dataset order and shuffle=False."
        )

    return y_true, y_pred


def plot_pred_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: str | None = None,
    max_points: int = 600,
    show: bool = True,
):
    """把 (N, horizon) 展平成一条时间序列画预测 vs 真实。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_flat = y_true.reshape(-1)
    pred_flat = y_pred.reshape(-1)

    if true_flat.size == 0:
        return

    if true_flat.size > max_points:
        idx = np.linspace(0, true_flat.size - 1, num=max_points, dtype=int)
        true_flat = true_flat[idx]
        pred_flat = pred_flat[idx]

    plt.figure(figsize=(12, 4))
    plt.plot(true_flat, label='True', linewidth=1.5)
    plt.plot(pred_flat, label='Pred', linewidth=1.2)
    plt.title(title)
    plt.xlabel('Time step (flattened)')
    plt.ylabel('Power')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close()


def _select_24h_slice(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    hours: float,
    mode: str,
    horizon_steps: int,
    start_step: int | None = None,
    seed: int = 42,
):
    """选择用于可视化的一段连续窗口（默认 24h）。

    约定：本项目里 horizon 通常就是 24h 的步数（例如 5min => 288）。
    因此默认取 window_len = horizon_steps（不依赖原始时间戳，完全对齐模型输出）。

    参数:
      - y_true/y_pred: (N, H)
      - hours: <=0 表示不截断
      - mode: start/end/random
      - horizon_steps: H
      - start_step: 可选，强制从 flatten 后的某个 step 开始

    返回:
      (y_true_slice, y_pred_slice, suffix)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if hours is None or float(hours) <= 0:
        return y_true, y_pred, ""

    H = int(horizon_steps)
    if H <= 0:
        return y_true, y_pred, ""

    # 默认认为 H 就是 24h 的长度；如果你把 horizon 改成别的，我们也按比例缩放
    window_len = int(round(H * float(hours) / 24.0))
    window_len = max(1, window_len)

    true_flat = y_true.reshape(-1)
    pred_flat = y_pred.reshape(-1)
    n = min(true_flat.size, pred_flat.size)
    if n <= 0:
        return y_true[:0], y_pred[:0], ""

    if window_len >= n:
        return y_true, y_pred, f" (plot {n} steps)"

    if start_step is not None:
        s = int(start_step)
        s = max(0, min(n - window_len, s))
        mode_used = f"start={s}"
    else:
        mode = str(mode or "end").lower()
        if mode == "start":
            s = 0
            mode_used = "start"
        elif mode == "random":
            rng = np.random.default_rng(int(seed))
            s = int(rng.integers(0, n - window_len + 1))
            mode_used = f"random(seed={seed})"
        else:
            s = n - window_len
            mode_used = "end"

    e = s + window_len
    true_flat = true_flat[s:e]
    pred_flat = pred_flat[s:e]

    # reshape 回 (N', H) 仅为了复用 plot_pred_vs_true 的 flatten 逻辑；这里让 N'=1 更直观
    return true_flat.reshape(1, -1), pred_flat.reshape(1, -1), f" (plot {window_len} steps, {mode_used})"


def flatten_pred_true_to_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    tag: str,
):
    """把 (N, horizon) 的 true/pred 展平成一列序列并转成 DataFrame 便于保存 CSV。"""
    true_flat = np.asarray(y_true).reshape(-1)
    pred_flat = np.asarray(y_pred).reshape(-1)
    n = min(true_flat.size, pred_flat.size)
    if n == 0:
        return pd.DataFrame(columns=["tag", "label", "t", "y_true", "y_pred"])

    return pd.DataFrame(
        {
            "tag": [tag] * n,
            "label": [label] * n,
            "t": np.arange(n, dtype=int),
            "y_true": true_flat[:n],
            "y_pred": pred_flat[:n],
        }
    )


# ================== Dataset ==================

class PVSeqDataset(Dataset):
    def __init__(self, df, label):
        if feature_cols is None or target_col is None or lookback is None or horizon is None:
            raise RuntimeError("Globals not initialized. Call main() first.")

        self._feature_cols: list[str] = feature_cols
        self._target_col: str = target_col
        self._lookback: int = lookback
        self._horizon: int = horizon

        self.label = label
        self.df = (
            df[df['window_label']==label]
            .sort_values('timestamp')
            .reset_index(drop=True)
        )

        self.data = self.df[self._feature_cols + [self._target_col]].values.astype(np.float32)

    def __len__(self):
        n = len(self.data) - (self._lookback + self._horizon) + 1
        return max(0, n)

    def __getitem__(self, idx):
        n = len(self)
        if n <= 0:
            raise IndexError(
                f"Dataset '{self.label}' has 0 samples (len(data)={len(self.data)}, lookback={self._lookback}, horizon={self._horizon})."
            )

        if idx < 0:
            idx = n + idx

        if idx < 0 or idx >= n:
            raise IndexError(f"Index {idx} out of range for dataset '{self.label}' with length {n}.")

        x = self.data[idx:idx+self._lookback, :-1]
        y = self.data[idx+self._lookback:idx+self._lookback+self._horizon, -1]

        if x.shape[0] != self._lookback or y.shape[0] != self._horizon:
            raise IndexError(
                f"Bad slice for dataset '{self.label}': idx={idx}, x.shape={x.shape}, y.shape={y.shape}, "
                f"len(data)={len(self.data)}"
            )

        return torch.from_numpy(x), torch.from_numpy(y)


# ================== Model ==================

# ---- 1) LSTM baseline（保留原实现） ----
class PVLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden=64, horizon=96, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, horizon)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ---- 2) Informer-like（Transformer encoder-only） ----
class PositionalEncoding(nn.Module):
    """标准正余弦位置编码（batch_first 版本）。"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10_000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return self.dropout(x)


class PVInformer(nn.Module):
    """Informer 风格（encoder-only）的时间序列预测模型。

    说明：这里实现的是“可直接替换 LSTM 的 Transformer/Informer-like forecaster”，
    保持与 PVLSTM 一致的接口：
      输入 x: (B, lookback, input_dim)
      输出 y_hat: (B, horizon)
    """

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 10_000,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=use_layer_norm,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)          # (B, L, D)
        z = self.pos_enc(z)             # (B, L, D)
        z = self.encoder(z)             # (B, L, D)
        z_last = z[:, -1, :]            # (B, D)
        return self.head(z_last)        # (B, H)


# 兼容旧名字：如果别的地方仍引用 PVLSTM，让它指向新的 Informer 实现
# PVLSTM = PVInformer


def train_one_epoch(model, loader, optimizer, criterion):
    if loader is None:
        return None

    model.train()
    loss_sum = 0.0

    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        # Transformer 类模型更容易梯度爆炸，做一次轻量的 clip，默认不改变原行为太多
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_sum += loss.item()

    return loss_sum / len(loader)


def build_argparser():
    p = argparse.ArgumentParser(description="PV forecasting with KMeans weather labels + per-state model (LSTM/Informer) + Markov ensemble")

    # IO
    p.add_argument("--data-path", type=str, default="data/91-Site_DKA-M9_B-Phase.csv")
    p.add_argument("--output-dir", type=str, default="./Plot_v2")

    # debug / logging
    p.add_argument(
        "--print-fixed-P",
        action="store_true",
        help="Print the global fixed Markov transition matrix P_fixed (mainly for debugging/comparison)",
    )
    p.add_argument(
        "--save-markov",
        default=True,
        action="store_true",
        help="Save the full rolling Markov transition sequence P_t to files under output-dir.",
    )
    p.add_argument(
        "--markov-save-format",
        type=str,
        default="csv",
        choices=["csv", "npz", "both"],
        help="Format for saving P_t sequence: csv (human-readable), npz (compact), or both.",
    )

    # columns
    p.add_argument("--target-col", type=str, default="Power")
    p.add_argument(
        "--feature-cols",
        type=str,
        nargs="+",
        default=[
            'Wind_Speed',
            'Weather_Temperature_Celsius',
            'Weather_Relative_Humidity',
            'Global_Horizontal_Radiation',
            'Diffuse_Horizontal_Radiation',
            'Radiation_Global_Tilted'
        ],
    )

    # -------- cluster -> S/C/R 映射：白天掩码 + 多指标打分 --------
    p.add_argument(
        "--cluster-label-mode",
        type=str,
        default="score_day",
        choices=["ghi", "ghi_day", "power_day", "score_day"],
        help=(
            "How to map KMeans clusters to S/C/R. "
            "ghi: rank by overall GHI mean; "
            "ghi_day: rank by GHI mean on daytime-only samples; "
            "power_day: rank by Power mean on daytime-only samples; "
            "score_day: multi-metric linear score on daytime-only samples (recommended)."
        ),
    )
    p.add_argument(
        "--day-ghi-thr",
        type=float,
        default=10.0,
        help="Daytime mask threshold for GHI (W/m^2). Samples with GHI>=thr are treated as daytime.",
    )
    # 多指标打分（score_day）：score = w_ghi*z(GHI) + w_dhi*z(DHI) + w_rh*z(RH) + w_ws*z(WindSpeed) + w_p*z(Power)
    # 建议：晴天 => 高GHI、低湿度；雨/差天气 => 低GHI、高湿度
    p.add_argument("--score-w-ghi", type=float, default=+1.0, help="weight for z-score(GHI) in score_day")
    p.add_argument("--score-w-dhi", type=float, default=-0.3, help="weight for z-score(DHI) in score_day")
    p.add_argument("--score-w-rh", type=float, default=-1.0, help="weight for z-score(RelativeHumidity) in score_day")
    p.add_argument("--score-w-ws", type=float, default=+0.0, help="weight for z-score(WindSpeed) in score_day")
    p.add_argument("--score-w-p", type=float, default=+0.2, help="weight for z-score(Power) in score_day")

    # sliding window / labeling
    p.add_argument("--window-days", type=int, default=7)
    p.add_argument("--step-days", type=int, default=1)
    p.add_argument("--kmeans-clusters", type=int, default=3)
    p.add_argument("--kmeans-random-state", type=int, default=42)

    # sequence config
    p.add_argument("--lookback", type=int, default=72)
    p.add_argument("--horizon", type=int, default=288)

    # split & dataloader
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--test-batch-size", type=int, default=64)

    # training
    p.add_argument("--epochs", type=int, default=110)
    p.add_argument("--lr", type=float, default=1e-3)

    # model selection
    p.add_argument(
        "--model",
        type=str,
        default="informer",
        choices=["lstm", "informer"],
        help="Choose which forecasting model to use: lstm or informer.",
    )

    # model (LSTM)
    p.add_argument("--hidden", type=int, default=64, help="LSTM hidden size")
    p.add_argument("--num-layers", type=int, default=3, help="LSTM number of layers")
    p.add_argument("--dropout", type=float, default=0.2)

    # model (Informer)
    p.add_argument("--d-model", type=int, default=128, help="Informer/Transformer model dimension")
    p.add_argument("--n-heads", type=int, default=4, help="number of attention heads")
    p.add_argument("--e-layers", type=int, default=2, help="number of encoder layers")
    p.add_argument("--d-ff", type=int, default=256, help="feed-forward hidden dimension")

    # markov / ensemble
    p.add_argument("--pi", type=float, nargs=3, default=[1.0, 0.0, 0.0], help="initial state distribution for ensemble")
    p.add_argument(
        "--markov-window",
        type=int,
        default=30,
        help="rolling window size (#window labels) for time-varying Markov matrix P_t; 0 => use all history (fixed P)",
    )
    p.add_argument(
        "--markov-min-transitions",
        type=int,
        default=1,
        help="minimum number of transitions required to trust rolling estimation; otherwise fall back to fixed P",
    )

    # plotting
    p.add_argument("--max-plot-points", type=int, default=800)
    p.add_argument("--no-show", action="store_true", help="do not pop up matplotlib windows")

    # test visualization window
    p.add_argument(
        "--plot-window-hours",
        type=float,
        default=24.0,
        help="Test visualization: only plot this many hours from the test set (<=0 means plot full test set).",
    )
    p.add_argument(
        "--plot-window-mode",
        type=str,
        default="end",
        choices=["end", "start", "random"],
        help="Which window to plot when --plot-window-hours>0.",
    )
    p.add_argument(
        "--plot-window-start",
        type=str,
        default=None,
        help="Optional explicit window start timestamp (parseable by pandas). Overrides --plot-window-mode.",
    )
    p.add_argument(
        "--plot-window-seed",
        type=int,
        default=42,
        help="Random seed for --plot-window-mode random.",
    )

    return p


def parse_args(argv: list[str] | None = None):
    """单独封装一层，方便以后从别的模块调用或做单元测试。"""
    return build_argparser().parse_args(argv)


# ================== 业务流程拆分（模块化） ==================


def init_globals(args):
    global device, feature_cols, target_col, lookback, horizon

    feature_cols = list(args.feature_cols)
    target_col = args.target_col
    lookback = int(args.lookback)
    horizon = int(args.horizon)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['date'] = df['timestamp'].dt.date
    return df


def compute_window_labels(df: pd.DataFrame, args) -> list[str]:
    unique_days = sorted(df['date'].unique())
    window_labels: list[str] = []

    # columns used for score_day
    col_ghi = 'Global_Horizontal_Radiation'
    col_dhi = 'Diffuse_Horizontal_Radiation'
    col_rh = 'Weather_Relative_Humidity'
    col_ws = 'Wind_Speed'

    for i in range(0, len(unique_days) - args.window_days + 1, args.step_days):
        days = unique_days[i: i + args.window_days]
        window_df = df[df['date'].isin(days)]

        X = window_df[feature_cols].values
        Xs = StandardScaler().fit_transform(X)

        kmeans = KMeans(n_clusters=args.kmeans_clusters, random_state=args.kmeans_random_state)
        labels = kmeans.fit_predict(Xs)

        window_df = window_df.copy()
        window_df['cluster'] = labels

        mode = str(getattr(args, 'cluster_label_mode', 'score_day'))

        # day mask
        have_ghi = col_ghi in window_df.columns
        if have_ghi:
            day_mask = window_df[col_ghi].astype(float) >= float(getattr(args, 'day_ghi_thr', 10.0))
        else:
            day_mask = pd.Series([True] * len(window_df), index=window_df.index)

        # 选择用于打分的数据子集：优先白天
        wdf_day = window_df.loc[day_mask] if day_mask.any() else window_df

        if mode == 'ghi':
            score = window_df.groupby('cluster')[col_ghi].mean()
        elif mode == 'ghi_day':
            score = wdf_day.groupby('cluster')[col_ghi].mean()
        elif mode == 'power_day':
            score = wdf_day.groupby('cluster')[target_col].mean()
        else:
            # score_day: 多指标线性打分（在白天样本上统计每个 cluster 的均值，然后做窗口内 z-score）
            # 1) 统计 cluster 的均值特征
            feats = {}
            if col_ghi in wdf_day.columns:
                feats['ghi'] = wdf_day.groupby('cluster')[col_ghi].mean()
            if col_dhi in wdf_day.columns:
                feats['dhi'] = wdf_day.groupby('cluster')[col_dhi].mean()
            if col_rh in wdf_day.columns:
                feats['rh'] = wdf_day.groupby('cluster')[col_rh].mean()
            if col_ws in wdf_day.columns:
                feats['ws'] = wdf_day.groupby('cluster')[col_ws].mean()
            if target_col in wdf_day.columns:
                feats['p'] = wdf_day.groupby('cluster')[target_col].mean()

            feat_df = pd.DataFrame(feats)

            # 若某些 cluster 在白天没有样本，会缺失行；补齐为 NaN，后续填充
            for c in range(int(args.kmeans_clusters)):
                if c not in feat_df.index:
                    feat_df.loc[c] = np.nan

            # 用全窗口（含夜间）该 cluster 的均值作为缺失回填，避免全 NaN
            fill_src = window_df.groupby('cluster').mean(numeric_only=True)
            for c in feat_df.index:
                if pd.isna(feat_df.loc[c]).all():
                    # 这个 cluster 全是夜间，回退用全量统计
                    if c in fill_src.index:
                        if 'ghi' in feat_df.columns and col_ghi in fill_src.columns:
                            feat_df.loc[c, 'ghi'] = fill_src.loc[c, col_ghi]
                        if 'dhi' in feat_df.columns and col_dhi in fill_src.columns:
                            feat_df.loc[c, 'dhi'] = fill_src.loc[c, col_dhi]
                        if 'rh' in feat_df.columns and col_rh in fill_src.columns:
                            feat_df.loc[c, 'rh'] = fill_src.loc[c, col_rh]
                        if 'ws' in feat_df.columns and col_ws in fill_src.columns:
                            feat_df.loc[c, 'ws'] = fill_src.loc[c, col_ws]
                        if 'p' in feat_df.columns and target_col in fill_src.columns:
                            feat_df.loc[c, 'p'] = fill_src.loc[c, target_col]

            # 2) 在 cluster 维度做 z-score（窗口内标准化，避免量纲影响）
            z = feat_df.copy()
            for col in z.columns:
                mu = np.nanmean(z[col].values.astype(float))
                sd = np.nanstd(z[col].values.astype(float))
                if sd < 1e-12:
                    z[col] = 0.0
                else:
                    z[col] = (z[col].astype(float) - mu) / sd

            # 3) 线性组合
            w_ghi = float(getattr(args, 'score_w_ghi', 1.0))
            w_dhi = float(getattr(args, 'score_w_dhi', -0.3))
            w_rh = float(getattr(args, 'score_w_rh', -1.0))
            w_ws = float(getattr(args, 'score_w_ws', 0.0))
            w_p = float(getattr(args, 'score_w_p', 0.2))

            score = pd.Series(0.0, index=z.index, dtype=float)
            if 'ghi' in z.columns:
                score = score + w_ghi * z['ghi']
            if 'dhi' in z.columns:
                score = score + w_dhi * z['dhi']
            if 'rh' in z.columns:
                score = score + w_rh * z['rh']
            if 'ws' in z.columns:
                score = score + w_ws * z['ws']
            if 'p' in z.columns:
                score = score + w_p * z['p']

        # 保证所有 cluster 都有 score
        for c in range(int(args.kmeans_clusters)):
            if c not in score.index:
                score.loc[c] = -1e12

        sorted_clusters = score.sort_values(ascending=False).index.tolist()

        # 默认 3 类 => S/C/R；如果你改了 kmeans-clusters，这里需要扩展映射。
        cluster_to_label = {
            sorted_clusters[0]: 'S',
            sorted_clusters[1]: 'C',
            sorted_clusters[2]: 'R'
        }

        window_df['weather_label'] = window_df['cluster'].map(cluster_to_label)
        dominant_label = window_df['weather_label'].value_counts().idxmax()
        window_labels.append(dominant_label)

    return window_labels


def assign_window_labels_to_df(df: pd.DataFrame, window_labels: list[str]) -> pd.DataFrame:
    unique_days = sorted(df['date'].unique())
    df = df.copy()
    df['window_label'] = None
    for day, lab in zip(unique_days[:len(window_labels)], window_labels):
        df.loc[df['date'] == day, 'window_label'] = lab
    return df


def build_markov(window_labels: list[str]):
    state_map = {'S': 0, 'C': 1, 'R': 2}
    N = np.zeros((3, 3))

    for t in range(len(window_labels) - 1):
        i = state_map[window_labels[t]]
        j = state_map[window_labels[t + 1]]
        N[i, j] += 1

    N += 1  # 拉普拉斯平滑
    P = N / N.sum(axis=1, keepdims=True)
    return P


def build_markov_timevarying(
    window_labels: list[str],
    window_size: int = 0,
    min_transitions: int = 1,
):
    """构造随时间滚动更新的转移矩阵序列 P_t。

    返回:
        Ps: list[np.ndarray], len= len(window_labels)
            Ps[t] 是使用 labels[max(0,t-window_size+1):t+1] 估计得到的 P_t。
            约定: P_t 用于从 state_t 推到 state_{t+1}。

    设计要点:
    - window_size=0 => 用全量历史（等价固定矩阵）
    - 若切片内转移数 < min_transitions，则回退为全量固定矩阵（保证稳定）
    - 内部使用拉普拉斯平滑，避免 0 概率
    """
    if len(window_labels) == 0:
        return []

    state_map = {'S': 0, 'C': 1, 'R': 2}

    def _safe_estimate(labels_slice: list[str], fallback_P: np.ndarray):
        if len(labels_slice) < 2:
            return fallback_P

        N = np.zeros((3, 3), dtype=np.float64)
        transitions = 0
        for i in range(len(labels_slice) - 1):
            a = labels_slice[i]
            b = labels_slice[i + 1]
            if a not in state_map or b not in state_map:
                continue
            N[state_map[a], state_map[b]] += 1.0
            transitions += 1

        if transitions < int(min_transitions):
            return fallback_P

        N += 1.0  # Laplace smoothing
        return N / N.sum(axis=1, keepdims=True)

    P_fixed = build_markov(window_labels)
    Ps: list[np.ndarray] = []

    for t in range(len(window_labels)):
        if window_size and window_size > 0:
            start = max(0, t - window_size + 1)
            labels_slice = window_labels[start: t + 1]
        else:
            labels_slice = window_labels

        Ps.append(_safe_estimate(labels_slice, fallback_P=P_fixed))

    return Ps


def build_datasets_and_loaders(df: pd.DataFrame, args):
    train_df, test_df = time_split_df(df, test_ratio=args.test_ratio)

    train_ds = {
        'S': PVSeqDataset(train_df, 'S'),
        'C': PVSeqDataset(train_df, 'C'),
        'R': PVSeqDataset(train_df, 'R'),
    }
    test_ds = {
        'S': PVSeqDataset(test_df, 'S'),
        'C': PVSeqDataset(test_df, 'C'),
        'R': PVSeqDataset(test_df, 'R'),
    }

    train_loader = {
        k: (DataLoader(v, batch_size=args.batch_size, shuffle=True) if len(v) > 0 else None)
        for k, v in train_ds.items()
    }
    test_loader = {
        # 关键：测试必须保持时间顺序，方便与滚动权重对齐
        k: (DataLoader(v, batch_size=args.test_batch_size, shuffle=False) if len(v) > 0 else None)
        for k, v in test_ds.items()
    }

    return train_ds, test_ds, train_loader, test_loader


def build_models(args):
    if feature_cols is None or horizon is None or device is None:
        raise RuntimeError("Globals not initialized. Call init_globals() first.")

    model_name = str(getattr(args, "model", "informer")).lower()

    if model_name == "lstm":
        models = {
            'S': PVLSTM(input_dim=len(feature_cols), hidden=int(args.hidden), horizon=int(horizon), num_layers=int(args.num_layers), dropout=float(args.dropout)).to(device),
            'C': PVLSTM(input_dim=len(feature_cols), hidden=int(args.hidden), horizon=int(horizon), num_layers=int(args.num_layers), dropout=float(args.dropout)).to(device),
            'R': PVLSTM(input_dim=len(feature_cols), hidden=int(args.hidden), horizon=int(horizon), num_layers=int(args.num_layers), dropout=float(args.dropout)).to(device),
        }
    else:
        # Informer/Transformer 超参
        d_model = int(getattr(args, "d_model", getattr(args, "d-model", 128)))
        n_heads = int(getattr(args, "n_heads", getattr(args, "n-heads", 4)))
        e_layers = int(getattr(args, "e_layers", getattr(args, "e-layers", 2)))
        d_ff = int(getattr(args, "d_ff", getattr(args, "d-ff", 256)))
        dropout = float(getattr(args, "dropout", 0.1))

        models = {
            'S': PVInformer(input_dim=len(feature_cols), horizon=int(horizon), d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff, dropout=dropout).to(device),
            'C': PVInformer(input_dim=len(feature_cols), horizon=int(horizon), d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff, dropout=dropout).to(device),
            'R': PVInformer(input_dim=len(feature_cols), horizon=int(horizon), d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff, dropout=dropout).to(device),
        }

    opts = {k: torch.optim.Adam(m.parameters(), lr=args.lr) for k, m in models.items()}
    return models, opts


def train_models(models: dict, train_loader: dict, opts: dict, args):
    if lookback is None or horizon is None:
        raise RuntimeError("Globals not initialized. Call init_globals() first.")

    criterion = nn.MSELoss()

    for name in ['S', 'C', 'R']:
        loader = train_loader[name]
        if loader is None:
            print(f"\n跳过训练 {name} 模型：该类别样本不足 (lookback+horizon={lookback + horizon})")
            continue

        print(f"\n训练 {name} 模型")
        for e in range(args.epochs):
            loss = train_one_epoch(models[name], loader, opts[name], criterion)
            print(f"Epoch {e+1}, loss={loss:.6f}")


def evaluate_and_plot(models: dict, test_loader: dict, pi_next: np.ndarray, args):
    models_dict = models

    print("\n================== Test / Evaluation ==================")
    print(f"test_ratio={args.test_ratio}, lookback={lookback}, horizon={horizon}")

    show_plots = not args.no_show

    metrics_rows = []
    pred_rows = []

    # ---------- 单模型 ----------
    for lab in ['S', 'C', 'R']:
        tl = test_loader[lab]
        if tl is None:
            print(f"[{lab}] 跳过评估：测试样本不足")
            metrics_rows.append({"tag": "single", "label": lab, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "n_samples": 0})
            continue

        y_true, y_pred = collect_predictions(models[lab], tl)
        metrics = calc_metrics(y_true, y_pred)
        n_samples = int(y_true.shape[0])

        metrics_rows.append({"tag": "single", "label": lab, **metrics, "n_samples": n_samples})

        print(f"[{lab}] MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")

        # CSV：保存该模型的预测 vs 真实（flatten）
        pred_rows.append(flatten_pred_true_to_df(y_true, y_pred, label=lab, tag="single"))

        # 可视化：只画 24h（默认 horizon=24h）
        y_true_plot, y_pred_plot, suffix = _select_24h_slice(
            y_true,
            y_pred,
            hours=float(getattr(args, 'plot_window_hours', 24.0)),
            mode=str(getattr(args, 'plot_window_mode', 'end')),
            horizon_steps=int(horizon or y_true.shape[1]),
            start_step=None,
            seed=int(getattr(args, 'plot_window_seed', 42)),
        )

        plot_pred_vs_true(
            y_true_plot,
            y_pred_plot,
            title=f"Test Pred vs True (label={lab}){suffix}",
            save_path=os.path.join(args.output_dir, f"test_pred_vs_true_{lab}.png"),
            max_points=args.max_plot_points,
            show=show_plots,
        )

    # ---------- 加权集成（随样本时间滚动更新权重） ----------
    # 说明：
    # - 这里的权重不是固定的 pi_next，而是根据窗口索引 t 使用 pi_series[t]
    # - pi_series 由 main() 在全局范围通过 Markov Ps 递推得出
    pi_series = getattr(args, "_pi_series", None)
    day_to_win = getattr(args, "_day_to_win", None)

    for lab in ['S', 'C', 'R']:
        tl = test_loader[lab]
        if tl is None:
            continue

        if pi_series is None or day_to_win is None:
            # 回退：沿用旧逻辑（固定权重）
            y_true_e, y_pred_e = collect_weighted_predictions(pi_next, models_dict, tl)
            tag = "ensemble_fixed"
        else:
            # 为该 label 的 test dataset 生成每个样本对应的窗口索引
            ds = tl.dataset
            if not hasattr(ds, "df"):
                raise RuntimeError("PVSeqDataset should have .df for time-varying index computation")

            # 每个样本的 anchor 时间取：x 的 마지막时间点（即 lookback-1 位置）
            ts = ds.df['timestamp'].reset_index(drop=True)
            if len(ts) < (lookback + horizon):
                continue

            sample_ts = ts.iloc[lookback - 1: len(ts) - horizon].to_list()
            sample_days = [t.date() for t in sample_ts]
            sample_win_idx = np.array([day_to_win.get(d, -1) for d in sample_days], dtype=int)

            y_true_e, y_pred_e = collect_weighted_predictions_timevarying(
                models=models_dict,
                loader=tl,
                pi_series=pi_series,
                sample_window_indices=sample_win_idx,
            )
            tag = "ensemble_timevarying"

        metrics = calc_metrics(y_true_e, y_pred_e)
        n_samples = int(y_true_e.shape[0])

        metrics_rows.append({"tag": tag, "label": lab, **metrics, "n_samples": n_samples})

        print(f"[{tag} on {lab}-test] MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")

        # CSV：保存集成预测 vs 真实（flatten）
        pred_rows.append(flatten_pred_true_to_df(y_true_e, y_pred_e, label=lab, tag=tag))

        # 可视化：只画 24h（默认 horizon=24h）
        y_true_plot, y_pred_plot, suffix = _select_24h_slice(
            y_true_e,
            y_pred_e,
            hours=float(getattr(args, 'plot_window_hours', 24.0)),
            mode=str(getattr(args, 'plot_window_mode', 'end')),
            horizon_steps=int(horizon or y_true_e.shape[1]),
            start_step=None,
            seed=int(getattr(args, 'plot_window_seed', 42)),
        )

        plot_pred_vs_true(
            y_true_plot,
            y_pred_plot,
            title=f"{tag} Pred vs True (on {lab}-test){suffix}",
            save_path=os.path.join(args.output_dir, f"test_pred_vs_true_{tag}_on_{lab}.png"),
            max_points=args.max_plot_points,
            show=show_plots,
        )

    # ---------- 保存 CSV ----------
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(args.output_dir, "test_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    # 预测序列：按 (tag, label) 拆分保存，避免单个 CSV 过大
    pred_index_rows = []
    if len(pred_rows) > 0:
        pred_dir = os.path.join(args.output_dir, "test_predictions")
        os.makedirs(pred_dir, exist_ok=True)

        for df_part in pred_rows:
            if df_part is None or len(df_part) == 0:
                continue

            tag = str(df_part["tag"].iloc[0]) if "tag" in df_part.columns else "unknown"
            lab = str(df_part["label"].iloc[0]) if "label" in df_part.columns else "unknown"

            fname = f"test_predictions_flat_{tag}_{lab}.csv"
            fpath = os.path.join(pred_dir, fname)
            df_part.to_csv(fpath, index=False, encoding="utf-8-sig")

            pred_index_rows.append(
                {
                    "tag": tag,
                    "label": lab,
                    "n_rows": int(len(df_part)),
                    "file": os.path.join("test_predictions", fname),
                }
            )

    pred_index_path = os.path.join(args.output_dir, "test_predictions_files.csv")
    pd.DataFrame(pred_index_rows, columns=["tag", "label", "n_rows", "file"]).to_csv(
        pred_index_path, index=False, encoding="utf-8-sig"
    )

    print(f"\n[CSV] 已保存指标: {metrics_path}")
    if len(pred_index_rows) > 0:
        print(f"[CSV] 已按 tag/label 拆分保存预测序列: {os.path.join(args.output_dir, 'test_predictions')}")
        print(f"[CSV] 预测文件索引: {pred_index_path}")
    else:
        print("[CSV] 未生成预测序列文件：pred_rows 为空")


def forecast_one(models: dict, df: pd.DataFrame, pi_next: np.ndarray, args):
    """对最新时刻做一次 24h 预测。

    旧逻辑：用单个 pi_next 对整个 horizon 加权。
    新逻辑（按你的要求）：对未来每一步递推权重 pi(t) 并逐步加权输出。

    说明：
      - 这里缺少“真实未来天气标签”的观测，所以未来的 P_t 无法严格按真实时间滚动估计。
      - 因此采用一个常见可行假设：未来每一步使用当前最新的转移矩阵 P_last（或回退 P_fixed）。
        即：pi_{k+1} = P_last @ pi_k。
      - 如果你后续有对未来天气状态的先验（例如用另一个模型先预测 S/C/R），
        我们也可以把 P_t 切换为按 predicted-state 切行的非齐次转移。
    """

    if lookback is None or horizon is None:
        raise RuntimeError("Globals not initialized. Call init_globals() first.")

    def _normalize_pi(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).reshape(3)
        s = float(v.sum())
        if s > 0:
            return v / s
        return np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)

    # 选择一个可以用于推理的 dataset bucket（优先最近 label，其次任意非空 bucket）
    latest_label = df['window_label'].dropna().iloc[-1] if df['window_label'].notna().any() else None
    label_to_ds = {
        'S': PVSeqDataset(df, 'S'),
        'C': PVSeqDataset(df, 'C'),
        'R': PVSeqDataset(df, 'R')
    }

    candidates = []
    if latest_label in label_to_ds:
        candidates.append(label_to_ds[latest_label])
    candidates.extend([label_to_ds['S'], label_to_ds['C'], label_to_ds['R']])

    ds_now = next((d for d in candidates if len(d) > 0), None)
    if ds_now is None:
        raise RuntimeError(
            f"No usable samples for inference. Need at least lookback+horizon={lookback + horizon} rows in any label bucket."
        )

    x_now, _ = ds_now[-1]
    x_now = x_now.unsqueeze(0).to(device)

    # 三模型给出整段 horizon 预测
    with torch.no_grad():
        pred_S = models['S'](x_now).cpu().numpy().reshape(-1)  # (H,)
        pred_C = models['C'](x_now).cpu().numpy().reshape(-1)
        pred_R = models['R'](x_now).cpu().numpy().reshape(-1)

    H = int(pred_S.shape[0])

    # 构造未来每一步的权重序列 w_k
    # 默认使用 main() 里计算的最后一个 P_last；若没有则退回固定 P
    P_last = getattr(args, "_P_last", None)
    if P_last is None:
        # 兜底：用当前 pi_next 推回一个近似 P_last ( 不可靠)，因此更安全是直接用单位矩阵
        P_last = np.eye(3, dtype=np.float64)

    # 从当前时刻的 pi(t0+1)=pi_next 作为起点，逐步递推
    pi_cur = _normalize_pi(np.asarray(pi_next, dtype=np.float64).reshape(3))
    w_series = np.zeros((H, 3), dtype=np.float64)
    for k in range(H):
        w_series[k, :] = pi_cur
        pi_cur = _normalize_pi(P_last @ pi_cur)

    # 逐步加权输出
    pred_final = (
        w_series[:, 0] * pred_S +
        w_series[:, 1] * pred_C +
        w_series[:, 2] * pred_R
    )

    # ---------- 可视化 ----------
    plt.figure(figsize=(10, 4))
    plt.plot(pred_final, label="Predicted Power (rolling ensemble)")
    plt.xlabel("Time step (5 min)")
    plt.ylabel("Power")
    plt.title("PV Power Forecast (24h)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "24h_PVPower.png"), dpi=300)
    if not args.no_show:
        plt.show()
    else:
        plt.close()

    # ---------- CSV 输出（推理阶段） ----------
    # 保存每一步三模型预测、加权预测、以及权重 w_k
    out_df = pd.DataFrame(
        {
            "k": np.arange(H, dtype=int),
            "w_S": w_series[:, 0],
            "w_C": w_series[:, 1],
            "w_R": w_series[:, 2],
            "pred_S": pred_S,
            "pred_C": pred_C,
            "pred_R": pred_R,
            "pred_ensemble": pred_final,
        }
    )
    out_path = os.path.join(args.output_dir, "forecast_24h_predictions.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[CSV] 已保存推理阶段(24h)预测: {out_path}")


def main(argv: list[str] | None = None):
    args = parse_args(argv)#解析命令行参数

    os.makedirs(args.output_dir, exist_ok=True)

    init_globals(args)
    print("Using device:", device)

    df = load_data(args.data_path)

    window_labels = compute_window_labels(df, args)
    print("窗口数量:", len(window_labels))

    # 固定 P（用于对比 & 作为滚动估计回退）
    P_fixed = build_markov(window_labels)
    if getattr(args, "print_fixed_P", False):
        print("固定转移矩阵 P_fixed =\n", P_fixed)

    # 随时间变化的 P_t（滚动窗口估计）
    Ps = build_markov_timevarying(
        window_labels,
        window_size=int(args.markov_window),
        min_transitions=int(args.markov_min_transitions),
    )
    if len(Ps) > 0:
        print(
            f"滚动转移矩阵已启用: markov_window={args.markov_window}, "
            f"min_transitions={args.markov_min_transitions}; 展示最后一个 P_t (P_last)=\n{Ps[-1]}"
        )
    else:
        print("滚动转移矩阵为空：将回退为固定 P_fixed（通常表示 window_labels 太短或无法估计转移）")

    # 可选：保存所有 P_t 到文件（仍然只在控制台打印最后一个）
    if getattr(args, "save_markov", False):
        try:
            base = os.path.join(args.output_dir, "markov_transitions")
            fmt = str(getattr(args, "markov_save_format", "csv")).lower()

            if len(Ps) == 0:
                # 兜底：没有滚动序列时保存 P_fixed 作为 P_0
                Ps_to_save = [P_fixed]
            else:
                Ps_to_save = Ps

            if fmt in ("csv", "both"):
                # 每一行: t, from_state, to_state, prob
                rows = []
                states = ["S", "C", "R"]
                for t, P in enumerate(Ps_to_save):
                    P = np.asarray(P, dtype=np.float64).reshape(3, 3)
                    for i, fs in enumerate(states):
                        for j, ts in enumerate(states):
                            rows.append(
                                {
                                    "t": int(t),
                                    "from": fs,
                                    "to": ts,
                                    "p": float(P[i, j]),
                                }
                            )
                pd.DataFrame(rows).to_csv(base + ".csv", index=False, encoding="utf-8-sig")

            if fmt in ("npz", "both"):
                arr = np.stack([np.asarray(P, dtype=np.float64).reshape(3, 3) for P in Ps_to_save], axis=0)
                np.savez_compressed(base + ".npz", Ps=arr)

            print(
                f"[Markov] 已保存所有 P_t: format={fmt}, count={len(Ps_to_save)} -> {base}.(csv/npz)"
            )
        except Exception as e:
            print(f"[Markov] 保存 P_t 失败: {e}")

    # 预计算：pi_series[t] 供测试集按样本时间滚动加权
    pi0 = np.array(args.pi, dtype=np.float64)
    pi_series = build_pi_series(Ps, pi0)

    # 预计算：day -> window index 映射
    unique_days = sorted(df['date'].unique())
    day_to_win = _compute_day_to_window_index(unique_days, window_days=int(args.window_days), step_days=int(args.step_days))

    # 把需要的滚动信息塞进 args（避免大量改函数签名）
    args._pi_series = pi_series
    args._day_to_win = day_to_win

    # 供 forecast_one 的多步递推使用：记录最后一个可用的 P_last
    args._P_last = Ps[-1] if len(Ps) > 0 else P_fixed

    df = assign_window_labels_to_df(df, window_labels)

    train_ds, test_ds, train_loader, test_loader = build_datasets_and_loaders(df, args)

    print("\n[Train] S samples:", len(train_ds['S']))
    print("[Train] C samples:", len(train_ds['C']))
    print("[Train] R samples:", len(train_ds['R']))

    print("\n[Test]  S samples:", len(test_ds['S']))
    print("[Test]  C samples:", len(test_ds['C']))
    print("[Test]  R samples:", len(test_ds['R']))

    models, opts = build_models(args)
    train_models(models, train_loader, opts, args)

    # 用最新时刻的 P_t 计算 pi_next（用于 forecast_one 的单次推理起点）
    pi = np.array(args.pi, dtype=np.float64)
    P_now = Ps[-1] if len(Ps) > 0 else P_fixed
    pi_next = P_now @ pi
    s = float(np.sum(pi_next))
    if s > 0:
        pi_next = pi_next / s

    evaluate_and_plot(models, test_loader, pi_next, args)
    forecast_one(models, df, pi_next, args)


if __name__ == "__main__":
    main()
