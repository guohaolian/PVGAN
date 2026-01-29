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
    """收集加权集成预测与真实，返回 (y_true, y_pred) shape: (N, horizon)。"""
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

    return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)


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
        optimizer.step()

        loss_sum += loss.item()

    return loss_sum / len(loader)


def build_argparser():
    p = argparse.ArgumentParser(description="PV forecasting with KMeans weather labels + per-state LSTM + Markov ensemble")

    # IO
    p.add_argument("--data-path", type=str, default="data/91-Site_DKA-M9_B-Phase.csv")
    p.add_argument("--output-dir", type=str, default="./Plot_v1")

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

    # sliding window / labeling
    p.add_argument("--window-days", type=int, default=7)
    p.add_argument("--step-days", type=int, default=1)
    p.add_argument("--kmeans-clusters", type=int, default=3)
    p.add_argument("--kmeans-random-state", type=int, default=42)

    # sequence config
    p.add_argument("--lookback", type=int, default=72)
    p.add_argument("--horizon", type=int, default=96)

    # split & dataloader
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--test-batch-size", type=int, default=64)

    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)

    # model
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)

    # markov / ensemble
    p.add_argument("--pi", type=float, nargs=3, default=[1.0, 0.0, 0.0], help="initial state distribution for ensemble")

    # plotting
    p.add_argument("--max-plot-points", type=int, default=800)
    p.add_argument("--no-show", action="store_true", help="do not pop up matplotlib windows")

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

    for i in range(0, len(unique_days) - args.window_days + 1, args.step_days):
        days = unique_days[i: i + args.window_days]
        window_df = df[df['date'].isin(days)]

        X = window_df[feature_cols].values
        Xs = StandardScaler().fit_transform(X)

        kmeans = KMeans(n_clusters=args.kmeans_clusters, random_state=args.kmeans_random_state)
        labels = kmeans.fit_predict(Xs)

        window_df = window_df.copy()
        window_df['cluster'] = labels

        ghi_mean = window_df.groupby('cluster')['Global_Horizontal_Radiation'].mean()
        sorted_clusters = ghi_mean.sort_values(ascending=False).index.tolist()

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
        k: (DataLoader(v, batch_size=args.test_batch_size, shuffle=False) if len(v) > 0 else None)
        for k, v in test_ds.items()
    }

    return train_ds, test_ds, train_loader, test_loader


def build_models(args):
    if feature_cols is None or horizon is None or device is None:
        raise RuntimeError("Globals not initialized. Call init_globals() first.")

    models = {
        'S': PVLSTM(input_dim=len(feature_cols), hidden=args.hidden, horizon=horizon, num_layers=args.num_layers, dropout=args.dropout).to(device),
        'C': PVLSTM(input_dim=len(feature_cols), hidden=args.hidden, horizon=horizon, num_layers=args.num_layers, dropout=args.dropout).to(device),
        'R': PVLSTM(input_dim=len(feature_cols), hidden=args.hidden, horizon=horizon, num_layers=args.num_layers, dropout=args.dropout).to(device),
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

        # 可视化
        plot_pred_vs_true(
            y_true,
            y_pred,
            title=f"Test Pred vs True (label={lab})",
            save_path=os.path.join(args.output_dir, f"test_pred_vs_true_{lab}.png"),
            max_points=args.max_plot_points,
            show=show_plots,
        )

    # ---------- 加权集成 ----------
    for lab in ['S', 'C', 'R']:
        tl = test_loader[lab]
        if tl is None:
            continue

        y_true_e, y_pred_e = collect_weighted_predictions(pi_next, models_dict, tl)
        metrics = calc_metrics(y_true_e, y_pred_e)
        n_samples = int(y_true_e.shape[0])

        metrics_rows.append({"tag": "ensemble", "label": lab, **metrics, "n_samples": n_samples})

        print(f"[Ensemble on {lab}-test] MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")

        # CSV：保存集成预测 vs 真实（flatten）
        pred_rows.append(flatten_pred_true_to_df(y_true_e, y_pred_e, label=lab, tag="ensemble"))

        # 可视化
        plot_pred_vs_true(
            y_true_e,
            y_pred_e,
            title=f"Weighted Ensemble Pred vs True (on {lab}-test)",
            save_path=os.path.join(args.output_dir, f"test_pred_vs_true_ensemble_on_{lab}.png"),
            max_points=args.max_plot_points,
            show=show_plots,
        )

    # ---------- 保存 CSV ----------
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(args.output_dir, "test_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    if len(pred_rows) > 0:
        pred_df = pd.concat(pred_rows, ignore_index=True)
    else:
        pred_df = pd.DataFrame(columns=["tag", "label", "t", "y_true", "y_pred"])

    pred_path = os.path.join(args.output_dir, "test_predictions_flat.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    print(f"\n[CSV] 已保存指标: {metrics_path}")
    print(f"[CSV] 已保存预测序列: {pred_path}")


def forecast_one(models: dict, df: pd.DataFrame, pi_next: np.ndarray, args):
    if lookback is None or horizon is None:
        raise RuntimeError("Globals not initialized. Call init_globals() first.")

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

    with torch.no_grad():
        pred_S = models['S'](x_now).cpu().numpy()
        pred_C = models['C'](x_now).cpu().numpy()
        pred_R = models['R'](x_now).cpu().numpy()

    pred_final = (
        pi_next[0]*pred_S +
        pi_next[1]*pred_C +
        pi_next[2]*pred_R
    )

    plt.figure(figsize=(10, 4))
    plt.plot(pred_final.flatten(), label="Predicted Power")
    plt.xlabel("Time step (5 min)")
    plt.ylabel("Power")
    plt.title("PV Power Forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "Plot_v1/24h_PVPower.png"), dpi=300)
    if not args.no_show:
        plt.show()
    else:
        plt.close()


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    init_globals(args)
    print("Using device:", device)

    df = load_data(args.data_path)

    window_labels = compute_window_labels(df, args)
    print("窗口数量:", len(window_labels))

    P = build_markov(window_labels)
    print("转移矩阵 P =\n", P)

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

    pi = np.array(args.pi, dtype=np.float64)
    pi_next = P @ pi

    evaluate_and_plot(models, test_loader, pi_next, args)
    forecast_one(models, df, pi_next, args)


if __name__ == "__main__":
    main()
