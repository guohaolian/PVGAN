# evaluate_imputation.py
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from scipy.stats import wasserstein_distance
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _to_numeric(series: pd.Series) -> pd.Series:
    # 把空字符串等情况转成 NaN，再转数值
    s = series.replace(r"^\s*$", np.nan, regex=True)
    return pd.to_numeric(s, errors="coerce")


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2:
        return float("nan")
    if a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
        return float("nan")
    return float(a.corr(b))


def evaluate(df: pd.DataFrame,
             col_imputed: str = "Power",
             col_missing: str = "_Power_missing",
             col_true: str = "ActualPower",
             eps: float = 1e-6,
             near_zero: float = 0.0) -> dict:
    if col_imputed not in df.columns or col_missing not in df.columns or col_true not in df.columns:
        raise ValueError(f"缺少列。需要列: {col_imputed}, {col_missing}, {col_true}")

    imputed = _to_numeric(df[col_imputed])
    missing = df[col_missing].replace(r"^\s*$", np.nan, regex=True)
    true = _to_numeric(df[col_true])

    # 缺失位置 mask 只认 missing 列的 NaN
    mask = missing.isna()

    y_pred = imputed[mask]
    y_true = true[mask]

    # 丢掉无法数值化的行
    valid = y_pred.notna() & y_true.notna()
    y_pred = y_pred[valid]
    y_true = y_true[valid]

    n_eval = int(valid.sum())

    if n_eval == 0:
        return {
            "n_eval": 0,
            "note": "缺失位置上没有可用于评价的有效数值，请检查数据是否读入正确，以及缺失标记是否为 NaN/空白。",
        }

    # 点对点误差
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    mse = float(mean_squared_error(y_true, y_pred, squared=True))

    # 相对误差，真实值接近 0 时也能算但更稳
    smape = float((2.0 * (y_pred - y_true).abs() / (y_true.abs() + y_pred.abs() + eps)).mean())
    mape = float(((y_pred - y_true).abs() / (y_true.abs() + eps)).mean())

    # 结构一致性
    corr = _safe_corr(y_true, y_pred)

    # 峰值误差
    peak_true = float(y_true.max())
    peak_pred = float(y_pred.max())
    peak_abs_err = float(abs(peak_pred - peak_true))

    # 造假功率比例，默认以真实值等于 0 为零功率
    if near_zero == 0.0:
        zero_mask = (y_true == 0)
    else:
        zero_mask = (y_true.abs() <= near_zero)

    false_power_ratio = float((y_pred[zero_mask] != 0).mean()) if zero_mask.any() else float("nan")
    false_power_mean = float(y_pred[zero_mask].mean()) if zero_mask.any() else float("nan")
    false_power_max = float(y_pred[zero_mask].abs().max()) if zero_mask.any() else float("nan")

    # 分布距离，可选
    wd = float(wasserstein_distance(y_true.values, y_pred.values)) if _HAS_SCIPY else float("nan")

    # baseline 对照
    baseline_zero = pd.Series(np.zeros(len(y_true)), index=y_true.index, dtype=float)
    baseline_mean = pd.Series(np.full(len(y_true), float(y_true.mean())), index=y_true.index, dtype=float)

    out = {
        "n_eval": n_eval,
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "mape": mape,
        "smape": smape,
        "corr": corr,
        "peak_true": peak_true,
        "peak_pred": peak_pred,
        "peak_abs_err": peak_abs_err,
        "false_power_ratio": false_power_ratio,
        "false_power_mean": false_power_mean,
        "false_power_abs_max": false_power_max,
        "wasserstein": wd,
        "baseline_zero_mae": float(mean_absolute_error(y_true, baseline_zero)),
        "baseline_zero_rmse": float(mean_squared_error(y_true, baseline_zero, squared=False)),
        "baseline_mean_mae": float(mean_absolute_error(y_true, baseline_mean)),
        "baseline_mean_rmse": float(mean_squared_error(y_true, baseline_mean, squared=False)),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="inputed_filled.csv", help="csv 或 xlsx 文件路径")
    parser.add_argument("--sheet", default=None, help="如果是 xlsx，填写 sheet 名，不填默认第一个")
    parser.add_argument("--imputed", default="Power")
    parser.add_argument("--missing", default="_Power_missing")
    parser.add_argument("--true", default="ActualPower")
    parser.add_argument("--near_zero", type=float, default=0.0, help="把真实值绝对值小于等于该阈值视为零功率，默认严格等于 0")
    parser.add_argument("--out", default="imputation_report.txt", help="输出报告文件名")
    args = parser.parse_args()

    if args.file.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(args.file, sheet_name=args.sheet)
    else:
        df = pd.read_csv(args.file)

    report = evaluate(
        df,
        col_imputed=args.imputed,
        col_missing=args.missing,
        col_true=args.true,
        near_zero=args.near_zero
    )

    # 打印
    print("\n评价结果")
    for k, v in report.items():
        print(f"{k}: {v}")

    # 保存
    pd.DataFrame([report]).to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n已保存报告到: {args.out}")
    with open(args.out, "w",encoding="utf-8-sig") as f:
        for key,value in report.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    main()
