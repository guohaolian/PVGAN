import os
import math
import random
import argparse
from typing import Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam


# -------------------------
# Utilities
# -------------------------
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def makedirs(path):
    os.makedirs(path, exist_ok=True)


def gradient_penalty(D, real, fake, device='cpu', lambda_gp=10.0):
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, device=device)
    alpha = alpha.expand_as(real)
    interpolates = alpha * real + ((1 - alpha) * fake)
    interpolates = interpolates.detach().requires_grad_(True)

    disc_interpolates = D(interpolates)
    grad_outputs = torch.ones_like(disc_interpolates, device=device)

    grads = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    grads = grads.view(grads.size(0), -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gp


# -------------------------
# Dataset
# -------------------------
class PVTimeSeriesDataset(Dataset):
    """
    PV 时序数据集，支持在 Power 列上做随机 block 遮挡增强。

    关键点：
    - orig_mask: 原始是否有值 (NaN -> 0, 有值 -> 1)
    - aug_mask: 我们人为遮挡的位置 (只对 Power 列)
    - input_mask: 喂给模型的观测掩码 = orig_mask * aug_mask
    - train_miss_mask_power: 只在 (orig_mask=1 & aug_mask=0) 的 Power 位置 = 1，用于 L1 / grad 监督
    """
    def __init__(self,
                 csv_path: str,
                 seq_len=288,
                 power_col='Power',
                 feature_cols=None,
                 create_aug_masks=True,
                 aug_prob=0.5,
                 aug_max_block=50):
        super().__init__()
        self.seq_len = seq_len
        self.power_col = power_col
        self.create_aug_masks = create_aug_masks
        self.aug_prob = aug_prob
        self.aug_max_block = aug_max_block

        df = pd.read_csv(csv_path)

        # 选择特征列：排除各种时间列
        if feature_cols is None:
            candidate = [c for c in df.columns
                         if c.lower() not in ('timestamp', 'time', 'date', 'datetime')]#if c.lower() in ('power')]
            feature_cols = [c for c in candidate if c in df.columns]
        self.feature_cols = feature_cols

        # 读入为 float，空字符串视为 NaN
        data = df[self.feature_cols].replace(r'^\s*$', np.nan, regex=True).astype(float).values

        self.raw = data  # (T, D)，保留 NaN
        self.T, self.D = self.raw.shape

        # 原始掩码：1 = 有值, 0 = NaN
        self.orig_mask = (~np.isnan(self.raw)).astype(np.float32)

        # 缺失填 0 便于标准化
        self.raw_filled = np.nan_to_num(self.raw, nan=0.0).astype(np.float32)

        # 按非 NaN 的值计算均值和方差
        self.mean = np.nanmean(self.raw, axis=0)
        self.std = np.nanstd(self.raw, axis=0)
        self.std[self.std < 1e-6] = 1.0

        # 标准化后缺失位置为 0
        self.normed = ((self.raw_filled - self.mean[None, :]) / self.std[None, :]).astype(np.float32)
        self.orig_mask = self.orig_mask.astype(np.float32)

        # 所有可能的起始位置
        self.starts = []
        max_start = max(1, self.T - self.seq_len + 1)
        for s in range(0, max_start):
            self.starts.append(s)

    def __len__(self):
        return len(self.starts)

    def _create_random_block_mask(self):
        """
        仅对 Power 列生成随机 block 掩码。
        1 = 保留（观测），0 = 人为遮挡。
        """
        mask = np.ones((self.seq_len, self.D), dtype=np.float32)
        if self.create_aug_masks and random.random() < self.aug_prob:
            nblocks = random.randint(1, 3)
            for _ in range(nblocks):
                block_len = random.randint(
                    1, int(min(self.aug_max_block, max(1, self.seq_len // 2)))
                )
                start = random.randint(0, max(0, self.seq_len - block_len))
                mask[start:start + block_len, self.feature_cols.index(self.power_col)] = 0.0
        return mask

    def __getitem__(self, idx):
        s = self.starts[idx]
        end = s + self.seq_len
        if end <= self.T:
            seq = self.normed[s:end]  # (seq_len, D)
            orig_mask = self.orig_mask[s:end]
            raw_slice = self.raw_filled[s:end]
        else:
            # 末尾不足一整段时 padding
            avail = self.T - s
            seq = np.zeros((self.seq_len, self.D), dtype=np.float32)
            orig_mask = np.zeros((self.seq_len, self.D), dtype=np.float32)
            raw_slice = np.zeros((self.seq_len, self.D), dtype=np.float32)
            if avail > 0:
                seq[:avail] = self.normed[s:s + avail]
                orig_mask[:avail] = self.orig_mask[s:s + avail]
                raw_slice[:avail] = self.raw_filled[s:s + avail]

        # 在 Power 列上做随机 block 遮挡
        aug_mask = self._create_random_block_mask()
        pidx = self.feature_cols.index(self.power_col)

        # 喂给模型的 mask：原本有值且未被增强遮挡
        input_mask = orig_mask.copy()
        input_mask[:, pidx] = orig_mask[:, pidx] * aug_mask[:, pidx]

        # 监督掩码：只在“原来有值 & 我们故意遮挡”的 Power 位置为 1
        train_miss_mask_power = ((orig_mask[:, pidx] == 1) &
                                 (aug_mask[:, pidx] == 0)).astype(np.float32)

        sample = {
            'x': seq,
            'mask': input_mask,
            'raw_slice': raw_slice,
            'power_idx': pidx,
            'train_miss_mask_power': train_miss_mask_power
        }
        return sample


# -------------------------
# Conv Blocks (for Discriminators)
# -------------------------
class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, dilation=1, norm=True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding, dilation=dilation)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.norm = nn.InstanceNorm1d(out_ch, affine=True) if norm else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DeconvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=4, stride=2, padding=1, norm=True):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.norm = nn.InstanceNorm1d(out_ch, affine=True) if norm else nn.Identity()

    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# -------------------------
# Generator (Transformer-based)
# -------------------------
class PositionalEncoding1D(nn.Module):
    """标准正弦位置编码，用于 1D 序列."""
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, L, d_model)
        """
        L = x.size(1)
        return x + self.pe[:, :L]


class GeneratorTransformer1D(nn.Module):
    """
    Transformer 版生成器：
    输入 : (B, C_in, L)，C_in = D + 1(agg_mask) + 1(noise)
    输出 : (B, feat_ch, L)，feat_ch = D
    """
    def __init__(
        self,
        in_channels: int,
        feat_ch: int,
        base_ch: int = 64,
        num_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_ch = feat_ch
        self.d_model = base_ch

        # (features + mask + noise) -> d_model
        self.input_proj = nn.Linear(in_channels, self.d_model)

        self.pos_encoder = PositionalEncoding1D(self.d_model, max_len=4096)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # 使用 (B, L, C)
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出投影回特征维度 D
        self.out_proj = nn.Linear(self.d_model, feat_ch)

    def forward(self, x, agg_mask=None):
        """
        x: (B, C_in, L)
        agg_mask: (B, 1, L)，目前没显式用在 attention mask 中，只作为输入通道的一部分
        """
        # -> (B, L, C_in)
        x = x.permute(0, 2, 1)

        # 线性投影到 d_model
        x = self.input_proj(x)  # (B, L, d_model)

        # 加位置编码
        x = self.pos_encoder(x)

        # 此处可以根据 pad 情况构造 key_padding_mask，这里先简单不加
        out = self.encoder(x)  # (B, L, d_model)

        out = self.out_proj(out)  # (B, L, feat_ch)

        # -> (B, feat_ch, L)，保持与原 U-Net 输出一致
        out = out.permute(0, 2, 1)
        return out


# -------------------------
# Discriminators
# -------------------------
class GlobalDiscriminator(nn.Module):
    """
    全局判别器，看整段序列。
    输入: (B, C_in, L)，这里 C_in = D + 1 (features + noise)
    """
    def __init__(self, in_channels, base_ch=64):
        super().__init__()
        self.main = nn.Sequential(
            ConvBlock1D(in_channels, base_ch, kernel=4, stride=2, padding=1),
            ConvBlock1D(base_ch, base_ch * 2, kernel=4, stride=2, padding=1),
            ConvBlock1D(base_ch * 2, base_ch * 4, kernel=4, stride=2, padding=1),
            ConvBlock1D(base_ch * 4, base_ch * 8, kernel=4, stride=2, padding=1),
        )
        self.fc = nn.Linear(base_ch * 8, 1)

    def forward(self, x):
        # x: (B, C, L)
        h = self.main(x)
        h = h.mean(dim=2)  # global pooling over time
        out = self.fc(h)
        return out  # (B, 1)


class LocalPatchDiscriminator(nn.Module):
    """
    局部 patch 判别器，随机裁剪一个窗口判别。
    输入: 同上。
    """
    def __init__(self, in_channels, base_ch=64, patch_len=64):
        super().__init__()
        self.patch_len = patch_len
        self.main = nn.Sequential(
            ConvBlock1D(in_channels, base_ch, kernel=4, stride=2, padding=1),
            ConvBlock1D(base_ch, base_ch * 2, kernel=4, stride=2, padding=1),
            ConvBlock1D(base_ch * 2, base_ch * 4, kernel=4, stride=2, padding=1),
        )
        self.fc = nn.Linear(base_ch * 4, 1)

    def forward(self, x):
        B, C, L = x.shape
        if L <= self.patch_len:
            h = self.main(x)
            h = h.mean(dim=2)
            out = self.fc(h)
            return out

        starts = torch.randint(low=0, high=L - self.patch_len + 1, size=(B,), device=x.device)
        patches = []
        for i in range(B):
            s = starts[i].item()
            patches.append(x[i:i + 1, :, s:s + self.patch_len])
        patches = torch.cat(patches, dim=0)

        h = self.main(patches)
        h = h.mean(dim=2)
        out = self.fc(h)
        return out


# -------------------------
# Trainer
# -------------------------
class Trainer:
    def __init__(self, config):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.config = config
        seed_all(config.seed)

        # Dataset & Dataloader
        ds = PVTimeSeriesDataset(config.csv, seq_len=config.seq_len, power_col=config.power_col,
                                 create_aug_masks=True, aug_prob=0.6, aug_max_block=config.aug_max_block)
        self.ds = ds
        self.dl = DataLoader(ds, batch_size=config.batch_size, shuffle=True,
                             drop_last=True, num_workers=0)

        # 通道数设置：G 和 D 不一样
        in_ch_G = ds.D + 1 + 1  # features + agg_mask + noise
        in_ch_D = ds.D + 1      # features + noise

        # 使用 Transformer 生成器
        self.G = GeneratorTransformer1D(
            in_channels=in_ch_G,
            feat_ch=self.ds.D,
            base_ch=config.base_ch,
            num_layers=4,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
        ).to(self.device)

        self.Dglob = GlobalDiscriminator(in_channels=in_ch_D, base_ch=config.base_ch).to(self.device)
        self.Dlocal = LocalPatchDiscriminator(in_channels=in_ch_D, base_ch=config.base_ch).to(self.device)

        self.optG = Adam(self.G.parameters(), lr=config.lr_g, betas=(0.5, 0.9))
        self.optD = Adam(list(self.Dglob.parameters()) + list(self.Dlocal.parameters()),
                         lr=config.lr_d, betas=(0.5, 0.9))

        makedirs(config.ckpt_dir)
        self.iter = 0

    def sample_noise(self, B, L):
        return torch.randn(B, 1, L, device=self.device)

    def ensure_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.device).float()
        else:
            return torch.from_numpy(data).to(self.device).float()

    def prepare_input(self, x, mask):
        """
        x: (B, L, D)  normalized
        mask: (B, L, D)
        生成器输入: [features, agg_mask, noise] -> (B, D+2, L)
        """
        B, L, D = x.shape
        x_t = self.ensure_tensor(x).permute(0, 2, 1)       # (B, D, L)
        mask_t = self.ensure_tensor(mask).permute(0, 2, 1) # (B, D, L)

        agg_mask = mask_t.mean(dim=1, keepdim=True)        # (B, 1, L)
        noise = self.sample_noise(B, L)                    # (B, 1, L)

        inp = torch.cat([x_t, agg_mask, noise], dim=1)     # (B, D+2, L)
        return inp, agg_mask

    def train(self):
        cfg = self.config
        device = self.device

        # === Phase 1: Pretraining Generator with L1 Loss ===
        if cfg.pretrain_epochs > 0:
            for epoch in range(cfg.pretrain_epochs):
                pbar = tqdm(self.dl, desc=f"pretrain ep{epoch}")
                for batch in pbar:
                    x = batch['x']
                    mask = batch['mask']
                    train_miss = batch['train_miss_mask_power']
                    B, L, D = x.shape
                    inp, agg_mask = self.prepare_input(x, mask)

                    self.G.train()
                    pred = self.G(inp, agg_mask)  # (B, D, L)

                    pidx = self.ds.feature_cols.index(cfg.power_col)
                    pred_power = pred.permute(0, 2, 1)[:, :, pidx]    # (B, L)
                    target_power = self.ensure_tensor(x[:, :, pidx])  # (B, L)

                    train_miss_t = self.ensure_tensor(train_miss)     # (B, L)

                    l1 = torch.abs(pred_power - target_power) * train_miss_t
                    loss = l1.sum() / (train_miss_t.sum() + 1e-8)

                    self.optG.zero_grad()
                    loss.backward()
                    self.optG.step()

                    pbar.set_postfix(L1=loss.item())

                torch.save({'G': self.G.state_dict()},
                           os.path.join(cfg.ckpt_dir, f'pretrain_g_ep{epoch}.pt'))

        # === Phase 2: Adversarial Training (WGAN-GP) ===
        for epoch in range(cfg.epochs):
            pbar = tqdm(self.dl, desc=f"adv ep{epoch}")
            for batch in pbar:
                x = batch['x']
                mask = batch['mask']
                train_miss = batch['train_miss_mask_power']
                B, L, D = x.shape
                inp, agg_mask = self.prepare_input(x, mask)
                inp_real = inp.clone()  # (B, D+2, L)

                # ------------------- Train Discriminators -------------------
                for _ in range(cfg.n_critic):
                    fake = self.G(inp_real, agg_mask)
                    fake_det = fake.detach()

                    # 判别器只看特征 + noise
                    real_feat = inp_real[:, :self.ds.D, :]                 # (B, D, L)
                    fake_full_D = torch.cat([fake_det, self.sample_noise(B, L)], dim=1)  # (B, D+1, L)
                    real_full_D = torch.cat([real_feat, self.sample_noise(B, L)], dim=1) # (B, D+1, L)

                    real_score_g = self.Dglob(real_full_D)
                    fake_score_g = self.Dglob(fake_full_D)
                    real_score_l = self.Dlocal(real_full_D)
                    fake_score_l = self.Dlocal(fake_full_D)

                    d_loss = fake_score_g.mean() + fake_score_l.mean() \
                             - real_score_g.mean() - real_score_l.mean()

                    gp = gradient_penalty(self.Dglob, real_full_D, fake_full_D,
                                          device=device, lambda_gp=cfg.gp_lambda)
                    d_loss_total = d_loss + gp

                    self.optD.zero_grad()
                    d_loss_total.backward()
                    self.optD.step()

                # ------------------- Train Generator -------------------
                fake = self.G(inp_real, agg_mask)
                fake_full_D = torch.cat([fake, self.sample_noise(B, L)], dim=1)

                fake_score_g = self.Dglob(fake_full_D)
                fake_score_l = self.Dlocal(fake_full_D)
                adv_loss = - (fake_score_g.mean() + fake_score_l.mean()) * 0.5

                pidx = self.ds.feature_cols.index(cfg.power_col)
                pred_power = fake.permute(0, 2, 1)[:, :, pidx]        # (B, L)
                target_power = self.ensure_tensor(x[:, :, pidx])      # (B, L)

                train_miss_t = self.ensure_tensor(train_miss)         # (B, L)

                # 只在“可监督缺失”位置算重构
                rec_l1 = torch.abs(pred_power - target_power) * train_miss_t
                rec_loss = rec_l1.sum() / (train_miss_t.sum() + 1e-8)

                # feature matching：全局均值差
                fm = ((fake.mean(dim=[1, 2]) - inp_real.mean(dim=[1, 2])).pow(2)).mean()

                # 梯度约束，同样只在可监督缺失位置
                dpred = pred_power[:, 1:] - pred_power[:, :-1]
                dtarget = target_power[:, 1:] - target_power[:, :-1]
                grad_mask = train_miss_t[:, 1:]
                grad_loss = torch.abs(dpred - dtarget) * grad_mask
                grad_loss = grad_loss.sum() / (grad_mask.sum() + 1e-8)

                g_loss = (cfg.lambda_adv * adv_loss +
                          cfg.lambda_rec * rec_loss +
                          cfg.lambda_fm * fm +
                          cfg.lambda_grad * grad_loss)

                self.optG.zero_grad()
                g_loss.backward()
                self.optG.step()

                self.iter += 1
                pbar.set_postfix(Dloss=d_loss.item(),
                                 Gloss=g_loss.item(),
                                 Rec=rec_loss.item())

            torch.save(
                {
                    'G': self.G.state_dict(),
                    'Dglob': self.Dglob.state_dict(),
                    'Dlocal': self.Dlocal.state_dict()
                },
                os.path.join(cfg.ckpt_dir, f'ckpt_ep{epoch}.pt')
            )

        print("Training finished.")

    # -------------------------
    # Imputation on a full csv
    # -------------------------
    def impute(self, input_csv: str, ckpt_path: str, out_csv: str = None):
        device = self.device

        ckpt = torch.load(ckpt_path, map_location=device)
        self.G.load_state_dict(ckpt['G'])
        self.G.eval()

        df = pd.read_csv(input_csv)
        feature_cols = self.ds.feature_cols
        data = df[feature_cols].replace(r'^\s*$', np.nan, regex=True).astype(float).values

        raw = data.copy()
        mask = (~np.isnan(raw)).astype(np.float32)
        filled = np.nan_to_num(raw, nan=0.0)

        normed = (filled - self.ds.mean[None, :]) / self.ds.std[None, :]

        T, D = normed.shape
        seq_len = self.config.seq_len

        imputed = filled.copy()

        with torch.no_grad():
            for start in tqdm(range(0, T, seq_len), desc="Imputing"):
                end = min(start + seq_len, T)
                chunk_len = end - start

                x_seq = np.zeros((1, seq_len, D), dtype=np.float32)
                m_seq = np.zeros((1, seq_len, D), dtype=np.float32)

                x_seq[0, :chunk_len] = normed[start:end]
                m_seq[0, :chunk_len] = mask[start:end]

                inp, agg_mask = self.prepare_input(x_seq, m_seq)
                fake = self.G(inp, agg_mask)  # (1, D, L)
                fake_np = fake.cpu().numpy()[0, :, :chunk_len].transpose(1, 0)  # (chunk_len, D)

                fake_denorm = fake_np * self.ds.std[None, :] + self.ds.mean[None, :]

                for t in range(chunk_len):
                    for d in range(D):
                        if mask[start + t, d] == 0:
                            imputed[start + t, d] = fake_denorm[t, d]

        out_df = df.copy()
        for i, col in enumerate(feature_cols):
            out_df[col] = imputed[:, i]

        if out_csv is not None:
            out_df.to_csv(out_csv, index=False)

        return out_df


# -------------------------
# Config & Main
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/weather_data_with_missing.csv")
    parser.add_argument("--power_col", type=str, default="Power")
    parser.add_argument("--seq_len", type=int, default=288)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--pretrain_epochs", type=int, default=3)
    parser.add_argument("--n_critic", type=int, default=3)
    parser.add_argument("--gp_lambda", type=float, default=10.0)
    parser.add_argument("--lambda_adv", type=float, default=1.0)
    parser.add_argument("--lambda_rec", type=float, default=200.0)
    parser.add_argument("--lambda_fm", type=float, default=10.0)
    parser.add_argument("--lambda_grad", type=float, default=10.0)
    parser.add_argument("--aug_max_block", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")

    parser.add_argument("--mode", type=str, choices=["train", "impute"], default="train")
    parser.add_argument("--impute_input_csv", type=str, default=None)
    parser.add_argument("--impute_ckpt", type=str, default="checkpoints/ckpt_ep1.pt")
    parser.add_argument("--impute_out_csv", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    trainer = Trainer(args)

    if args.mode == "train":
        trainer.train()
    else:
        input_csv = args.impute_input_csv or args.csv
        ckpt_path = args.impute_ckpt
        out_csv = args.impute_out_csv or "pv_data_filled_trans.csv"
        trainer.impute(input_csv, ckpt_path, out_csv)


if __name__ == "__main__":
    main()
