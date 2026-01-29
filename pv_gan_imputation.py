"""
pv_gan_imputation.py

完整 PyTorch 实现：用于光伏发电时间序列缺失补全的 GAN（生成器 + 判别器 + 训练）
使用说明（最小）:
    python pv_gan_imputation.py --csv data.csv --power_col Power
或直接在模块中修改 main() 中的路径并运行。

依赖:
    pip install torch pandas numpy tqdm matplotlib

架构概要:
- Generator: 双向 conv encoders (past/future) -> 融合 -> U-Net style decoder (1D convs)
- Discriminator: Global conv-based 判别器 + Local patch 判别器
- Losses: WGAN-GP (对抗), L1 重建（缺失位置）, feature matching, gradient loss
"""

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

# -------------------------
# Dataset
# -------------------------
class PVTimeSeriesDataset(Dataset):
    """
    Data loader for CSV time series.
    Expects a timestamp column (optional) and feature columns including target power column.
    It returns sequences of fixed length (seq_len). Missing values in the CSV are read as NaN,
    and converted to mask (1 observed, 0 missing). For training data augmentation we optionally
    create artificial missing blocks.
    """
    def __init__(self, csv_path: str, seq_len=256, power_col='Power', feature_cols=None,
                 create_aug_masks=True, aug_prob=0.5, aug_max_block=64):
        df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.power_col = power_col
        # choose feature columns: keep numeric columns except timestamp
        if feature_cols is None:
            candidate = [c for c in df.columns if c.lower() not in ('timestamp','time','date')]
            # keep columns present
            feature_cols = [c for c in candidate if c in df.columns]
        self.feature_cols = feature_cols
        # build numpy array
        data = df[self.feature_cols].replace(r'^\s*$', np.nan, regex=True).astype(float).values
        # We'll normalize per-feature later
        self.raw = data  # shape (T, dim)
        self.T, self.D = self.raw.shape
        # build masks: observed=1, missing=0
        self.orig_mask = (~np.isnan(self.raw)).astype(float)
        # fill NaNs with 0 temporarily
        self.raw_filled = np.nan_to_num(self.raw, nan=0.0)
        # standardize features (per feature)
        self.mean = np.nanmean(self.raw, axis=0)
        self.std = np.nanstd(self.raw, axis=0) + 1e-6
        self.normed = (self.raw_filled - self.mean[None, :]) / self.std[None, :]
        # store as float32
        self.normed = self.normed.astype(np.float32)
        self.orig_mask = self.orig_mask.astype(np.float32)

        self.create_aug_masks = create_aug_masks
        self.aug_prob = aug_prob
        self.aug_max_block = aug_max_block

        # compute available sequence start indices
        self.starts = []
        for s in range(0, max(1, self.T - self.seq_len + 1)):
            self.starts.append(s)

    def __len__(self):
        return len(self.starts)

    def _create_random_block_mask(self):
        # return a mask array shape (seq_len, D) with some additional artificial missing blocks
        mask = np.ones((self.seq_len, self.D), dtype=np.float32)
        if random.random() < self.aug_prob:
            # choose number of blocks
            nblocks = random.randint(1, 3)
            for _ in range(nblocks):
                block_len = random.randint(1, int(min(self.aug_max_block, self.seq_len//2)))
                start = random.randint(0, self.seq_len - block_len)
                mask[start:start+block_len, :] = 0.0
        return mask

    def __getitem__(self, idx):
        s = self.starts[idx]
        seq = self.normed[s:s+self.seq_len]  # (seq_len, D)
        mask = self.orig_mask[s:s+self.seq_len]  # observed mask
        # optionally augment with random missing blocks (simulates extra missingness)
        if self.create_aug_masks:
            aug_mask = self._create_random_block_mask()
            # combine: we cannot mark originally observed as missing unless aug sets it to 0;
            # final mask = orig_mask * aug_mask
            mask = mask * aug_mask
        # we also return the original unnormalized target for metric calculation (only power col)
        # find index of power col in feature_cols
        try:
            power_idx = self.feature_cols.index(self.power_col)
        except ValueError:
            power_idx = None
        sample = {
            'x': seq,               # normalized features with NaNs filled
            'mask': mask,           # observed mask (1 observed, 0 missing)
            'raw_slice': self.raw[s:s+self.seq_len],  # original raw (with zeros where NaN)
            'power_idx': power_idx
        }
        return sample

# -------------------------
# Model Blocks
# -------------------------
class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, dilation=1, norm=True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding, dilation=dilation)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.norm = nn.InstanceNorm1d(out_ch, affine=True) if norm else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class GatedConv1D(nn.Module):
    """
    Simple gated conv: splits channels into two, one for content, one for gate.
    Helps to prevent 'mask leakage' when input contains placeholders for missing.
    """
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch*2, kernel_size=kernel, padding=padding)
        self.act = nn.ELU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        a, b = out.chunk(2, dim=1)
        return self.act(a) * self.sig(b)

# -------------------------
# Generator (Dual-branch encoder + U-Net decoder)
# -------------------------
class GeneratorUNet1D(nn.Module):
    def __init__(self, in_channels, base_ch=64, latent_ch=256):
        """
        in_channels: number of input channels (features + mask + optional time enc + noise)
        """
        super().__init__()
        # encoder layers for past branch
        self.enc1_p = ConvBlock1D(in_channels, base_ch)
        self.enc2_p = ConvBlock1D(base_ch, base_ch*2, dilation=2, padding=2)
        self.enc3_p = ConvBlock1D(base_ch*2, base_ch*4, dilation=4, padding=4)

        # encoder layers for future branch (same structure)
        self.enc1_f = ConvBlock1D(in_channels, base_ch)
        self.enc2_f = ConvBlock1D(base_ch, base_ch*2, dilation=2, padding=2)
        self.enc3_f = ConvBlock1D(base_ch*2, base_ch*4, dilation=4, padding=4)

        # fusion conv
        self.fuse = ConvBlock1D(base_ch*8, latent_ch)

        # decoder (upsample via conv transpose)
        self.dec3 = nn.ConvTranspose1d(latent_ch, base_ch*4, kernel_size=3, stride=1, padding=1)
        self.dec2 = nn.ConvTranspose1d(base_ch*4, base_ch*2, kernel_size=3, stride=1, padding=1)
        self.dec1 = nn.ConvTranspose1d(base_ch*2, base_ch, kernel_size=3, stride=1, padding=1)

        self.out_conv = nn.Conv1d(base_ch, in_channels, kernel_size=1)  # outputs correction for all channels

        # small gating convs
        self.gate1 = GatedConv1D(in_channels, base_ch)
        self.gate2 = GatedConv1D(base_ch, base_ch*2)
        self.gate3 = GatedConv1D(base_ch*2, base_ch*4)

    def forward(self, x, mask):
        """
        x: (B, C, L) normalized input (missing filled with zeros)
        mask: (B, 1, L) mask for target channel(s) (1 observed, 0 missing)
        We'll concat mask as an extra channel outside.
        """
        # Past branch (normal order)
        p1 = self.enc1_p(x)
        p1g = self.gate1(x)  # gate from raw input
        p2 = self.enc2_p(p1)
        p2g = self.gate2(p1)
        p3 = self.enc3_p(p2)
        p3g = self.gate3(p2)

        # Future branch: reverse along time axis
        xr = torch.flip(x, dims=[2])
        f1 = self.enc1_f(xr)
        f2 = self.enc2_f(f1)
        f3 = self.enc3_f(f2)
        # flip back features to align
        f1 = torch.flip(f1, dims=[2])
        f2 = torch.flip(f2, dims=[2])
        f3 = torch.flip(f3, dims=[2])

        # concat features along channel
        fused = torch.cat([p3, f3], dim=1)  # (B, 8*base_ch / or combined)
        z = self.fuse(fused)

        d3 = self.dec3(z) + p3  # skip add
        d2 = self.dec2(d3) + p2
        d1 = self.dec1(d2) + p1

        out = self.out_conv(d1)
        # out is same shape as x: it's the generator's predicted correction; we produce final prediction = x + out
        pred = x + out
        return pred

# -------------------------
# Discriminator(s)
# -------------------------
class GlobalDiscriminator(nn.Module):
    def __init__(self, in_channels, base_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_ch, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(base_ch, base_ch*2, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(base_ch*2, base_ch*4, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(base_ch*4, base_ch*8, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_ch*8, 1)
        )

    def forward(self, x):
        return self.net(x).view(-1)

class LocalPatchDiscriminator(nn.Module):
    def __init__(self, in_channels, base_ch=64):
        super().__init__()
        # outputs a score per patch window (PatchGAN-like)
        self.conv1 = nn.Conv1d(in_channels, base_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(base_ch, base_ch*2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(base_ch*2, base_ch*4, 3, stride=2, padding=1)
        self.out = nn.Conv1d(base_ch*4, 1, 1)

        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        out = self.out(h)  # shape (B, 1, L')
        # global pooling to single scalar per sample (for WGAN) - we average over patches
        return out.view(x.size(0), -1).mean(dim=1)

# -------------------------
# Losses (WGAN-GP)
# -------------------------
def gradient_penalty(D, real, fake, device='cpu'):
    alpha = torch.rand(real.size(0), 1, 1, device=device)
    alpha = alpha.expand_as(real)
    inter = alpha * real + (1 - alpha) * fake
    inter.requires_grad_(True)
    d_inter = D(inter)
    grads = torch.autograd.grad(outputs=d_inter.sum(), inputs=inter,
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
    grads = grads.view(real.size(0), -1)
    grad_norm = grads.norm(2, dim=1)
    gp = ((grad_norm - 1) ** 2).mean()
    return gp

# -------------------------
# Trainer
# -------------------------
class Trainer:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        seed_all(config.seed)

        # Create dataset & dataloader
        ds = PVTimeSeriesDataset(config.csv, seq_len=config.seq_len, power_col=config.power_col,
                                 create_aug_masks=True, aug_prob=0.6, aug_max_block=config.aug_max_block)
        self.ds = ds
        self.dl = DataLoader(ds, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=0)

        # compute input channels: features + mask channel + optional noise channel (we include noise)
        in_ch = ds.D + 1 + 1  # features + mask + noise
        print(f"Dataset T={ds.T}, feature dims={ds.D}, in_channels={in_ch}")

        # models
        self.G = GeneratorUNet1D(in_channels=in_ch, base_ch=config.base_ch).to(self.device)
        self.Dglob = GlobalDiscriminator(in_channels=in_ch, base_ch=config.base_ch).to(self.device)
        self.Dlocal = LocalPatchDiscriminator(in_channels=in_ch, base_ch=config.base_ch).to(self.device)

        # optimizers
        self.optG = Adam(self.G.parameters(), lr=config.lr_g, betas=(0.5, 0.9))
        self.optD = Adam(list(self.Dglob.parameters()) + list(self.Dlocal.parameters()), lr=config.lr_d, betas=(0.5, 0.9))

        # training state
        makedirs(config.ckpt_dir)
        self.iter = 0

    def sample_noise(self, B, L):
        # simple gaussian noise channel, same length L
        return torch.randn(B, 1, L, device=self.device)

    def prepare_input(self, x_np, mask_np):
        """
        x_np: (B, L, D) normalized data
        mask_np: (B, L, D) observed mask
        We will concatenate: features (D) + observed_mask_for_powerchannel? We'll use per-feature mask, but to keep channels manageable,
        we create a single aggregated mask channel: 1 if feature target (power) observed else 0.
        For simplicity, we use the mask for the power column only if available; else use any-observed mask.
        """
        B, L, D = x_np.shape
        if isinstance(x_np, torch.Tensor):
            x = x_np.to(self.device)# (B,L,D)
        else:
            x = torch.from_numpy(x_np).to(self.device)

        if isinstance(mask_np, torch.Tensor):
            mask = mask_np.to(self.device)
        else:
            mask = torch.from_numpy(mask_np).to(self.device)

        # Use the dataset's power index
        pidx = self.ds.feature_cols.index(self.config.power_col) if self.config.power_col in self.ds.feature_cols else 0
        mask_power = mask[:, :, pidx:pidx+1]  # (B,L,1)
        # aggregate mask channel (B,L,1)
        agg_mask = mask_power
        # noise channel
        noise = torch.randn(B, L, 1, device=self.device)
        # concat along channel dim, final shape (B,L,in_ch)
        inp = torch.cat([x, agg_mask, noise], axis=2)
        # transpose to (B, C, L)
        inp = inp.permute(0, 2, 1)
        agg_mask_t = agg_mask.permute(0, 2, 1)  # (B,1,L)
        return inp, agg_mask_t

    def train(self):
        cfg = self.config
        device = self.device

        # Phase 1: pretrain G with L1 only (reconstruction on missing positions)
        if cfg.pretrain_epochs > 0:
            print("=== Pretraining generator with L1 for {} epochs ===".format(cfg.pretrain_epochs))
            for epoch in range(cfg.pretrain_epochs):
                pbar = tqdm(self.dl, desc=f"pretrain ep{epoch}")
                for batch in pbar:
                    x = batch['x']  # (B,L,D)
                    mask = batch['mask']  # (B,L,D)
                    B, L, D = x.shape
                    inp, agg_mask = self.prepare_input(x, mask)
                    self.G.train()
                    pred = self.G(inp, agg_mask)  # (B, C, L)
                    # recover predicted features (permute back)
                    pred_t = pred.permute(0, 2, 1).cpu().detach().numpy()
                    # compute L1 only on missing positions (where mask==0)
                    mask_any = mask[:, :, self.ds.feature_cols.index(cfg.power_col)]
                    mask_miss = (mask_any == 0).to(torch.float32)
                    # get power channel index in features (0..D-1)
                    pidx = self.ds.feature_cols.index(cfg.power_col)
                    # extract predicted power channel
                    pred_power = pred.permute(0, 2, 1)[:, :, pidx]  # (B,L)
                    # target true (normalized) power from dataset (note original missing are zeros in normed)
                    target_power = torch.from_numpy(x[:, :, pidx]).to(device)
                    l1 = torch.abs(pred_power * 1.0 - target_power) * torch.from_numpy(mask_miss).to(device)
                    loss = l1.sum() / (mask_miss.sum() + 1e-8)
                    self.optG.zero_grad()
                    loss.backward()
                    self.optG.step()
                    pbar.set_postfix(L1=loss.item())
                # save checkpoint
                torch.save({'G': self.G.state_dict()}, os.path.join(cfg.ckpt_dir, f'pretrain_g_ep{epoch}.pt'))

        # Phase 2: adversarial training (WGAN-GP)
        print("=== Adversarial training ===")
        for epoch in range(cfg.epochs):
            pbar = tqdm(self.dl, desc=f"adv ep{epoch}")
            for batch in pbar:
                x = batch['x']  # (B,L,D)
                mask = batch['mask']  # (B,L,D)
                B, L, D = x.shape
                inp, agg_mask = self.prepare_input(x, mask)  # (B,C,L)
                inp_real = inp  # real input (with observed values in feature channels)
                # create "corrupted" input for generator: keep observed values, set missing features to 0 in feature channels,
                # note: dataset already has zeros in missing positions in normalized data
                corrupted = inp_real.clone()

                # -------------------
                # Train Discriminators (n_critic times)
                # -------------------
                for _ in range(cfg.n_critic):
                    # Generate fake
                    self.G.train(); self.Dglob.train(); self.Dlocal.train()
                    fake = self.G(corrupted, agg_mask)  # (B,C,L)
                    # Detach
                    fake_det = fake.detach()

                    # D outputs
                    real_score_g = self.Dglob(inp_real)
                    fake_score_g = self.Dglob(fake_det)
                    real_score_l = self.Dlocal(inp_real)
                    fake_score_l = self.Dlocal(fake_det)

                    # WGAN loss
                    d_loss = fake_score_g.mean() + fake_score_l.mean() - real_score_g.mean() - real_score_l.mean()

                    # gradient penalty (use global discriminator for gp)
                    gp = gradient_penalty(self.Dglob, inp_real, fake_det, device=device)
                    d_loss_total = d_loss + cfg.gp_lambda * gp

                    self.optD.zero_grad()
                    d_loss_total.backward()
                    self.optD.step()

                # -------------------
                # Train Generator
                # -------------------
                self.G.train(); self.Dglob.eval(); self.Dlocal.eval()
                fake = self.G(corrupted, agg_mask)
                # adversarial scores (want to maximize D(fake) => minimize -D(fake))
                fake_score_g = self.Dglob(fake)
                fake_score_l = self.Dlocal(fake)
                adv_loss = - (fake_score_g.mean() + fake_score_l.mean()) * 0.5

                # reconstruction L1 on missing positions (power channel)
                pidx = self.ds.feature_cols.index(cfg.power_col)
                pred_power = fake.permute(0, 2, 1)[:, :, pidx]  # (B,L)
                target_power = torch.from_numpy(x[:, :, pidx]).to(device)
                mask_any = mask[:, :, pidx]  # (B,L)
                mask_miss = (mask_any == 0).astype(np.float32)
                mask_miss_t = torch.from_numpy(mask_miss).to(device)
                rec_l1 = torch.abs(pred_power - target_power) * mask_miss_t
                rec_loss = rec_l1.sum() / (mask_miss_t.sum() + 1e-8)

                # feature matching: get intermediate features from Dglob by a simple hook - here we'll approximate by using final scalar
                # (A full FM would capture intermediate layers; for brevity we add small L2 between statistics)
                # compute mean features
                fm = ((fake.mean(dim=[1,2]) - inp_real.mean(dim=[1,2])).pow(2)).mean()

                # gradient (temporal) loss on predicted power (first diff)
                dpred = pred_power[:, 1:] - pred_power[:, :-1]
                dtarget = target_power[:, 1:] - target_power[:, :-1]
                grad_loss = torch.abs(dpred - dtarget) * mask_miss_t[:, 1:]
                grad_loss = grad_loss.sum() / (mask_miss_t[:, 1:].sum() + 1e-8)

                g_loss = cfg.lambda_adv * adv_loss + cfg.lambda_rec * rec_loss + cfg.lambda_fm * fm + cfg.lambda_grad * grad_loss

                self.optG.zero_grad()
                g_loss.backward()
                self.optG.step()

                self.iter += 1
                pbar.set_postfix(Dloss=d_loss.item(), Gloss=g_loss.item(), Rec=rec_loss.item())

            # save checkpoints each epoch
            torch.save({'G': self.G.state_dict(),
                        'Dglob': self.Dglob.state_dict(),
                        'Dlocal': self.Dlocal.state_dict()},
                       os.path.join(cfg.ckpt_dir, f'ckpt_ep{epoch}.pt'))

        print("Training finished.")

# -------------------------
# Config and main
# -------------------------
class Config:
    def __init__(self):
        self.csv = "data/weather_data_with_missing.csv"         # set to your csv
        self.seq_len = 128
        self.batch_size = 16
        self.power_col = "Power"
        self.base_ch = 48
        self.lr_g = 1e-4
        self.lr_d = 4e-4
        self.epochs = 5               # set higher for real training
        self.pretrain_epochs = 1
        self.n_critic = 3
        self.gp_lambda = 10.0
        self.aug_max_block = 64
        self.ckpt_dir = "checkpoints"
        # loss weights
        self.lambda_adv = 1.0
        self.lambda_rec = 200.0
        self.lambda_fm = 1.0
        self.lambda_grad = 5.0
        self.seed = 42

def main():
    cfg = Config()
    # If you want to test with the example CSV snippet, create a small CSV file named data.csv in the same folder.
    # Example CSV (from your message):
    # timestamp,AltimeterSetting,DewPointTemperature,Temperature,Precipitation,RelativeHumidity,Pressure,Visibility,WindSpeed,Power
    # 2017/8/1 0:00,,60.0,,0.049,65.77609,29.05,,,0.0
    # 2017/8/1 0:01,,60.0,,,71.69948,,9.975,3.627,0.0
    # 2017/8/1 0:02,29.3,58.0,59.84137,0.0764,70.62198,29.28,,4.171,
    # 2017/8/1 0:03,28.69,57.0,,,,,10.703,,0.0
    #
    # Note: missing values in CSV should be blank or empty; pandas will parse them as NaN.
    #
    # Set cfg.csv to your actual CSV path before running
    if not os.path.exists(cfg.csv):
        print(f"CSV {cfg.csv} not found in current folder. Create one (e.g. using your snippet) and re-run.")
        return

    trainer = Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
