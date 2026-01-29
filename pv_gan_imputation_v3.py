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
    def __init__(self, csv_path: str, seq_len=256, power_col='Power', feature_cols=None,
                 create_aug_masks=True, aug_prob=0.5, aug_max_block=64):
        df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.power_col = power_col
        if feature_cols is None:
            candidate = [c for c in df.columns if c.lower() not in ('timestamp', 'time', 'date')]
            feature_cols = [c for c in candidate if c in df.columns]
        self.feature_cols = feature_cols

        # Fill empty cells with NaN for Power column and ensure others are 0 for valid values
        data = df[self.feature_cols].replace(r'^\s*$', np.nan, regex=True).astype(float).values
        self.raw = data
        self.T, self.D = self.raw.shape

        # Masks: observed = 1, missing = 0
        self.orig_mask = (~np.isnan(self.raw)).astype(float)
        self.raw_filled = np.nan_to_num(self.raw, nan=0.0)  # Only replace missing with 0

        # Standardization (use only the non-NaN values for each feature)
        self.mean = np.nanmean(self.raw, axis=0)
        self.std = np.nanstd(self.raw, axis=0) + 1e-6
        nan_mean = np.isnan(self.mean)
        if nan_mean.any():
            self.mean[nan_mean] = 0.0
            self.std[nan_mean] = 1.0

        # Standardize the data and replace NaN with 0
        normed = (self.raw - self.mean[None, :]) / self.std[None, :]
        self.normed = np.where(np.isnan(normed), 0.0, normed).astype(np.float32)

        self.orig_mask = self.orig_mask.astype(np.float32)
        self.raw_filled = np.where(np.isnan(self.raw), 0.0, self.raw).astype(np.float32)

        self.create_aug_masks = create_aug_masks
        self.aug_prob = aug_prob
        self.aug_max_block = aug_max_block
        self.starts = []
        max_start = max(1, self.T - self.seq_len + 1)
        for s in range(0, max_start):
            self.starts.append(s)

    def __len__(self):
        return len(self.starts)

    def _create_random_block_mask(self):
        mask = np.ones((self.seq_len, self.D), dtype=np.float32)
        if random.random() < self.aug_prob:
            nblocks = random.randint(1, 3)
            for _ in range(nblocks):
                block_len = random.randint(1, int(min(self.aug_max_block, max(1, self.seq_len // 2))))
                start = random.randint(0, max(0, self.seq_len - block_len))
                mask[start:start + block_len, self.feature_cols.index(self.power_col)] = 0.0
        return mask

    def __getitem__(self, idx):
        s = self.starts[idx]
        end = s + self.seq_len
        if end <= self.T:
            seq = self.normed[s:end]  # (seq_len, D)
            mask = self.orig_mask[s:end]
            raw_slice = self.raw_filled[s:end]
        else:
            avail = self.T - s
            seq = np.zeros((self.seq_len, self.D), dtype=np.float32)
            mask = np.zeros((self.seq_len, self.D), dtype=np.float32)
            raw_slice = np.zeros((self.seq_len, self.D), dtype=np.float32)
            if avail > 0:
                seq[:avail] = self.normed[s:s + avail]
                mask[:avail] = self.orig_mask[s:s + avail]
                raw_slice[:avail] = self.raw_filled[s:s + avail]

        # Augmenting with missing blocks in Power column only
        aug_mask = self._create_random_block_mask()
        mask = mask * aug_mask

        pidx = self.feature_cols.index(self.power_col)
        sample = {
            'x': seq,
            'mask': mask,
            'raw_slice': raw_slice,
            'power_idx': pidx
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
    def __init__(self, feat_ch, in_channels, base_ch=64, latent_ch=256):
        """
        feat_ch: number of raw feature channels to predict (D)
        in_channels: full input channels (feat + mask + noise)
        """
        super().__init__()
        self.feat_ch = feat_ch

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

        # Output only feature corrections for feat_ch channels (not mask/noise)
        self.out_conv = nn.Conv1d(base_ch, feat_ch, kernel_size=1)

        # small gating convs
        self.gate1 = GatedConv1D(in_channels, base_ch)
        self.gate2 = GatedConv1D(base_ch, base_ch*2)
        self.gate3 = GatedConv1D(base_ch*2, base_ch*4)

    def forward(self, x, mask):
        """
        x: (B, C, L) normalized input (missing filled with zeros), C=in_channels
        mask: (B, 1, L) mask for target channel(s) (1 observed, 0 missing)
        We'll concat mask as an extra channel outside.
        """
        # Past branch (normal order)
        p1 = self.enc1_p(x)
        p1g = self.gate1(x)  # gate from raw input (unused but kept)
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

        # concat features along channel (skip connections use deepest)
        fused = torch.cat([p3, f3], dim=1)
        z = self.fuse(fused)

        d3 = self.dec3(z) + p3  # skip add
        d2 = self.dec2(d3) + p2
        d1 = self.dec1(d2) + p1

        out_feat = self.out_conv(d1)  # (B, feat_ch, L)
        # We output corrections for feature channels; leave mask/noise channels untouched upstream.
        return out_feat

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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.config = config
        seed_all(config.seed)

        # Dataset & Dataloader
        ds = PVTimeSeriesDataset(config.csv, seq_len=config.seq_len, power_col=config.power_col,
                                 create_aug_masks=True, aug_prob=0.6, aug_max_block=config.aug_max_block)
        self.ds = ds
        self.dl = DataLoader(ds, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=0)

        in_ch = ds.D + 1 + 1  # features + mask + noise
        self.G = GeneratorUNet1D(in_channels=in_ch, feat_ch=self.ds.D, base_ch=config.base_ch).to(self.device)
        self.Dglob = GlobalDiscriminator(in_channels=in_ch, base_ch=config.base_ch).to(self.device)
        self.Dlocal = LocalPatchDiscriminator(in_channels=in_ch, base_ch=config.base_ch).to(self.device)

        self.optG = Adam(self.G.parameters(), lr=config.lr_g, betas=(0.5, 0.9))
        self.optD = Adam(list(self.Dglob.parameters()) + list(self.Dlocal.parameters()), lr=config.lr_d,
                         betas=(0.5, 0.9))

        makedirs(config.ckpt_dir)
        self.iter = 0

    def sample_noise(self, B, L):
        return torch.randn(B, 1, L, device=self.device)

    def ensure_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return torch.from_numpy(data).to(self.device)

    def prepare_input(self, x_np, mask_np):
        B, L, D = x_np.shape
        x = self.ensure_tensor(x_np)
        mask = self.ensure_tensor(mask_np)

        power_idx = self.ds.feature_cols.index(self.config.power_col)
        mask_power = mask[:, :, power_idx:power_idx + 1]  # (B,L,1)
        noise = torch.randn(B, L, 1, device=self.device)

        inp = torch.cat([x, mask_power, noise], dim=2).permute(0, 2, 1)
        return inp, mask_power.permute(0, 2, 1)

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
                    B, L, D = x.shape
                    inp, agg_mask = self.prepare_input(x, mask)

                    self.G.train()
                    pred = self.G(inp, agg_mask)  # (B,C,L)

                    pidx = self.ds.feature_cols.index(cfg.power_col)
                    pred_power = pred.permute(0, 2, 1)[:, :, pidx]
                    target_power = self.ensure_tensor(x[:, :, pidx])

                    mask_any = mask[:, :, pidx]
                    mask_miss = (mask_any == 0).float()
                    mask_miss_t = self.ensure_tensor(mask_miss)

                    l1 = torch.abs(pred_power - target_power) * mask_miss_t
                    loss = l1.sum() / (mask_miss_t.sum() + 1e-8)

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
                B, L, D = x.shape
                inp, agg_mask = self.prepare_input(x, mask)
                inp_real = inp.clone()

                # ------------------- Train Discriminators -------------------
                for _ in range(cfg.n_critic):
                    fake = self.G(inp_real, agg_mask)
                    fake_det = fake.detach()

                    real_feat = inp_real[:, :self.ds.D, :]
                    fake_full_D = torch.cat([fake_det, self.sample_noise(B, L)], dim=1)
                    real_full_D = torch.cat([real_feat, self.sample_noise(B, L)], dim=1)

                    real_score_g = self.Dglob(real_full_D)
                    fake_score_g = self.Dglob(fake_full_D)
                    real_score_l = self.Dlocal(real_full_D)
                    fake_score_l = self.Dlocal(fake_full_D)

                    d_loss = fake_score_g.mean() + fake_score_l.mean() - real_score_g.mean() - real_score_l.mean()

                    gp = gradient_penalty(self.Dglob, real_full_D, fake_full_D, device=device)
                    d_loss_total = d_loss + cfg.gp_lambda * gp

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
                pred_power = fake.permute(0, 2, 1)[:, :, pidx]
                target_power = self.ensure_tensor(x[:, :, pidx])

                mask_any = mask[:, :, pidx]
                mask_miss = (mask_any == 0).float()
                mask_miss_t = self.ensure_tensor(mask_miss)

                rec_l1 = torch.abs(pred_power - target_power) * mask_miss_t
                rec_loss = rec_l1.sum() / (mask_miss_t.sum() + 1e-8)

                fm = ((fake.mean(dim=[1, 2]) - inp_real.mean(dim=[1, 2])).pow(2)).mean()

                dpred = pred_power[:, 1:] - pred_power[:, :-1]
                dtarget = target_power[:, 1:] - target_power[:, :-1]
                grad_loss = torch.abs(dpred - dtarget) * mask_miss_t[:, 1:]
                grad_loss = grad_loss.sum() / (mask_miss_t[:, 1:].sum() + 1e-8)

                g_loss = (cfg.lambda_adv * adv_loss +
                          cfg.lambda_rec * rec_loss +
                          cfg.lambda_fm * fm +
                          cfg.lambda_grad * grad_loss)

                self.optG.zero_grad()
                g_loss.backward()
                self.optG.step()

                self.iter += 1
                pbar.set_postfix(Dloss=d_loss.item(), Gloss=g_loss.item(), Rec=rec_loss.item())

            torch.save({'G': self.G.state_dict(),
                        'Dglob': self.Dglob.state_dict(),
                        'Dlocal': self.Dlocal.state_dict()},
                       os.path.join(cfg.ckpt_dir, f'ckpt_ep{epoch}.pt'))

        print("Training finished.")

    def impute_csv(self, input_csv: str, output_csv: str, ckpt_path: str):
        device = self.device

        # Load checkpoint
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
        normed = normed.astype(np.float32)
        mask = mask.astype(np.float32)

        seq_len = self.config.seq_len
        step = seq_len // 2
        T, D = normed.shape
        filled_normed = normed.copy()

        with torch.no_grad():
            for start in tqdm(range(0, T, step)):
                end = min(T, start + seq_len)
                seq = np.zeros((seq_len, D), np.float32)
                msk = np.zeros((seq_len, D), np.float32)
                avail = end - start
                seq[:avail] = normed[start:end]
                msk[:avail] = mask[start:end]

                inp, agg_mask = self.prepare_input(seq[None, :, :], msk[None, :, :])
                pred = self.G(inp, agg_mask).cpu().numpy()[0].transpose(1, 0)  # (L,D)

                # Replace missing positions in Power only
                filled_normed[start:end][msk[:avail, self.ds.feature_cols.index(self.config.power_col)] == 0] = \
                pred[:avail][msk[:avail, self.ds.feature_cols.index(self.config.power_col)] == 0]

        filled_real = filled_normed * self.ds.std[None, :] + self.ds.mean[None, :]
        df_filled = df.copy()
        df_filled[feature_cols] = filled_real
        df_filled.to_csv(output_csv, index=False)

        print(f"✅ Imputed CSV saved to: {output_csv}")


# -------------------------
# Config and main
# -------------------------
class Config:
    def __init__(self):
        self.csv = "data/weather_data_with_missing.csv"
        self.seq_len = 128
        self.batch_size = 128
        self.power_col = "Power"
        self.base_ch = 48
        self.lr_g = 1e-4
        self.lr_d = 4e-4
        self.epochs = 3
        self.pretrain_epochs = 5
        self.n_critic = 3
        self.gp_lambda = 10.0
        self.aug_max_block = 64
        self.ckpt_dir = "checkpoints"
        self.lambda_adv = 1.0
        self.lambda_rec = 200.0
        self.lambda_fm = 1.0
        self.lambda_grad = 5.0
        self.seed = 42


def main():
    cfg = Config()
    if not os.path.exists(cfg.csv):
        print(f"CSV {cfg.csv} not found in current folder. Create one (e.g. using your snippet) and re-run.")
        return

    trainer = Trainer(cfg)
    trainer.train()

    input_csv = "data/weather_data_with_missing.csv"
    output_csv = "data/weather_data_filled.csv"
    ckpt_path = "checkpoints/ckpt_ep2.pt"  # Replace with your trained checkpoint path

    trainer.impute_csv(input_csv, output_csv, ckpt_path)


if __name__ == "__main__":
    main()
