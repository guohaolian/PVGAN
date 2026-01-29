import os
import math
import random
import argparse
from typing import Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# -------------------------
# Utility Functions
# -------------------------
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


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
    A dataset for PV time series with random block missingness augmentation
    on the 'Power' column. We separate:

      - orig_mask: 原始数据是否有值 (NaN -> 0, 有值 -> 1)
      - aug_mask: 仅对 Power 列做随机遮挡，用于训练监督
    """
    def __init__(self,
                 csv_path: str,
                 seq_len=288,
                 power_col="Power",
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

        # 正确筛选特征列：排除各种时间列
        if feature_cols is None:
            candidate = [c for c in df.columns
                         if c.lower() not in ("timestamp", "time", "date", "datetime")]
            feature_cols = [c for c in candidate if c in df.columns]
        self.feature_cols = feature_cols

        # 只对数值/可转为 float 的列做 replace + astype
        data = df[self.feature_cols].replace(r'^\s*$', np.nan, regex=True).astype(float).values

        # raw: keep NaN
        self.raw = data.copy()  # (T, D)
        self.T, self.D = self.raw.shape

        # Original mask: 1 = 有值, 0 = NaN
        self.orig_mask = (~np.isnan(self.raw)).astype(np.float32)

        # Fill NaN with 0 for normalization
        self.raw_filled = np.nan_to_num(self.raw, nan=0.0)

        # Compute mean & std ignoring NaN
        self.mean = np.nanmean(self.raw, axis=0)
        self.std = np.nanstd(self.raw, axis=0)
        self.std[self.std < 1e-6] = 1.0

        # Standardize the data and replace NaN with 0
        self.normed = ((self.raw_filled - self.mean[None, :]) / self.std[None, :]).astype(np.float32)
        self.raw_filled = self.raw_filled.astype(np.float32)
        self.orig_mask = self.orig_mask.astype(np.float32)

        # Build index of all possible sequence starts
        self.starts = []
        max_start = max(1, self.T - self.seq_len + 1)
        for s in range(0, max_start):
            self.starts.append(s)

    def __len__(self):
        return len(self.starts)

    def _create_random_block_mask(self):
        """
        Create a random block missing mask on the Power column only.
        1 = observed, 0 = synthetically hidden.
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
            # Tail padding
            avail = self.T - s
            seq = np.zeros((self.seq_len, self.D), dtype=np.float32)
            orig_mask = np.zeros((self.seq_len, self.D), dtype=np.float32)
            raw_slice = np.zeros((self.seq_len, self.D), dtype=np.float32)
            if avail > 0:
                seq[:avail] = self.normed[s:s + avail]
                orig_mask[:avail] = self.orig_mask[s:s + avail]
                raw_slice[:avail] = self.raw_filled[s:s + avail]

        # Augment with synthetic missing blocks in Power column only
        aug_mask = self._create_random_block_mask()

        pidx = self.feature_cols.index(self.power_col)

        # Mask fed into the model (observed positions = 1, missing = 0)
        # = 原始是否有值 AND 未被增强遮挡
        input_mask = orig_mask.copy()
        input_mask[:, pidx] = orig_mask[:, pidx] * aug_mask[:, pidx]

        # Supervision mask for Power:
        # 1 only where original data existed but we intentionally hide it (orig_mask=1 & aug_mask=0).
        # 这些位置有真实 label，可以算 L1 / grad loss。
        # 用来算监督损失的 mask：必须是原来有值但被我们遮挡掉的点
        # 原来有值：orig_mask[:, pidx] == 1
        # 被增强遮挡：aug_mask[:, pidx] == 0
        train_miss_mask_power = ((orig_mask[:, pidx] == 1) & (aug_mask[:, pidx] == 0)).astype(np.float32)

        sample = {
            'x': seq,
            'mask': input_mask,
            'raw_slice': raw_slice,
            'power_idx': pidx,
            'train_miss_mask_power': train_miss_mask_power
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
# Generator (1D U-Net)
# -------------------------
class GeneratorUNet1D(nn.Module):
    def __init__(self, in_channels, feat_ch, base_ch=64):
        """
        in_channels: number of channels in input (features + mask + noise)
        feat_ch: number of feature channels in the output (should = D)
        """
        super().__init__()
        self.enc1 = ConvBlock1D(in_channels, base_ch, kernel=3, stride=1, padding=1)
        self.enc2 = ConvBlock1D(base_ch, base_ch * 2, kernel=4, stride=2, padding=1)
        self.enc3 = ConvBlock1D(base_ch * 2, base_ch * 4, kernel=4, stride=2, padding=1)
        self.enc4 = ConvBlock1D(base_ch * 4, base_ch * 8, kernel=4, stride=2, padding=1)

        self.bottleneck = ConvBlock1D(base_ch * 8, base_ch * 8, kernel=3, stride=1, padding=1)

        self.dec4 = DeconvBlock1D(base_ch * 8, base_ch * 4)
        self.dec3 = DeconvBlock1D(base_ch * 8, base_ch * 2)
        self.dec2 = DeconvBlock1D(base_ch * 4, base_ch)
        self.dec1 = ConvBlock1D(base_ch * 2, feat_ch, kernel=3, stride=1, padding=1, norm=False)

        self.out_act = nn.Identity()

    def forward(self, x, agg_mask=None):
        """
        x: (B, C_in, L)
        agg_mask: ignored in this simple UNet, but kept for extensibility
        """
        e1 = self.enc1(x)       # (B, base, L)
        e2 = self.enc2(e1)      # (B, 2base, L/2)
        e3 = self.enc3(e2)      # (B, 4base, L/4)
        e4 = self.enc4(e3)      # (B, 8base, L/8)

        b = self.bottleneck(e4)

        d4 = self.dec4(b)
        d4 = torch.cat([d4, e3], dim=1)

        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)

        out = self.dec1(d2)
        out = self.out_act(out)
        return out  # (B, feat_ch, L)

# -------------------------
# Discriminators
# -------------------------
class GlobalDiscriminator(nn.Module):
    """
    A global discriminator that looks at the entire sequence.
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
        # global pooling over time
        h = h.mean(dim=2)  # (B, C)
        out = self.fc(h)
        return out  # (B, 1)


class LocalPatchDiscriminator(nn.Module):
    """
    A local patch discriminator focusing on smaller windows.
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
# Training & Imputation
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

        # 区分 G 和 D 的输入通道数
        in_ch_G = ds.D + 1 + 1  # features + mask(agg) + noise
        in_ch_D = ds.D + 1      # features + noise（判别器里我们只拼了这两个）

        self.G = GeneratorUNet1D(in_channels=in_ch_G,
                                 feat_ch=self.ds.D,
                                 base_ch=config.base_ch).to(self.device)

        self.Dglob = GlobalDiscriminator(in_channels=in_ch_D,
                                         base_ch=config.base_ch).to(self.device)

        self.Dlocal = LocalPatchDiscriminator(in_channels=in_ch_D,
                                              base_ch=config.base_ch).to(self.device)

        self.optG = Adam(self.G.parameters(), lr=config.lr_g, betas=(0.5, 0.9))
        self.optD = Adam(list(self.Dglob.parameters()) + list(self.Dlocal.parameters()), lr=config.lr_d,
                         betas=(0.5, 0.9))

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
        We create in_channels = D + 1 (mask) + 1 (noise).
        """
        B, L, D = x.shape
        x_t = self.ensure_tensor(x).permute(0, 2, 1)          # (B, D, L)
        mask_t = self.ensure_tensor(mask).permute(0, 2, 1)    # (B, D, L)

        # Aggregate mask for a single channel; here we keep all dims
        agg_mask = mask_t.mean(dim=1, keepdim=True)  # (B,1,L) average observed ratio

        noise = self.sample_noise(B, L)  # (B,1,L)

        inp = torch.cat([x_t, agg_mask, noise], dim=1)  # (B, D+2, L)
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
                    pred = self.G(inp, agg_mask)  # (B,C,L)

                    pidx = self.ds.feature_cols.index(cfg.power_col)
                    pred_power = pred.permute(0, 2, 1)[:, :, pidx]
                    target_power = self.ensure_tensor(x[:, :, pidx])

                    # Only use positions where we synthetically hid an originally valid Power value
                    train_miss_t = self.ensure_tensor(train_miss)

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

                train_miss = batch['train_miss_mask_power']
                train_miss_t = self.ensure_tensor(train_miss)

                # Reconstruction loss only on synthetically hidden-but-valid Power positions
                rec_l1 = torch.abs(pred_power - target_power) * train_miss_t
                rec_loss = rec_l1.sum() / (train_miss_t.sum() + 1e-8)

                fm = ((fake.mean(dim=[1, 2]) - inp_real.mean(dim=[1, 2])).pow(2)).mean()

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
                pbar.set_postfix(Dloss=d_loss.item(), Gloss=g_loss.item(), Rec=rec_loss.item())

            torch.save({'G': self.G.state_dict()},
                       os.path.join(cfg.ckpt_dir, f'ckpt_ep{epoch}.pt'))

    # -------------------------
    # Imputation on a full csv
    # -------------------------
    def impute(self, input_csv, ckpt_path, out_csv=None):
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

                # Denormalize
                fake_denorm = fake_np * self.ds.std[None, :] + self.ds.mean[None, :]

                # Where mask==0, we fill with generated values
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
        out_csv = args.impute_out_csv or "pv_data_filled.csv"
        trainer.impute(input_csv, ckpt_path, out_csv)


if __name__ == "__main__":
    main()
