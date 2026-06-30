# ═══════════════════════════════════════════════════════════════════════════════
# DiPPeR Inference — DDIM Sampler (ddim-algo)
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Setup ──────────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import subprocess
subprocess.run(['pip', 'install', 'diffusers', 'einops', '-q'])
subprocess.run(['unzip', '-q', '/content/drive/MyDrive/diffusion/data.zip',
                '-d', '/content/'], capture_output=True)

# ── 2. Imports ────────────────────────────────────────────────────────────────
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math, os, time, random
import torch
import torch.nn as nn
import torchvision
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import interp1d

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ── 3. Config ─────────────────────────────────────────────────────────────────
DATA_DIR            = '/content/data'
CKPT_DIR            = '/content/drive/MyDrive/dipper_checkpoints/original_v1'
RESUME_CKPT         = f'{CKPT_DIR}/dipper_orig_epoch_150.pth'
PRED_HORIZON        = 160
OBS_HORIZON         = 1
IMG_SIZE            = 100
ACTION_DIM          = 2
VISION_DIM          = 512
START_DIM           = 2
GOAL_DIM            = 2
OBS_DIM             = VISION_DIM + START_DIM + GOAL_DIM  # 516
BATCH_SIZE          = 64
NUM_EPOCHS          = 0
LR                  = 1e-4
SAVE_EVERY          = 10
NUM_WORKERS         = 2
MAX_SAMPLES         = 50000
NUM_DIFFUSION_ITERS = 100
DDIM_STEPS          = 50   # change to 50 for DDIM-50
STATS               = {'min': -5., 'max': 5.}

os.makedirs(CKPT_DIR, exist_ok=True)

# ── 4. Normalization helpers ──────────────────────────────────────────────────
def normalize_data(data, stats):
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    return ndata * 2 - 1

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    return ndata * (stats['max'] - stats['min']) + stats['min']

# ── 5. Dataset ────────────────────────────────────────────────────────────────
class Planning2DDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, action_horizon=160, max_samples=50000):
        self.dataset_path = dataset_path
        self.stats = STATS

        self.file_indices = {}
        idx = 0
        for file in os.listdir(dataset_path):
            if file.endswith('.npy'):
                self.file_indices[idx] = file
                idx += 1
        self.file_len = idx
        print(f'Total trajectory files: {self.file_len}')

        self.indices = self._make_indices(action_horizon, max_samples)
        print(f'Total training samples: {len(self.indices)}')

    def _make_indices(self, horizon, max_samples):
        indices = []
        for file_idx in range(self.file_len):
            fname = self.file_indices[file_idx]
            traj_path = os.path.join(self.dataset_path, fname)
            try:
                path_length = np.load(traj_path).shape[1]
            except:
                continue
            max_start = path_length - horizon
            if max_start <= 0:
                continue
            for start in range(max_start):
                end = start + horizon
                indices.append((file_idx, start, end))
                if len(indices) >= max_samples:
                    return np.array(indices)
        return np.array(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, start, end = self.indices[idx]
        traj_fname = self.file_indices[file_idx]
        traj_path  = os.path.join(self.dataset_path, traj_fname)

        traj_array = np.load(traj_path)[:, start:end].T
        traj_array = np.float32(traj_array)
        action     = normalize_data(traj_array, self.stats)

        nsample = {}
        nsample['action'] = action
        nsample['start']  = np.expand_dims(action[0, :],  axis=0)
        nsample['goal']   = np.expand_dims(action[-1, :], axis=0)

        image_num   = traj_fname.split('_')[1]
        image_fname = f'maze_occu_{image_num}.png'
        image_path  = os.path.join(self.dataset_path, image_fname)

        img_array = cv2.imread(image_path)
        if img_array is None:
            img_array = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        else:
            img_array = np.array(img_array, dtype=np.float32)

        img_array = np.moveaxis(img_array, -1, 0)
        nsample['image'] = np.expand_dims(img_array, axis=0)

        return nsample

# ── 6. Network — EXACT original DiPPeR ───────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device   = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)
    def forward(self, x): return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)
    def forward(self, x): return self.conv(x)

class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )
    def forward(self, x): return self.block(x)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels,  out_channels, kernel_size, n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
        ])
        cond_channels     = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out   = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        out   = embed[:, 0] * out + embed[:, 1]
        out   = self.blocks[1](out)
        return out + self.residual_conv(x)

class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim,
                 diffusion_step_embed_dim=256,
                 down_dims=[256, 512, 1024],
                 kernel_size=5, n_groups=8):
        super().__init__()
        all_dims  = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed      = diffusion_step_embed_dim
        cond_dim  = dsed + global_cond_dim
        in_out    = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim   = all_dims[-1]

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
        ])

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in,  dim_out, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        print(f'ConditionalUnet1D parameters: {sum(p.numel() for p in self.parameters()):,}')

    def forward(self, sample, timestep, global_cond=None):
        sample    = sample.moveaxis(-1, -2)
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x, h = sample, []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        return self.final_conv(x).moveaxis(-1, -2)

# ── 7. Vision encoder ─────────────────────────────────────────────────────────
def get_resnet(name='resnet18', weights=None):
    resnet = getattr(torchvision.models, name)(weights=weights)
    resnet.fc = nn.Identity()
    return resnet

def replace_submodules(root_module, predicate, func):
    if predicate(root_module):
        return func(root_module)
    bn_list = [k.split('.') for k, m
               in root_module.named_modules(remove_duplicate=True)
               if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    assert len([k for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]) == 0
    return root_module

def replace_bn_with_gn(root_module, features_per_group=16):
    return replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group,
            num_channels=x.num_features)
    )

# ── 8. Build dataset ──────────────────────────────────────────────────────────
print('Building dataset...')
dataset = Planning2DDataset(DATA_DIR, action_horizon=PRED_HORIZON, max_samples=MAX_SAMPLES)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
)
print(f'Batches/epoch: {len(dataloader)}')

# ── 9. Build network ──────────────────────────────────────────────────────────
vision_encoder = replace_bn_with_gn(get_resnet('resnet18'))
noise_pred_net = ConditionalUnet1D(
    input_dim=ACTION_DIM,
    global_cond_dim=OBS_DIM * OBS_HORIZON,
    diffusion_step_embed_dim=256,
    down_dims=[256, 512, 1024],
    kernel_size=5, n_groups=8,
)
nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net,
}).to(DEVICE)

ema = EMAModel(parameters=nets.parameters(), power=0.75)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=NUM_DIFFUSION_ITERS,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True, prediction_type='epsilon',
)

optimizer = torch.optim.AdamW(nets.parameters(), lr=LR, weight_decay=1e-6)
lr_scheduler = get_scheduler(
    name='cosine', optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * max(NUM_EPOCHS, 1),
)
print(f'Total parameters: {sum(p.numel() for p in nets.parameters()):,}')

# ── 10. Load checkpoint ───────────────────────────────────────────────────────
start_epoch = 1
if RESUME_CKPT and os.path.exists(RESUME_CKPT):
    ckpt = torch.load(RESUME_CKPT, map_location=DEVICE, weights_only=False)
    nets.load_state_dict(ckpt['nets'])
    ema = EMAModel(parameters=nets.parameters(), power=0.75)
    start_epoch = ckpt['epoch'] + 1
    print(f'Loaded checkpoint: epoch {ckpt["epoch"]}, loss {ckpt["loss"]:.5f}')
else:
    print('Checkpoint not found!')

# ── 11. Skip training ─────────────────────────────────────────────────────────
print('Skipping training — inference only (DDIM)')

# ── 12. Loss curve — skipped ──────────────────────────────────────────────────
print('Skipping loss curve (no training this run)')

# ── 13. Visualization — DDIM ─────────────────────────────────────────────────
nets.eval()

ddim_scheduler = DDIMScheduler(
    num_train_timesteps=NUM_DIFFUSION_ITERS,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon',
)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for i in range(8):
    ax = axes[i // 4][i % 4]
    while True:
        sample_idx = random.randint(0, len(dataset) - 1)
        sample     = dataset[sample_idx]
        s    = unnormalize_data(sample['start'].squeeze(), STATS)
        g    = unnormalize_data(sample['goal'].squeeze(),  STATS)
        dist = np.linalg.norm(s - g)
        if dist < 3.0:
            break

    file_idx   = dataset.indices[sample_idx][0]
    traj_fname = dataset.file_indices[file_idx]
    img_num    = traj_fname.split('_')[1]

    nimage  = torch.tensor(sample['image'][:OBS_HORIZON], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    nstart  = torch.tensor(sample['start'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    ngoal   = torch.tensor(sample['goal'],  dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = nets['vision_encoder'](nimage.flatten(end_dim=1))
        image_features = image_features.reshape(*nimage.shape[:2], -1)
        obs_features   = torch.cat([image_features, nstart, ngoal], dim=-1)
        obs_cond       = obs_features.flatten(start_dim=1)

        naction = torch.randn((1, PRED_HORIZON, ACTION_DIM), device=DEVICE)
        naction[0, 0, :]  = nstart[0, 0, :]
        naction[0, -1, :] = ngoal[0, 0, :]

        ddim_scheduler.set_timesteps(DDIM_STEPS)

        for k in ddim_scheduler.timesteps:
            naction[0, 0, :]  = nstart[0, 0, :]
            naction[0, -1, :] = ngoal[0, 0, :]
            noise_pred = nets['noise_pred_net'](sample=naction, timestep=k, global_cond=obs_cond)
            naction[0, 0, :]  = nstart[0, 0, :]
            naction[0, -1, :] = ngoal[0, 0, :]
            naction = ddim_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
            naction[0, 0, :]  = nstart[0, 0, :]
            naction[0, -1, :] = ngoal[0, 0, :]

    traj_real = unnormalize_data(naction[0].cpu().numpy(), STATS)
    origin    = np.array([-5., -5.]).reshape(1, 2)
    t_px      = np.clip((traj_real - origin) / 0.1, 0, 99)
    s_px      = np.clip((unnormalize_data(nstart[0, 0].cpu().numpy(), STATS) - origin.flatten()) / 0.1, 0, 99)
    g_px      = np.clip((unnormalize_data(ngoal[0, 0].cpu().numpy(),  STATS) - origin.flatten()) / 0.1, 0, 99)

    map_img_raw = cv2.imread(os.path.join(DATA_DIR, f'maze_occu_{img_num}.png'), 0)
    if map_img_raw is None:
        map_img_raw = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    map_show = 255 - map_img_raw

    ax.imshow(map_show, cmap='gray')
    ax.scatter(t_px[:, 0], t_px[:, 1], s=2, c=range(len(t_px)), cmap='autumn')
    ax.scatter(s_px[0], s_px[1], marker='o', s=80, c='green', zorder=5)
    ax.scatter(g_px[0], g_px[1], marker='x', s=80, c='blue',  zorder=5)
    ax.set_title(f'Sample {i+1}')
    ax.axis('off')

plt.suptitle(f'DiPPeR DDIM ({DDIM_STEPS} steps) — epoch {ckpt["epoch"]} | loss={ckpt["loss"]:.5f}', fontsize=14)
plt.tight_layout()
out = f'{CKPT_DIR}/ddim_algo_epoch_{ckpt["epoch"]}.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved: {out}')