import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
import numpy as np

# ── 58-channel layout (matching CHANNEL_DICT in EEGAtlas.py) ──────────────
CHANNEL_NAMES_58 = [
    'FP1', 'FPZ', 'FP2',
    'AF3', 'AF4',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
    'O1', 'OZ', 'O2',
]

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self, img_size=(64, 1000), patch_size=16, patch_stride=None, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        if patch_stride is None:
            self.num_patches = ((img_size[0]), (img_size[1] // patch_size))
        else:
            self.num_patches = ((img_size[0]), ((img_size[1] - patch_size) // patch_stride + 1))

        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(1, patch_size),
                              stride=(1, patch_size if patch_stride is None else patch_stride))

    def forward(self, x):
        # x: B,C,T
        x = x.unsqueeze(1)  # B, 1, C, T
        x = self.proj(x).transpose(1, 3)  # B, T, C, D
        return x


class ConvEmbed(nn.Module):
    """CSBrain 风格"短步长卷积降维" (NewVersion2 增强项).

    独立于 PatchEmbed, 直接从原始 x (B, C, T) 出发:
      1. 按 T=64 无重叠切片得到 (B, N, C, 64) — N = T/64
      2. 3 层 Conv1d(k=3, s=2, p=1) + GN + GELU 短步长降维:
           第 1 层: 1→64 通道, T 64→32
           第 2 层: 64→64,     T 32→16
           第 3 层: 64→64,     T 16→8
      3. T(8) × hidden(64) = 512, reshape 直接得到 embed_dim
         — 8*64=512 完美匹配 embed_dim, 无需 Linear
    输出 (B, N, C, D), 与 PatchEmbed 输出同形, 残差相加.

    粒度: 第 1 层后最细粒度 = 12ms @ 256Hz; 末了 8 节点 ≈ 47ms 粒度覆盖整段 250ms.
    物理意义: 给 PatchEmbed 的每个 token 注入短步长局部时序上下文.
    """

    def __init__(self, img_size=(58, 1024), embed_dim=512, patch_t=64, hidden=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_t = patch_t
        C, T = img_size
        assert T % patch_t == 0, f"T ({T}) 必须能被 patch_t ({patch_t}) 整除"
        self.N = T // patch_t  # 1024 / 64 = 16
        self.hidden = hidden

        # 3 层 Conv1d(k=3, s=2, p=1) 短步长降维 T 64→32→16→8
        # GroupNorm num_groups=8, hidden=64 -> 8 groups * 8 channels/组
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, hidden, kernel_size=3, stride=2, padding=1, groups=1),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel_size=3, stride=2, padding=1, groups=1),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel_size=3, stride=2, padding=1, groups=1),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
            ),
        ])
        # T 64→32→16→8, 要求 8*hidden = embed_dim
        self.T_out = 8
        assert self.T_out * hidden == embed_dim, \
            f"8*hidden ({self.T_out*hidden}) 必须等于 embed_dim ({embed_dim})"

    def forward(self, x):
        # x: 原始 (B, C, T)
        B, C, T = x.shape
        N = T // self.patch_t
        Tp = self.patch_t

        # 1) 切 patch: (B, C, T) → (B, C, N, Tp) → (B, N, C, Tp)
        x = x.reshape(B, C, N, Tp).permute(0, 2, 1, 3).contiguous()
        # 2) 合并 batch + 视作单通道信号
        x = x.reshape(B * N * C, 1, Tp)  # (B*N*C, 1, 64)

        # 3) 3 层 Conv1d 短步长降维 T 64→32→16→8
        for layer in self.layers:
            x = layer(x)
        # x: (B*N*C, hidden=64, T=8)

        # 4) T(8) × hidden(64) = 512, reshape 直接得到 embed_dim
        #    (B*N*C, hidden, T) → (B*N*C, T, hidden) → (B*N*C, T*hidden=512)
        x = x.permute(0, 2, 1).reshape(B * N * C, self.embed_dim)

        # 5) 还原: (B, N, C, D)
        x = x.reshape(B, N, C, self.embed_dim)
        return x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000, learned_freq=False, interpolate_factor=1.0):
        """
        Rotary Positional Embedding module to encode sequential information into embeddings.

        Parameters:
            dim (int): Dimension of the frequency embedding.
            theta (float): A hyperparameter that influences the scale of the frequency embedding.
            learned_freq (bool): Whether the frequencies are learnable parameters.
            interpolate_factor (float): Scaling factor for interpolated positional encoding.
        """
        super().__init__()
        assert interpolate_factor >= 1.0, "Interpolate factor must be >= 1.0"

        # Initialize frequency parameters
        self.freqs = nn.Parameter(
            1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim)),
            requires_grad=learned_freq)

        self.interpolate_factor = interpolate_factor
        self.cache = {}

    def prepare_freqs(self, num_patches, device='cuda', dtype=torch.float32, offset=0):
        """
        Prepares the frequency embeddings for the given number of patches.

        Parameters:
            num_patches (tuple): Tuple specifying the dimensions (C, N) where
                                 C is the channels and N is the number of positions.
            device (str): Device to store the frequencies on (e.g., 'cuda' or 'cpu').
            dtype (torch.dtype): Data type for the frequencies.
            offset (float): Offset added to position indexes before scaling.

        Returns:
            torch.Tensor: Prepared frequency embeddings with shape [C * N, dim].
        """
        C, N = num_patches
        cache_key = f'freqs:{num_patches}'

        # Return cached result if available
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Generate sequence positions and apply offset and scale
        seq_pos = torch.arange(N, device=device, dtype=dtype).repeat_interleave(repeats=C)
        seq_pos = (seq_pos + offset) / self.interpolate_factor

        # Compute outer product of positions and frequencies, then expand along the last dimension
        freqs_scaled = torch.outer(seq_pos.type(self.freqs.dtype), self.freqs).repeat_interleave(repeats=2, dim=-1)

        # Cache and return the computed frequencies
        self.cache[cache_key] = freqs_scaled
        return freqs_scaled


# ── 3D spatial embedding (MNE-based) ────────────────────────────────────

def get_3d_coordinates(channel_names=None):
    """Extract 3D MRI coordinates from MNE standard_1020 montage.

    Maps channel names (our FP1/FZ/... convention) to MNE positions via
    case-insensitive lookup (MNE uses Fp1/Fz/...).

    Args:
        channel_names: list of channel names. If None, uses CHANNEL_NAMES_58.
    Returns:
        Tensor (num_channels, 3) in MRI meters.
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES_58

    montage = mne.channels.make_standard_montage('standard_1020')
    pos_dict = montage.get_positions()['ch_pos']
    mne_lookup = {k.upper(): k for k in pos_dict}

    coords = []
    for ch in channel_names:
        mne_key = mne_lookup[ch.upper()]
        coords.append(torch.from_numpy(pos_dict[mne_key]).float())

    return torch.stack(coords, dim=0)  # (C, 3)


class Spatial3DEmbedding(nn.Module):
    """3D Fourier-feature electrode position embedding from MNE montage.

    Encodes 3D electrode coordinates (standard_1020, MRI frame) into
    dense embedding vectors via multi-scale Fourier features + MLP.
    The Fourier features capture spatial relationships (distance,
    neighbourhood) at multiple frequency scales, giving the model a
    geometry-aware channel prior beyond pure learned embeddings.

    Args:
        embed_dim: output embedding dimension (typically 512).
        num_freq_bands: Fourier frequency bands (default 12).
        max_freq: maximum frequency multiplier (default 10.0).
        channel_names: channel list or None for 58 defaults.
        freeze_coords: if True, coords are fixed buffers.
    """

    def __init__(self, embed_dim, num_freq_bands=12, max_freq=10.0,
                 channel_names=None, freeze_coords=True):
        super().__init__()
        self.embed_dim = embed_dim

        coords = get_3d_coordinates(channel_names)  # (C, 3)
        # Normalize to unit sphere
        coords = coords / coords.norm(dim=-1, keepdim=True).max()

        if freeze_coords:
            self.register_buffer('coords', coords)
        else:
            self.coords = nn.Parameter(coords)

        self.num_freq_bands = num_freq_bands
        freq_bands = torch.linspace(1.0, max_freq, num_freq_bands)
        self.register_buffer('freq_bands', freq_bands)

        # Fourier dim: raw(3) + 3 axes * 2(sin,cos) * num_freq_bands
        fourier_dim = 3 + 3 * 2 * num_freq_bands
        self.proj = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, chan_ids=None):
        """Returns spatial embedding for all or selected channels.

        Args:
            chan_ids: optional LongTensor of channel indices to select.
                      If None, returns (num_channels, embed_dim) for all channels.
        Returns:
            Tensor (C', embed_dim).
        """
        C = self.coords.shape[0]
        flat = self.coords  # (C, 3)

        # Build Fourier features vectorised: (C, 1, 3) * (1, B, 1) → (C, B, 3)
        scaled = flat.unsqueeze(1) * self.freq_bands.view(1, -1, 1)  # (C, B, 3)
        sin_feats = torch.sin(scaled).flatten(1)   # (C, B*3)
        cos_feats = torch.cos(scaled).flatten(1)   # (C, B*3)

        fourier_feats = torch.cat([flat, sin_feats, cos_feats], dim=-1)  # (C, fourier_dim)

        emb = self.proj(fourier_feats)  # (C, embed_dim)

        if chan_ids is not None:
            emb = emb[chan_ids.long()]
        return emb
