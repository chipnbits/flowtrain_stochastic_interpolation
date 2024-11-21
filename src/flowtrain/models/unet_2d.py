""" 
Unet model adapted from  https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main
With some changes and annotation made by Simon Ghyselincks 2024
See unetdiagram for a visual guide to the model structure

MIT License
Copyright (c) 2020 Phil Wang

References:
[1] Vaswani et. al, (2017). Attention is All you Need. Advances in Neural Information Processing Systems, 30. https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

"""

import math
from collections import namedtuple
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from packaging import version
from torch import einsum, nn
from torch.nn import Module, ModuleList


# helpers functions
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(t, length=1):
    if isinstance(t, tuple):
        return t
    return (t,) * length


def divisible_by(numer, denom):
    return (numer % denom) == 0


# small helper modules
def Upsample(dim, dim_out=None):
    """
    Upsample by a factor of 2 with a convolutional layer.

    Input: N x dim x H x W
    Output: N x dim_out x 2H x 2W
    """
    return nn.Sequential(
        # Upsample to 2H x 2W using nearest neighbor
        nn.Upsample(scale_factor=2, mode="nearest"),
        # 3x3 kernel with padding to preserve 2Hx2W dimensions
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    """
    Downsample by a factor of 2 with a convolutional layer.

    Input: N x dim x H x W
    Output: N x dim_out x H/2 x W/2
    """
    return nn.Sequential(
        # Rearrange the input to N x (dim*4) x (H/2) x (W/2) on a 2x2 patch basis
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        # Take a 1x1 conv to change number of channels using weighted sum pixel by pixel
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class RMSNorm(Module):
    """
    Custom module for RMS normalization across C dimension.

    Normalize across C for vector of length 1
    Rescale weightings across C by a learnable parameter g and 1/sqrt(dim(C))

    Input: N x C x H x W
    Output: N x C x H x W
    """

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale


class SinusoidalPosEmb(Module):
    """
    Non-learnable sinusoidal positional embeddings.

    * Modified from Lucidrain's implementation to have alternating indices for sin/cos embeddings.
    Implemented like in the original transformer paper [1] section 3.5.

    Input: Nx1 (time)
    """

    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        # Adapt time [0,1] to [0,1000] range
        x = x * 1000
        device = x.device
        # Get vector theta^-(2i/d) for i = 0 to d/2
        half_dim = self.dim // 2
        emb = math.log(self.theta) / half_dim
        # the vector with the range i from 1 to d/2 (d/2 elements total)
        emb = torch.exp((torch.arange(half_dim, device=device) + 1) * -emb)
        # create t*theta^-(2i/d) for t in x and i in d/2 for full argument to sin/cos
        emb = x[:, None] * emb[None, :]
        # interleave the sin and cos embeddings
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = emb.view(emb.shape[0], -1)
        return emb


class RandomFourierEmbedding(Module):
    """
    Time/Positional Embedding with random Fourier features.

    Parameters:
    num_channels: Dimension of the embedding vector
    bandwidth: Bandwidth of the frequencies, use higher frequencies for narrow time windows

    Forward:
    t: time/pos embedding vector with length = batch_size (one scalar per batch element)

    Basis Functions:
    This basis uses frequencies 'f' and phases 'phi' that are drawn from N(0,1) and U(0,1) respectively
    """

    def __init__(self, num_channels, bandwidth=100.0):
        super().__init__()
        self.freqs = nn.Parameter(
            torch.randn(num_channels) * bandwidth, requires_grad=False
        )
        self.phases = nn.Parameter(torch.rand(num_channels), requires_grad=False)

    def forward(self, t: torch.Tensor):
        # Outer product, with phases added then cosine
        y = t.ger(self.freqs)
        y = y + self.phases
        y = y.cos() * math.sqrt(2)  # scale for unit-variance
        return y


class LearnedFourierEmbedding(RandomFourierEmbedding):
    """A variation with learned frequencies and phases."""

    def __init__(self, num_channels, bandwidth=100.0):
        super().__init__(num_channels, bandwidth)
        # Learnable frequencies and phases
        self.freqs = nn.Parameter(torch.randn(num_channels) * bandwidth)
        self.phases = nn.Parameter(torch.rand(num_channels))


class LearnedSinusoidalPosEmb(Module):
    """following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=True)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class Block(Module):
    """Basic block with a 3x3 convolution, RMSNorm, SiLU activation, and optional dropout."""

    def __init__(self, dim, dim_out, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            # Scaling and shifting the input for the activation function (based on embeddings)
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)


class ResnetBlock(Module):
    """Residual block with a 3x3 convolution, RMSNorm, SiLU activation, and optional dropout.

    Layers a time embedding MLP into a Block's activation, followed by a second Block.
    """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.0):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        # Calculate the sequential blocks with a time embedding on block 1
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        # Residual skip connection pass through a 1x1 convolution
        return h + self.res_conv(x)


""" Attention modules as described in the original transformer paper [1] section 3.2. """


class Attend(nn.Module):
    def __init__(self, dropout=0.0, flash=False, scale=None):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        # determine efficient attention configs for cuda and cpu
        AttentionConfig = namedtuple(
            "AttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
        )
        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        device_version = version.parse(
            f"{device_properties.major}.{device_properties.minor}"
        )

        if device_version > version.parse("8.0"):
            print("A100 GPU detected, using flash attention if input tensor is on cuda")
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print(
                "Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda"
            )
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = (
            *q.shape,
            k.shape[-2],
            q.is_cuda,
            q.device,
        )

        if exists(self.scale):
            default_scale = q.shape[-1]
            q = q * (self.scale / default_scale)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )

        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        if self.flash:
            return self.flash_attn(q, k, v)

        scale = default(self.scale, q.shape[-1] ** -0.5)

        # similarity
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out


class LinearAttention(Module):
    def __init__(
        self,
        dim,  # Number of input channels
        heads=4,  # Number of attention heads
        dim_head=32,  # Dimension per attention head
        num_mem_kv=4,  # Number of memory key-value pairs
    ):
        super().__init__()
        self.scale = dim_head**-0.5  # Scaling factor for queries
        self.heads = heads
        hidden_dim = dim_head * heads  # Total dimension for all heads

        self.norm = RMSNorm(dim)  # Normalization layer

        # Memory key-value pairs
        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        # Project input to queries, keys, and values
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        # Output projection and normalization
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape  # Batch size, channels, height, width

        x = self.norm(x)  # Normalize input

        # Project input to queries, keys, and values
        qkv = self.to_qkv(x).chunk(3, dim=1)  # Split into Q, K, V
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        # Repeat memory keys and values for each batch
        mk, mv = map(lambda t: repeat(t, "h c n -> b h c n", b=b), self.mem_kv)

        # Concatenate memory keys/values with projected keys/values
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))

        # Apply softmax to queries and keys
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale  # Scale queries

        # Compute context (weighted sum of values)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        # Compute output as weighted sum of context and queries
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)

        # Rearrange output back to original shape
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)  # Final output projection and normalization


class Attention(Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4, flash=False):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash=flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)
        # c -> is chunked into 3 parts for queries, keys, and values
        qkv = self.to_qkv(x).chunk(3, dim=1)
        # Convert shape c, x, y to shape h, x*y, c/h for queries, keys, and values chuncks
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads), qkv
        )
        # Duplicate memory key-value pairs for each batch
        mk, mv = map(lambda t: repeat(t, "h n d -> b h n d", b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class Unet2D(Module):
    def __init__(
        self,
        dim,  # Base number of channels in model
        dim_mults=(1, 2, 4, 8),
        data_channels=3,
        dropout=0.0,
        self_condition=False,
        time_resolution=64,
        time_sin_pos=False,
        time_bandwidth=100.0,
        time_learned_emb=False,
        attn_enabled=True,
        attn_dim_head=64,
        attn_heads=4,
        full_attn=None,  # defaults to full attention for final down/up and middle layers
        flash_attn=False,
    ):
        super().__init__()
        self.attn_enabled = attn_enabled
        # Determine dimensioning
        self.channels = data_channels
        self.self_condition = self_condition
        input_channels = data_channels * (2 if self_condition else 1)

        # Intial convolution with padding on data input, NxCxHxW -> NxCxHxW
        self.init_conv = nn.Conv2d(input_channels, dim, 7, padding=3)

        # Builds a list of tuples of input and output dimensions for each layer based on dim_mults
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Time embeddings via random Fourier features or learned Fourier features
        time_dim = dim * 4
        if time_sin_pos:
            time_embed = SinusoidalPosEmb(time_resolution, theta=10000)
        else:
            TimeEmbedding = (
                LearnedFourierEmbedding if time_learned_emb else RandomFourierEmbedding
            )
            time_embed = TimeEmbedding(time_resolution, bandwidth=time_bandwidth)

        self.time_mlp = nn.Sequential(
            time_embed,
            nn.Linear(time_resolution, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # attention, specify which layers will have full_attn and validate parameters
        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        # Broadcast attention parameters to all stages
        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)
        assert len(full_attn) == len(dim_mults)

        # prepare blocks, Partial used to create versions with parameters partially specified for convenience
        # Set flash attn for all full attention blocks
        FullAttention = partial(Attention, flash=flash_attn)
        # Set time embed and dropout across all resnet blocks
        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout)

        # Down section of the U-Net
        self.downs = ModuleList([])
        # Up section of the U-Net
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        # All down sections follow the same structure.
        for ind, (
            (dim_in, dim_out),
            layer_full_attn,
            layer_attn_heads,
            layer_attn_dim_head,
        ) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):

            # Check if this is the last layer before middle
            is_last = ind >= (num_resolutions - 1)

            if self.attn_enabled:  # Apply linear or full attenttention as specified
                attn_klass = FullAttention if layer_full_attn else LinearAttention
                attn_operation = attn_klass(
                    dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads
                )
            else:
                attn_operation = nn.Identity()

            """ Basic down structure:
            
            Two Resnet blocks with time embedding and pass through, followed by multiheaded self-attention layer and downsample.
            NxC_inxHxW -> NxC_outxH/2xW/2
            """
            self.downs.append(
                ModuleList(
                    [
                        resnet_block(dim_in, dim_in),
                        resnet_block(dim_in, dim_in),
                        attn_operation,
                        (
                            Downsample(dim_in, dim_out)
                            if not is_last
                            else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                        ),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        if attn_enabled:
            self.mid_attn = FullAttention(
                mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1]
            )
        else:
            self.mid_attn = nn.Identity()
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, (
            (dim_in, dim_out),
            layer_full_attn,
            layer_attn_heads,
            layer_attn_dim_head,
        ) in enumerate(
            zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))
        ):
            is_last = ind == (len(in_out) - 1)

            if self.attn_enabled:
                attn_klass = FullAttention if layer_full_attn else LinearAttention
                attn_operation = attn_klass(
                    dim_out, dim_head=layer_attn_dim_head, heads=layer_attn_heads
                )
            else:
                attn_operation = nn.Identity()

            self.ups.append(
                ModuleList(
                    [
                        resnet_block(dim_out + dim_in, dim_out),
                        resnet_block(dim_out + dim_in, dim_out),
                        attn_operation,
                        (
                            Upsample(dim_out, dim_in)
                            if not is_last
                            else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                        ),
                    ]
                )
            )

        self.out_dim = data_channels

        # Final skip concatenation and residual block
        self.final_res_block = resnet_block(dim * 2, dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond=None):
        assert all(
            [divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]
        ), f"your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet"

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)
        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            if self.attn_enabled:
                x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        if self.attn_enabled:
            x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            if self.attn_enabled:
                x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


if __name__ == "__main__":
    import time as clock

    from torchinfo import summary
    from tqdm import tqdm

    unet = Unet2D(
        dim=128,  # Base number of channels in model
        dim_mults=(1, 1, 2, 3, 4),
        data_channels=22,
        dropout=0.0,
        self_condition=False,
        time_resolution=128,
        time_bandwidth=100.0,
        time_learned_emb=False,
        attn_enabled=True,
        attn_dim_head=64,
        attn_heads=4,
        full_attn=None,  # defaults to full attention only for inner most layer
        flash_attn=False,
    )

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    unet.to(device)

    class TimeWrappedModel(nn.Module):
        def __init__(self, base_model, fixed_time):
            super().__init__()
            self.base_model = base_model
            self.fixed_time = fixed_time

        def forward(self, x):
            # Use the fixed time in every forward pass
            return self.base_model(x, self.fixed_time)

    # ------Testing and Summary------#
    test_shape = (16, 22, 128, 128)
    time_emb = torch.rand(test_shape[0]).to(device)  # Example time embedding
    x = torch.randn(*test_shape) * 0
    x[:, :10:20, 10:20, 10:20] = 1.0

    # Use time wrapper to get model(x) form for summary
    time_wrapped_model = TimeWrappedModel(unet, time_emb)
    summary(time_wrapped_model, input_size=test_shape, depth=4)

    # Forward pass
    print(0)
    # Forward pass
    num_iterations = 10
    total_time = 0.0

    # Example input tensor
    for _ in tqdm(range(num_iterations), desc="Measuring forward pass time"):
        start_time = clock.time()
        output = unet(x.to(device), time_emb.to(device))
        end_time = clock.time()
        total_time += end_time - start_time

    average_time = total_time / num_iterations
    print(f"Average forward pass time: {average_time:.5f} seconds")
    print(1)
    print(" ")
