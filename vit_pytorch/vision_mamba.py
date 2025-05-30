"""VisionMambaBlock module."""

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from zeta.nn import SSM


# Pair
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def output_head(dim: int, num_classes: int):
    """
    Creates a head for the output layer of a model.

    Args:
        dim (int): The input dimension of the head.
        num_classes (int): The number of output classes.

    Returns:
        nn.Sequential: The output head module.
    """
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, num_classes),
    )


class VisionEncoderMambaBlock(nn.Module):
    """
    VisionMambaBlock is a module that implements the Mamba block from the paper
    Vision Mamba: Efficient Visual Representation Learning with Bidirectional
    State Space Model

    Args:
        dim (int): The input dimension of the input tensor.
        heads (int): The number of heads in the multi-head attention mechanism.
        dt_rank (int): The rank of the state space model.
        dim_inner (int): The dimension of the inner layer of the
            multi-head attention.
        d_state (int): The dimension of the state space model.


    Example:
    >>> block = VisionMambaBlock(dim=256, heads=8, dt_rank=32,
            dim_inner=512, d_state=256)
    >>> x = torch.randn(1, 32, 256)
    >>> out = block(x)
    >>> out.shape
    torch.Size([1, 32, 256])
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.forward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.backward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.SiLU()
        self.ssm = SSM(dim, dt_rank, dim_inner, d_state)

        # Linear layer for z and x
        self.proj = nn.Linear(dim, dim)

        # Softplus
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        # x is of shape [batch_size, seq_len, dim]
        b, s, d = x.shape
        # print(x.shape)

        # Skip connection
        skip = x

        # Normalization
        x = self.norm(x)

        # Split x into x1 and x2 with linears
        z1 = self.proj(x)
        x1 = self.proj(x)

        # forward con1d
        x1_rearranged = rearrange(x1, "b s d -> b d s")
        forward_conv_output = self.forward_conv1d(x1_rearranged)
        forward_conv_output = rearrange(
            forward_conv_output, "b d s -> b s d"
        )
        x1_ssm = self.ssm(forward_conv_output)
        # x1_ssm = rearrange(x1_ssm, "b d s -> b s d")

        # backward conv x2
        x2_rearranged = rearrange(x1, "b s d -> b d s")
        x2 = self.backward_conv1d(x2_rearranged)
        x2 = rearrange(x2, "b d s -> b s d")

        # Backward ssm
        x2 = self.ssm(x2)

        # Activation
        z = self.activation(z1)
        # x2 = rearrange(x2, "b d s -> b s d")
        # print('test', x1_ssm.shape, x2.shape, z.shape)
        # matmul with z + backward ssm
        # x2 = x2 @ z
        x2 = x2 * z

        # Matmul with z and x1
        # x1 = x1_ssm @ z
        x1 = x1 * z
        # Add both matmuls
        x = x1 + x2

        # Add skip connection
        # print('test', x1.shape, skip.shape)
        return x + skip


class Vim(nn.Module):
    """
    Vision Mamba (Vim) model implementation.

    Args:
        dim (int): Dimension of the model.
        heads (int, optional): Number of attention heads. Defaults to 8.
        dt_rank (int, optional): Rank of the dynamic tensor. Defaults to 32.
        dim_inner (int, optional): Inner dimension of the model. Defaults to None.
        d_state (int, optional): State dimension of the model. Defaults to None.
        num_classes (int, optional): Number of output classes. Defaults to None.
        image_size (int, optional): Size of the input image. Defaults to 224.
        patch_size (int, optional): Size of the image patch. Defaults to 16.
        channels (int, optional): Number of image channels. Defaults to 3.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        depth (int, optional): Number of encoder layers. Defaults to 12.

    Attributes:
        dim (int): Dimension of the model.
        heads (int): Number of attention heads.
        dt_rank (int): Rank of the dynamic tensor.
        dim_inner (int): Inner dimension of the model.
        d_state (int): State dimension of the model.
        num_classes (int): Number of output classes.
        image_size (int): Size of the input image.
        patch_size (int): Size of the image patch.
        channels (int): Number of image channels.
        dropout (float): Dropout rate.
        depth (int): Number of encoder layers.
        to_patch_embedding (nn.Sequential): Sequential module for patch embedding.
        dropout (nn.Dropout): Dropout module.
        cls_token (nn.Parameter): Class token parameter.
        to_latent (nn.Identity): Identity module for latent representation.
        layers (nn.ModuleList): List of encoder layers.
        output_head (output_head): Output head module.

    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dt_rank: int = 32,
        dim_inner: int = None,
        d_state: int = None,
        num_classes: int = None,
        image_size: int = 224,
        patch_size: int = 16,
        channels: int = 3,
        dropout: float = 0.1,
        depth: int = 12,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.dropout = dropout
        self.depth = depth

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        patch_dim = channels * patch_height * patch_width
        window = image_size//patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_height,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.to_patch = nn.Sequential(
            nn.Linear(window*window, image_size*image_size),
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Latent
        self.to_latent = nn.Identity()

        # encoder layers
        self.layers = nn.ModuleList()

        # Append the encoder layers
        for _ in range(depth):
            self.layers.append(
                VisionEncoderMambaBlock(
                    dim=dim,
                    heads=heads,
                    dt_rank=dt_rank,
                    dim_inner=dim_inner,
                    d_state=d_state,
                    *args,
                    **kwargs,
                )
            )

        # Output head
        self.output_head = output_head(dim, num_classes)

    def forward(self, x: Tensor):
        # Patch embedding
        b, c, h, w = x.shape
        # print(x.shape)

        x = self.to_patch_embedding(x)
        # print(f"Patch embedding: {x.shape}")

        # Shape
        b, n, _ = x.shape

        # Cls tokens
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        # print(f"Cls tokens: {cls_tokens.shape}")

        # Concatenate
        x = torch.cat((cls_tokens, x), dim=1)

        # Dropout
        x = self.dropout(x)
        # print(x.shape)

        # Forward pass with the layers
        for layer in self.layers:
            x = layer(x)
            # print(f"Layer: {x.shape}")

        # Latent
        # x = self.to_latent(x)
        # x = self.output_head(x)
        x = x[:, 1:n+1, :].permute(0, 2, 1)
        b, n, _ = x.shape
        # x = x.reshape(b, n, h, w)
        x = self.to_patch(x).reshape(b, n, h, w)
        # Output head with the cls tokens
        return x


