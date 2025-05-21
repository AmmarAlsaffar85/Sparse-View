import torch
import torch.nn as nn
import torch.autograd
import warnings


class ConvLayer(nn.Module):
    """
    ODConv Layer. Implementation inspired by ResNet: https://arxiv.org/pdf/1512.03385.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x) -> torch.Tensor:
        output = self.conv(x)
        output = self.norm(output)
        output = self.activation(output)
        return output

    def update_temperature(self, temperature: int):
        self.odconv.attention.update_temperature(temperature)


class ResidualConvBlock(nn.Module):
    """
    Residual Conv Block. Implementation follows ResNet: https://arxiv.org/pdf/1512.03385.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 layers_per_block: int = 2):
        super(ResidualConvBlock, self).__init__()
        self.layers = nn.ModuleList([ConvLayer(in_channels, out_channels, kernel_size)
                                     for _ in range(layers_per_block)])

    def forward(self, x) -> torch.Tensor:
        residual = x
        for layer in self.layers:
            x = layer(x)
        return x + residual

    def update_temperature(self, temperature: int):
        for layer in self.layers:
            layer.update_temperature(temperature)


class ResConvSR(nn.Module):
    def __init__(self, residual_blocks: int = 1, layers_per_block: int = 2, embed_dim: int = 32,
                 kernel_size: int = 3, reorder_channels: bool = False):
        super(ResConvSR, self).__init__()

        # Whether to use PixelShuffle and PixelUnshuffle.
        self.reorder_channels = reorder_channels

        if self.reorder_channels:
            self.pixel_unshuffle = nn.PixelUnshuffle(2)
            self.initial_conv = nn.Conv2d(4, embed_dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        else:
            self.initial_conv = nn.Conv2d(1, embed_dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        self.blocks = nn.ModuleList([ResidualConvBlock(embed_dim, embed_dim, kernel_size=kernel_size,
                                                       layers_per_block=layers_per_block)
                                     for _ in range(residual_blocks)])

        # Final layer has to reduce channel dimension and ensure that output is not constrained by norm and activation.
        if self.reorder_channels:
            self.final_conv = nn.Conv2d(embed_dim, 4 * 1 ** 2, kernel_size=3, stride=1,
                                        padding=kernel_size // 2)
            self.pixel_shuffle = nn.PixelShuffle(2)
        else:
            self.final_conv = nn.Conv2d(embed_dim, 1 ** 2, kernel_size=3, stride=1, padding=kernel_size // 2)
        self.upsampler = torch.nn.PixelShuffle(upscale_factor=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x
        if self.reorder_channels:
            output = self.pixel_unshuffle(x)

        # Pass through internal blocks.
        output = self.initial_conv(output)
        for block in self.blocks:
            output = block(output)
        output = self.final_conv(output)

        if self.reorder_channels:
            output = self.pixel_shuffle(output)

        # Global skip connection. Repeat input along channel dimension.
        residual = torch.cat(tensors=tuple(x for _ in range(output.shape[-3])), dim=-3)
        output = torch.add(output, residual)

        output = self.upsampler(output)
        return output
