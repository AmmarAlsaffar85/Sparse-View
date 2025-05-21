import torch
import torch.nn as nn
#from torch.nn import Module
import torch.nn.functional as F
import torch.autograd
import warnings


class Attention(torch.nn.Module):
    """
    Attention module created by Chen et al. Code found here: https://github.com/OSVAI/ODConv/blob/main/modules/odconv.py
    """
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature: int):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2d(nn.Module):
    """
    Omni-dimensional convolution. Code from here: https://github.com/OSVAI/ODConv/blob/main/modules/odconv.py
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)

    def update_temperature(self, temperature: int):
        self.attention.update_temperature(temperature)


class ODConvLayer(nn.Module):
    """
    ODConv Layer. Implementation inspired by ResNet: https://arxiv.org/pdf/1512.03385.
    """
    def __init__(self, in_channels: int, out_channels: int, convolution_kernels: int, kernel_size: int):
        super(ODConvLayer, self).__init__()
        self.odconv = ODConv2d(in_channels, out_channels, kernel_num=convolution_kernels, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x) -> torch.Tensor:
        output = self.odconv(x)
        output = self.norm(output)
        output = self.activation(output)
        return output

    def update_temperature(self, temperature: int):
        self.odconv.attention.update_temperature(temperature)


class ResidualODConvBlock(nn.Module):
    """
    Residual ODConv Block. Implementation inspired by ResNet: https://arxiv.org/pdf/1512.03385.
    """
    def __init__(self, in_channels: int, out_channels: int, convolution_kernels: int, kernel_size: int,
                 layers_per_block: int = 2):
        super(ResidualODConvBlock, self).__init__()
        self.layers = nn.ModuleList([ODConvLayer(in_channels, out_channels, convolution_kernels, kernel_size)
                                     for _ in range(layers_per_block)])

    def forward(self, x) -> torch.Tensor:
        residual = x
        for layer in self.layers:
            x = layer(x)
        return x + residual

    def update_temperature(self, temperature: int):
        for layer in self.layers:
            layer.update_temperature(temperature)


class ODConvSR(nn.Module):
    def __init__(self, odconv_blocks: int = 1, layers_per_block: int = 2, embed_dim: int = 32,
                 convolution_kernels: int = 4, kernel_size: int = 3, reorder_channels: bool = False):
        super(ODConvSR, self).__init__()

        # Whether to use PixelShuffle and PixelUnshuffle. Scales inefficiently.
        self.reorder_channels = reorder_channels

        if self.reorder_channels:
            self.pixel_unshuffle = nn.PixelUnshuffle(2)
            self.initial_conv = nn.Conv2d(4, embed_dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        else:
            self.initial_conv = nn.Conv2d(1, embed_dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        self.blocks = nn.ModuleList([ResidualODConvBlock(embed_dim, embed_dim, convolution_kernels=convolution_kernels,
                                                         kernel_size=kernel_size, layers_per_block=layers_per_block)
                                     for _ in range(odconv_blocks)])

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

    def update_temperature(self, temperature: int):
        """
        Update temperature used for softmax used in attention weight computation. Temperature annealing can improve
        model performance by ensuring that all kernels are trained in early epochs (high temperature), then specialized
        more in later epochs (low temperature).

        Args:
            temperature: updated temperature.
        """
        if self.training:
            for block in self.blocks:
                block.update_temperature(temperature=temperature)
        else:
            warnings.warn('Can only decay temperature during training!')
