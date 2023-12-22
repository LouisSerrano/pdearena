# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from .activations import ACTIVATION_REGISTRY
from .fourier import SpectralConv1d


class BasicBlock1D(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        activation: str = "relu",
        norm: bool = True,
        num_groups: int = 8,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.bn1 = nn.GroupNorm(num_groups, planes) if norm else nn.Identity()
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.GroupNorm(num_groups, planes)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, self.expansion * planes) if norm else nn.Identity(),
            )

        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.activation(self.bn1(x)))
        out = self.conv2(self.activation(self.bn2(out)))
        out = out + self.shortcut(x)
        return out

class DilatedBasicBlock1D(nn.Module):
    """Basic block for Dilated ResNet with 1D convolutions

    Args:
        in_planes (int): number of input channels
        planes (int): number of output channels
        stride (int, optional): stride of the convolution. Defaults to 1.
        activation (str, optional): activation function. Defaults to "relu".
        norm (bool, optional): whether to use group normalization. Defaults to True.
        num_groups (int, optional): number of groups for group normalization. Defaults to 1.
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        activation: str = "relu",
        norm: bool = True,
        num_groups: int = 8,
    ):
        super().__init__()

        self.dilation = [1, 2, 4, 8, 4, 2, 1]
        dilation_layers = []
        for dil in self.dilation:
            dilation_layers.append(
                nn.Conv1d(
                    in_planes,
                    planes,
                    kernel_size=3,
                    stride=stride,
                    dilation=dil,
                    padding=dil,
                    bias=True,
                )
            )
        self.dilation_layers = nn.ModuleList(dilation_layers)
        self.norm_layers = nn.ModuleList(
            nn.GroupNorm(num_groups, planes) if norm else nn.Identity() for _ in self.dilation
        )
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer, norm in zip(self.dilation_layers, self.norm_layers):
            out = self.activation(layer(norm(out)))
        return out + x


class FourierBasicBlock1D(nn.Module):
    """Basic block for Fourier Neural Operators with 1D convolutions

    Args:
        in_planes (int): number of input channels
        planes (int): number of output channels
        stride (int, optional): stride of the convolution. Defaults to 1.
        modes (int, optional): number of modes for the spatial dimension. Defaults to 16.
        activation (str, optional): activation function. Defaults to "gelu".
        norm (bool, optional): whether to use group normalization. Defaults to False.

    """

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        modes: int = 16,
        activation: str = "gelu",
        norm: bool = False,
    ):
        super().__init__()
        self.modes = modes
        assert not norm
        self.fourier1 = SpectralConv1d(in_planes, planes, modes=self.modes)
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, padding=0, bias=True)
        self.fourier2 = SpectralConv1d(planes, planes, modes=self.modes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=True)

        # Adjust shortcut connections if necessary
        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion * planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1)
        #     )

        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.fourier1(x)
        x2 = self.conv1(x)
        out = self.activation(x1 + x2)

        x1 = self.fourier2(out)
        x2 = self.conv2(out)
        out = x1 + x2
        # out += self.shortcut(x)
        out = self.activation(out)
        return out
    

BLOCK_REGISTRY = {'basic': BasicBlock1D,
                  "fourier": FourierBasicBlock1D,
                  "dilated": DilatedBasicBlock1D}

class ResNet1D(nn.Module):
    """Class to support ResNet-like feedforward architectures for 1D convolutions

    Args:
        ... (same as before)
    """

    padding = 9

    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        block_name: str,
        num_blocks: list,
        time_history: int,
        time_future: int,
        hidden_channels: int = 64,
        activation: str = "gelu",
        norm: bool = True,
        diffmode: bool = False,
        usegrid: bool = False,
        **kwargs
    ):
        super().__init__()
        # (same initializations as before)

        block = BLOCK_REGISTRY[block_name]

        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.diffmode = diffmode
        self.usegrid = usegrid
        self.in_planes = hidden_channels

        insize = time_history * (self.n_input_scalar_components + self.n_input_vector_components)
        if self.usegrid:
            insize += 1  # Adjust for 1D
        self.conv_in1 = nn.Conv1d(insize, self.in_planes, kernel_size=1, bias=True)
        self.conv_in2 = nn.Conv1d(self.in_planes, self.in_planes, kernel_size=1, bias=True)
        self.conv_out1 = nn.Conv1d(self.in_planes, self.in_planes, kernel_size=1, bias=True)
        self.conv_out2 = nn.Conv1d(
            self.in_planes,
            time_future * (self.n_output_scalar_components + self.n_output_vector_components),
            kernel_size=1,
            bias=True,
        )

        self.layers = nn.ModuleList(
            [
                self._make_layer(
                    block,
                    self.in_planes,
                    num_blocks[i],
                    stride=1,
                    activation=activation,
                    norm=norm,
                )
                for i in range(len(num_blocks))
            ]
        )
        self.activation: nn.Module = ACTIVATION_REGISTRY.get(activation, None)
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

    def _make_layer(
        self,
        block: Callable,
        planes: int,
        num_blocks: int,
        stride: int,
        activation: str,
        norm: bool = True,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    activation=activation,
                    norm=norm,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4  # Adjust for 1D
        orig_shape = x.shape # original shape of (b c h t)
        #x = x.reshape(x.size(0), -1, x.shape[2])  # Adjust for 1D shape (b (t c) h)
        x = rearrange(x, 'b c h t -> b (c t) h')

        x = self.activation(self.conv_in1(x.float()))
        x = self.activation(self.conv_in2(x.float()))

        if self.padding > 0:
            x = F.pad(x, [0, self.padding])

        for layer in self.layers:
            x = layer(x)

        if self.padding > 0:
            x = x[..., : -self.padding]

        x = self.activation(self.conv_out1(x))
        x = self.conv_out2(x)

        if self.diffmode:
            raise NotImplementedError("diffmode")
        #return x.reshape(orig_shape[0], -1, self.n_output_scalar_components + self.n_output_vector_components, orig_shape[2])
        return rearrange(x, 'b (c t) h -> b c h t', c=self.n_output_scalar_components + self.n_output_vector_components)

    def __repr__(self):
        return "ResNet1D"



#######################################################################
#######################################################################
