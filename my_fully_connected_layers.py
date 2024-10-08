#asun: add dropout
#date: 01192024
import logging
from typing import Callable
from typing import Optional
from typing import Union

import torch.nn as nn
from torch import Tensor

from modulus.models.layers.weight_norm import WeightNormLinear
from modulus.models.layers.activation import Activation, get_activation_fn

logger = logging.getLogger(__name__)


class FCLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
        weight_norm: bool = False,
        activation_par: Optional[nn.Parameter] = None,
        dropout: float = None,

    ) -> None:
        super().__init__()
        self.activation_fn = activation_fn
        self.callable_activation_fn = get_activation_fn(
            activation_fn, out_features=out_features
        )
        self.weight_norm = weight_norm
        self.activation_par = activation_par

        #asun 01192024
        if not dropout is None:
            self.dropout =  nn.Dropout(dropout)
        else:
            self.dropout = None
        if weight_norm:
            self.linear = WeightNormLinear(in_features, out_features, bias=True)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=True)
        self.reset_parameters()

    def exec_activation_fn(self, x: Tensor) -> Tensor:
        return self.callable_activation_fn(x)

    def reset_parameters(self) -> None:
        nn.init.constant_(self.linear.bias, 0)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.weight_norm:
            nn.init.constant_(self.linear.weight_g, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        #asun 01192024
        if not self.dropout is None:
            x = self.dropout(x)

        x = self.linear(x)
        if self.activation_fn is not Activation.IDENTITY:
            if self.activation_par is None:
                x = self.exec_activation_fn(x)
            else:
                x = self.exec_activation_fn(self.activation_par * x)
        return x


# FC like layer for image channels
class ConvFCLayer(nn.Module):
    def __init__(
        self,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
        activation_par: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__()
        self.activation_fn = activation_fn
        self.callable_activation_fn = get_activation_fn(activation_fn)
        self.activation_par = activation_par

    def exec_activation_fn(self, x: Tensor) -> Tensor:
        return self.callable_activation_fn(x)

    def apply_activation(self, x: Tensor) -> Tensor:
        if self.activation_fn is not Activation.IDENTITY:
            if self.activation_par is None:
                x = self.exec_activation_fn(x)
            else:
                x = self.exec_activation_fn(self.activation_par * x)
        return x


class Conv1dFCLayer(ConvFCLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
        weight_norm: bool = False,
        activation_par: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_features
        self.out_channels = out_features
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1, bias=True)
        self.reset_parameters()

        if weight_norm:
            logger.warn("Weight norm not supported for Conv FC layers")

    def reset_parameters(self) -> None:
        nn.init.constant_(self.conv.bias, 0)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class Conv2dFCLayer(ConvFCLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
        activation_par: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.conv.bias, 0)
        self.conv.bias.requires_grad = False
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x


class Conv3dFCLayer(ConvFCLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
        activation_par: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__(activation_fn, activation_par)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.conv.bias, 0)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.apply_activation(x)
        return x
