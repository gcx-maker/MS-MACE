import logging

import ase
import numpy as np
import torch
from e3nn.util.jit import compile_mode
from typing import Optional

from minimace.tools import scatter


@compile_mode("script")
class BesselBasis(torch.nn.Module):
    def __init__(self, r_max: float, num_basis=8, trainable=False):
        super().__init__()

        bessel_weights = (
            np.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype()
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor", torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype())
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        numerator = torch.sin(self.bessel_weights * x)
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )


@compile_mode("script")
class PolynomialCutoff(torch.nn.Module):
    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max: float, p=6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.calculate_envelope(x, self.r_max, self.p.to(torch.int))

    @staticmethod
    def calculate_envelope(
        x: torch.Tensor, r_max: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        r_over_r_max = x / r_max
        envelope = (
            1.0
            -((p+1.0) * (p+2.0)/2.0) * torch.pow(r_over_r_max, p)
            + p * (p+2.0) * torch.pow(r_over_r_max, p+1)
            - (p * (p+1.0)/2) * torch.pow(r_over_r_max, p+2)
        )
        return envelope * (x < r_max)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, r_max={self.r_max}"


@compile_mode("script")
class SwitchingFunction(torch.nn.Module):

    r_short: torch.Tensor
    width: torch.Tensor

    def __init__(
        self,
        r_short: float,
        width: float,
    ):
        super().__init__()

        self.register_buffer(
            "r_short",
            torch.tensor(
                r_short,
                dtype=torch.get_default_dtype()
            )
        )

        self.register_buffer(
            "width",
            torch.tensor(
                width,
                dtype=torch.get_default_dtype()
            )
        )

    def forward(
        self,
        r: torch.Tensor
    ) -> torch.Tensor:

        x = (
            r
            - (self.r_short - self.width)
        ) / (2.0 * self.width)

        x = torch.clamp(
            x,
            0.0,
            1.0
        )

        return (
            1.0
            - 6.0 * x**5
            + 15.0 * x**4
            - 10.0 * x**3
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(center={self.r_short}, width={self.width})"
        )


class RadialMLP(torch.nn.Module):
    def __init__(self, channels_list) -> None:
        super().__init__()

        modules = []
        in_channels = channels_list[0]

        for idx, out_channels in enumerate(channels_list[1:], start=1):
            modules.append(torch.nn.Linear(in_channels, out_channels, bias=True))
            in_channels = out_channels
            if idx < len(channels_list) - 1:
                modules.append(torch.nn.LayerNorm(out_channels))
                modules.append(torch.nn.SiLU())

        self.net = torch.nn.Sequential(*modules)
        self.hs = channels_list

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)