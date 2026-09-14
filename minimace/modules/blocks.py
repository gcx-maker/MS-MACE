from abc import abstractmethod, ABC
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch.nn.functional
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

from minimace.tools.compile import simplify_if_compile
from minimace.tools.scatter import scatter_sum
from .radial import (
    BesselBasis,
    PolynomialCutoff,
    RadialMLP
)
from .symmetric_contraction import SymmetricContraction
from .irreps_tools import tp_out_irreps_with_instructions, reshape_irreps


@compile_mode("script")
class LinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(
            self,
            irreps_in: o3.Irreps,
            irreps_out: o3.Irreps
    ):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(
            self,
            node_attrs: torch.Tensor,
    ) -> torch.Tensor:  #[n_nodes, irreps]
        return self.linear(node_attrs)


@compile_mode("script")
class LinearReadoutBlock(torch.nn.Module):
    def __init__(
            self,
            irreps_in: o3.Irreps,
            irrep_out: o3.Irreps = o3.Irreps("0e")
    ):
        super().__init__()
        self.linear = o3.Linear(
            irreps_in=irreps_in, irreps_out=irrep_out
        )

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        return self.linear(x)


@compile_mode("script")
class NonLinearReadoutBlock(torch.nn.Module):
    def __init__(
            self,
            irreps_in: o3.Irreps,
            MLP_irreps: o3.Irreps,
            gate: Optional[callable],
            irrep_out: o3.Irreps = o3.Irreps("0e"),
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.linear_1 = o3.Linear(
            irreps_in=irreps_in, irreps_out=self.hidden_irreps
        )
        self.non_linearity = simplify_if_compile(nn.Activation)(
            irreps_in=self.hidden_irreps, acts=[gate]
        )
        self.linear_2 = o3.Linear(
            irreps_in=self.hidden_irreps, irreps_out=irrep_out
        )

    def forward(
            self, x:torch.Tensor,
    ):
        x = self.non_linearity(self.linear_1(x))
        return self.linear_2(x)


@compile_mode("script")
class AtomicEnergiesBlock(torch.nn.Module):
    atomic_energies: torch.Tensor

    def __init__(self, atomic_energies: Union[np.ndarray, torch.Tensor]):
        super().__init__()

        self.register_buffer(
            "atomic_energies",
            torch.tensor(atomic_energies, dtype=torch.get_default_dtype())
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        energies = torch.atleast_2d(self.atomic_energies).T.to(
            dtype=x.dtype, device=x.device
        )
        return torch.matmul(x, energies)

    def __repr__(self):
        formatted_energies = ", ".join(
            [
                "[" + ", ".join([f"{x:.4f}" for x in group]) + "]"
                for group in torch.atleast_2d(self.atomic_energies)
            ]
        )
        return f"{self.__class__.__name__}(energies=[{formatted_energies}])"


@compile_mode("script")
class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(
            self,
            r_max: float,
            num_bessel: int,
            num_polynomial_cutoff: int,
            radial_type: str = "bessel",
            distance_transform: str = "None",
            apply_cutoff: bool = True
    ):
        super().__init__()
        if radial_type == "bessel":
            self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel
        self.apply_cutoff = apply_cutoff

    def forward(
            self,
            edge_lengths: torch.Tensor, #[n_edges, 1]
            node_attrs: torch.Tensor,
            edge_index: torch.Tensor,
            atomic_numbers: torch.Tensor
    ):
        cutoff = self.cutoff_fn(edge_lengths)
        radial = self.bessel_fn(edge_lengths)
        if hasattr(self, "apply_cutoff"):
            if not self.apply_cutoff:
                return radial, cutoff
        return radial*cutoff, None


class PolynomialCutoffBlock(torch.nn.Module):
    def __init__(self, r_max, p):
        super().__init__()
        self.cutoff_fn = PolynomialCutoff(
            r_max=r_max,
            p=p
        )

    def forward(self, edge_lengths):
        return self.cutoff_fn(edge_lengths)


@compile_mode("script")
class EquivariantProductBasisBlock(torch.nn.Module):
    def __init__(
            self,
            node_feats_irreps: o3.Irreps,
            target_irreps: o3.Irreps,
            correlation: int,
            use_sc: bool = True,
            num_elements: Optional[int] = None,
            use_agnostic_product: bool = False,
            use_reduced_cg: Optional[bool] = None
    ) -> None:
        super().__init__()

        self.use_sc = use_sc
        self.use_agnostic_product = use_agnostic_product

        if self.use_agnostic_product:
            num_elements = 1
        self.symmetric_contractions = SymmetricContraction(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_reduced_cg=use_reduced_cg
        )
        self.linear = o3.Linear(
            target_irreps,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(
            self,
            node_feats: torch.Tensor,
            sc: Optional[torch.Tensor],
            node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_agnostic_product:
            node_attrs = torch.ones(
                (node_feats.shape[0], 1),
                dtype=node_feats.dtype,
                device=node_feats.device
            )
        else:
            node_feats = self.symmetric_contractions(node_feats, node_attrs)

        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc
        return self.linear(node_feats)


@compile_mode("script")
class InteractionBlock(torch.nn.Module, ABC):
    def __init__(
            self,
            node_attrs_irreps: o3.Irreps,
            node_feats_irreps: o3.Irreps,
            edge_attrs_irreps: o3.Irreps,
            edge_feats_irreps: o3.Irreps,
            target_irreps: o3.Irreps,
            hidden_irreps: o3.Irreps,
            avg_num_neighbors: float,
            edge_irreps: Optional[o3.Irreps] = None,
            radial_MLP: Optional[List[int]] = None
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        if edge_irreps is None:
            edge_irreps = self.node_feats_irreps
        self.radial_MLP = radial_MLP
        self.edge_irreps = edge_irreps
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
            self,
            node_attrs: torch.Tensor,
            node_feats: torch.Tensor,
            edge_attrs: torch.Tensor,
            edge_feats: torch.Tensor,
            edge_index: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


nonlinearities = {1: torch.nn.functional.silu, -1: torch.tanh}


@compile_mode("script")
class RealAgnosticInteractionBlock(InteractionBlock):
    def _setup(self) -> None:

        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
        )

        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.edge_irreps,
            self.edge_attrs_irreps,
            self.target_irreps
        )

        self.conv_tp = o3.TensorProduct(
            self.edge_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False
        )

        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu
        )

        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True
        )

        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out
        )
        self.reshape = reshape_irreps(self.irreps_out)

    def forward(
            self,
            node_attrs: torch.Tensor,
            node_feats: torch.Tensor,
            edge_attrs: torch.Tensor,
            edge_feats: torch.Tensor,
            edge_index: torch.Tensor,
            cutoff: Optional[torch.Tensor] = None,
            first_layer: bool = False
    ) -> Tuple[torch.Tensor, None]:
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff

        message = None
        mji = self.conv_tp(
            node_feats[edge_index[0]], edge_attrs, tp_weights
        )
        message = scatter_sum(
            src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0]
        )
        message = self.linear(message) / self.avg_num_neighbors
        message = self.skip_tp(message, node_attrs)
        return (
            self.reshape(message),
            None
        )


@compile_mode("script")
class RealAgnosticResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True
        )

        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.edge_irreps,
            self.edge_attrs_irreps,
            self.target_irreps
        )

        self.conv_tp = o3.TensorProduct(
            self.edge_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu
        )

        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True
        )

        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
        )
        self.reshape = reshape_irreps(self.irreps_out)

    def forward(
            self,
            node_attrs: torch.Tensor,
            node_feats: torch.Tensor,
            edge_attrs: torch.Tensor,
            edge_feats: torch.Tensor,
            edge_index: torch.Tensor,
            cutoff: Optional[torch.Tensor] = None,
            first_layer: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff
        message = None
        mji = self.conv_tp(node_feats[edge_index[0]], edge_attrs, tp_weights)
        message = scatter_sum(src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0])
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )
