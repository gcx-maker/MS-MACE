from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from minimace.tools.scatter import scatter_sum
from .radial import SwitchingFunction
from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearReadoutBlock,
    LinearNodeEmbeddingBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    PolynomialCutoffBlock
)

from .utils import prepare_graph, get_outputs, prepare_ms_graph


@compile_mode("script")
class MACE(torch.nn.Module):
    def __init__(
            self,
            r_max: float,
            num_bessel: int,
            num_polynomial_cutoff: int,
            max_ell : int,
            interaction_cls: Type[InteractionBlock],
            interaction_cls_first: Type[InteractionBlock],
            num_interactions: int,
            num_elements: int,
            hidden_irreps: o3.Irreps,
            MLP_irreps: o3.Irreps,
            atomic_energies: np.ndarray,
            avg_num_neighbors: float,
            atomic_numbers: List[int],
            correlation: Union[int, List[int]],
            gate: Optional[Callable],
            apply_cutoff: bool = True,
            use_reduced_cg: bool = True,
            use_so3: bool = True,
            use_agnostic_product: bool = False,
            use_last_readout_only: bool = False,
            distance_transform: str = "None",
            edge_irreps: Optional[o3.Irreps] = None,
            use_edge_irreps_first: bool = False,
            radial_MLP: Optional[List[int]] = None,
            radial_type: Optional[str] = "bessel",
            readout_cls: Optional[Type[NonLinearReadoutBlock]] = NonLinearReadoutBlock,
            keep_last_layer_irreps: bool = False
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64),
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        self.apply_cutoff = apply_cutoff
        self.edge_irreps = edge_irreps
        self.use_reduced_cg = use_reduced_cg
        self.use_agnostic_product = use_agnostic_product
        self.use_so3 = use_so3
        self.use_last_readout_only = use_last_readout_only
        self.use_edge_irreps_first = use_edge_irreps_first

        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
            apply_cutoff=apply_cutoff
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        if not use_so3:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        else:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p=1)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))

        def generate_irreps(l):
            str_irrep = "+".join([f"1x{i}e+1x{i}o" for i in range(l+1)])
            return o3.Irreps(str_irrep)

        sh_irreps_inter = sh_irreps
        if hidden_irreps.count(o3.Irrep(0, -1)) > 0:
            sh_irreps_inter = generate_irreps(max_ell)
        interaction_irreps = (sh_irreps_inter * num_features).sort()[0].simplify()
        interaction_irreps_first = (sh_irreps * num_features).sort()[0].simplify()

        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)
        if num_interactions == 1:
            hidden_irreps_out = str(hidden_irreps[0])
        else:
            hidden_irreps_out = hidden_irreps
        edge_irreps_first = None
        if use_edge_irreps_first and edge_irreps is not None:
            edge_irreps_first = o3.Irreps(f"{edge_irreps.count(o3.Irrep(0, 1))}x0e")
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps_first,
            hidden_irreps=hidden_irreps_out,
            edge_irreps=edge_irreps_first,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP
        )
        self.interactions = torch.nn.ModuleList([inter])

        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps_out,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()

        if not use_last_readout_only:
            self.readouts.append(
                LinearReadoutBlock(
                    hidden_irreps_out,
                    o3.Irreps("1x0e"),
                )
            )

        for i in range(num_interactions - 1):
            if i == num_interactions -2 and not keep_last_layer_irreps:
                hidden_irreps_out = str(hidden_irreps[0])
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                edge_irreps=edge_irreps,
                radial_MLP=radial_MLP
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i+1],
                num_elements=num_elements,
                use_sc=True,
                use_reduced_cg=use_reduced_cg,
                use_agnostic_product=use_agnostic_product
            )
            self.products.append(prod)
            if i == num_interactions-2:
                self.readouts.append(
                    readout_cls(
                        hidden_irreps_out,
                        MLP_irreps.simplify(),
                        gate,
                        o3.Irreps("1x0e")
                    )
                )
            elif not use_last_readout_only:
                self.readouts.append(
                    LinearReadoutBlock(
                        hidden_irreps,
                        o3.Irreps("1x0e")
                    )
                )
    def forward(
            self,
            data: Dict[str, torch.Tensor],
            training: bool = False,
            compute_force: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        ctx = prepare_graph(
            data
        )
        num_graphs = ctx.num_graphs
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths

        node_e0 = self.atomic_energies_fn(
            data["node_attrs"]
        ).squeeze(-1)
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        ).to(vectors.dtype)
        node_feats = self.node_embedding(data["node_attrs"])
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )
        energies = [e0]
        node_energies_list = [node_e0]
        node_feats_concat: List[torch.Tensor] = []

        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            node_attrs_slice = data["node_attrs"]
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
                first_layer=(i == 0),
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            node_feats_concat.append(node_feats)

        for i, readout in enumerate(self.readouts):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es = readout(node_feats_concat[feat_idx]).squeeze(-1)
            energy = scatter_sum(node_es, data["batch"], dim=0, dim_size=num_graphs)
            energies.append(energy)
            node_energies_list.append(node_es)

        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)
        node_energy = torch.sum(torch.stack(node_energies_list, dim=-1), dim=-1)
        node_feats_out = torch.cat(node_feats_concat, dim=-1)

        forces = get_outputs(
            energy=total_energy,
            positions=positions,
            training=training,
            compute_force=compute_force,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "node_feats": node_feats_out,
        }


@compile_mode("script")
class MSMACE(torch.nn.Module):
    def __init__(
            self,
            r_max: float,
            r_short: float,
            width: float,
            short_num_bessel: int,
            long_num_bessel:int,
            num_polynomial_cutoff: int,
            max_ell : int,
            interaction_cls: Type[InteractionBlock],
            interaction_cls_first: Type[InteractionBlock],
            short_num_interactions: int,
            long_num_interactions: int,
            num_elements: int,
            short_hidden_irreps: o3.Irreps,
            long_hidden_irreps: o3.Irreps,
            MLP_irreps: o3.Irreps,
            atomic_energies: np.ndarray,
            short_avg_num_neighbors: float,
            long_avg_num_neighbors: float,
            atomic_numbers: List[int],
            short_correlation: Union[int, List[int]],
            long_correlation: Union[int, List[int]],
            gate: Optional[Callable],
            apply_cutoff: bool = False,
            use_reduced_cg: bool = True,
            use_so3: bool = True,
            use_agnostic_product: bool = False,
            use_last_readout_only: bool = False,
            distance_transform: str = "None",
            edge_irreps: Optional[o3.Irreps] = None,
            use_edge_irreps_first: bool = False,
            short_radial_MLP: Optional[List[int]] = None,
            long_radial_MLP: Optional[List[int]] = None,
            radial_type: Optional[str] = "bessel",
            readout_cls: Optional[Type[NonLinearReadoutBlock]] = NonLinearReadoutBlock,
            keep_last_layer_irreps: bool = False
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64),
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "long_num_interactions", torch.tensor(long_num_interactions, dtype=torch.int64)
        )
        self.register_buffer(
            "short_num_interactions", torch.tensor(short_num_interactions, dtype=torch.int64)
        )
        self.register_buffer(
            "r_short", torch.tensor(r_short, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "width", torch.tensor(width, dtype=torch.get_default_dtype())
        )
        if isinstance(short_correlation, int):
            short_correlation = [short_correlation] * short_num_interactions
        if isinstance(long_correlation, int):
            long_correlation = [long_correlation] * long_num_interactions
        self.apply_cutoff = apply_cutoff
        self.edge_irreps = edge_irreps
        self.use_reduced_cg = use_reduced_cg
        self.use_agnostic_product = use_agnostic_product
        self.use_so3 = use_so3
        self.use_last_readout_only = use_last_readout_only
        self.use_edge_irreps_first = use_edge_irreps_first

        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])

        short_node_feats_irreps = o3.Irreps([(short_hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        long_node_feats_irreps = o3.Irreps([(long_hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.short_node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=short_node_feats_irreps
        )
        self.long_node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=long_node_feats_irreps
        )
        self.short_radial_embedding = RadialEmbeddingBlock(
            r_max=r_short+width,
            num_bessel=short_num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
            apply_cutoff=apply_cutoff
        )
        self.long_radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=long_num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
            apply_cutoff=apply_cutoff
        )
        self.cutoff_fn = PolynomialCutoffBlock(
            r_max=r_max,
            p=num_polynomial_cutoff
        )
        self.switch_functions = SwitchingFunction(
            r_short=r_short,
            width=width,
        )

        short_edge_feats_irreps = o3.Irreps(f"{self.short_radial_embedding.out_dim}x0e")
        long_edge_feats_irreps = o3.Irreps(f"{self.long_radial_embedding.out_dim}x0e")

        if not use_so3:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        else:
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell, p=1)

        short_num_features = short_hidden_irreps.count(o3.Irrep(0, 1))

        def generate_irreps(l):
            str_irrep = "+".join([f"1x{i}e+1x{i}o" for i in range(l+1)])
            return o3.Irreps(str_irrep)

        sh_irreps_inter = sh_irreps
        if short_hidden_irreps.count(o3.Irrep(0, -1)) > 0:
            sh_irreps_inter = generate_irreps(max_ell)
        short_interaction_irreps = (sh_irreps_inter * short_num_features).sort()[0].simplify()
        short_interaction_irreps_first = (sh_irreps * short_num_features).sort()[0].simplify()

        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        if short_radial_MLP is None:
            short_radial_MLP = [64, 64, 64]
        if long_radial_MLP is None:
            long_radial_MLP = [32, 32, 32]

        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)
        if short_num_interactions == 1:
            short_hidden_irreps_out = str(short_hidden_irreps[0])
        else:
            short_hidden_irreps_out = short_hidden_irreps
        edge_irreps_first = None
        if use_edge_irreps_first and edge_irreps is not None:
            edge_irreps_first = o3.Irreps(f"{edge_irreps.count(o3.Irrep(0, 1))}x0e")
        short_inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=short_node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=short_edge_feats_irreps,
            target_irreps=short_interaction_irreps_first,
            hidden_irreps=short_hidden_irreps_out,
            edge_irreps=edge_irreps_first,
            avg_num_neighbors=short_avg_num_neighbors,
            radial_MLP=short_radial_MLP
        )
        self.short_interactions = torch.nn.ModuleList([short_inter])

        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        short_node_feats_irreps_out = short_inter.target_irreps
        short_prod = EquivariantProductBasisBlock(
            node_feats_irreps=short_node_feats_irreps_out,
            target_irreps=short_hidden_irreps_out,
            correlation=short_correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first
        )
        self.short_products = torch.nn.ModuleList([short_prod])

        for i in range(short_num_interactions - 1):
            if i == short_num_interactions -2 and not keep_last_layer_irreps:
                hidden_irreps_out = str(short_hidden_irreps[0])
            else:
                hidden_irreps_out = short_hidden_irreps
            short_inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=short_hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=short_edge_feats_irreps,
                target_irreps=short_interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=short_avg_num_neighbors,
                edge_irreps=edge_irreps,
                radial_MLP=short_radial_MLP
            )
            self.short_interactions.append(short_inter)
            short_prod = EquivariantProductBasisBlock(
                node_feats_irreps=short_interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=short_correlation[i+1],
                num_elements=num_elements,
                use_sc=True,
                use_reduced_cg=use_reduced_cg,
                use_agnostic_product=use_agnostic_product
            )
            self.short_products.append(short_prod)

        long_num_features = long_hidden_irreps.count(o3.Irrep(0, 1))
        if long_hidden_irreps.count(o3.Irrep(0, -1)) > 0:
            sh_irreps_inter = generate_irreps(max_ell)
        long_interaction_irreps = (sh_irreps_inter * long_num_features).sort()[0].simplify()
        long_interaction_irreps_first = (sh_irreps * long_num_features).sort()[0].simplify()

        if long_num_interactions == 1:
            long_hidden_irreps_out = str(long_hidden_irreps[0])
        else:
            long_hidden_irreps_out = long_hidden_irreps

        long_inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=long_node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=long_edge_feats_irreps,
            target_irreps=long_interaction_irreps_first,
            hidden_irreps=long_hidden_irreps_out,
            edge_irreps=edge_irreps_first,
            avg_num_neighbors=long_avg_num_neighbors,
            radial_MLP=long_radial_MLP
        )
        self.long_interactions = torch.nn.ModuleList([long_inter])

        node_feats_irreps_out = long_inter.target_irreps
        long_prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=long_hidden_irreps_out,
            correlation=long_correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first
        )
        self.long_products = torch.nn.ModuleList([long_prod])


        for i in range(long_num_interactions - 1):
            if i == long_num_interactions -2 and not keep_last_layer_irreps:
                hidden_irreps_out = str(long_hidden_irreps[0])
            else:
                hidden_irreps_out = long_hidden_irreps
            long_inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=long_hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=long_edge_feats_irreps,
                target_irreps=long_interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=long_avg_num_neighbors,
                edge_irreps=edge_irreps,
                radial_MLP=long_radial_MLP
            )
            self.long_interactions.append(long_inter)
            long_prod = EquivariantProductBasisBlock(
                node_feats_irreps=long_interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=long_correlation[i+1],
                num_elements=num_elements,
                use_sc=True,
                use_reduced_cg=use_reduced_cg,
                use_agnostic_product=use_agnostic_product
            )
            self.long_products.append(long_prod)

        self.final_readout = readout_cls(
            (o3.Irreps(str(short_hidden_irreps[0])) + o3.Irreps(str(long_hidden_irreps[0]))).simplify(),
            MLP_irreps.simplify(),
            gate=None,
            irrep_out=o3.Irreps("1x0e")
        )

    def forward(
            self,
            data: Dict[str, torch.Tensor],
            training: bool = False,
            compute_force: bool = True,
            compute_virials: bool = False,
            compute_stress: bool = False,
            compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        ctx = prepare_ms_graph(
            data,
            compute_stress=compute_stress,
            compute_virials=compute_virials,
            compute_displacement=compute_displacement,
        )
        num_graphs = ctx.num_graphs
        positions = ctx.positions
        short_vectors = ctx.short_vectors
        long_vectors = ctx.long_vectors
        lengths = ctx.lengths
        short_lengths = ctx.short_lengths
        long_lengths = ctx.long_lengths
        cell = ctx.cell
        displacement = ctx.displacement



        node_e0 = self.atomic_energies_fn(
            data["node_attrs"]
        ).squeeze(-1)
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        ).to(short_vectors.dtype)
        short_node_feats = self.short_node_embedding(data["node_attrs"])
        long_node_feats = self.long_node_embedding(data["node_attrs"])
        short_edge_attrs = self.spherical_harmonics(short_vectors)
        long_edge_attrs = self.spherical_harmonics(long_vectors)
        short_edge_feats, _ = self.short_radial_embedding(
            short_lengths, data["node_attrs"], data["short_edge_index"], self.atomic_numbers
        )
        long_edge_feats, _ = self.long_radial_embedding(
            long_lengths, data["node_attrs"], data["long_edge_index"], self.atomic_numbers
        )
        short_cutoff = self.cutoff_fn(short_lengths)
        long_cutoff = self.cutoff_fn(long_lengths)
        short_edge_feats = short_edge_feats*self.switch_functions(short_lengths)*short_cutoff
        long_edge_feats = long_edge_feats*(1-self.switch_functions(long_lengths))*long_cutoff

        short_node_feats_concat: List[torch.Tensor] = []
        long_node_feats_concat: List[torch.Tensor] = []

        for i, (interaction, product) in enumerate(
            zip(self.short_interactions, self.short_products)
        ):
            node_attrs_slice = data["node_attrs"]
            short_node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=short_node_feats,
                edge_attrs=short_edge_attrs,
                edge_feats=short_edge_feats,
                edge_index=data["short_edge_index"],
                first_layer=(i == 0),
            )
            short_node_feats = product(
                node_feats=short_node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            short_node_feats_concat.append(short_node_feats)

        for i, (interaction, product) in enumerate(
            zip(self.long_interactions, self.long_products)
        ):
            node_attrs_slice = data["node_attrs"]
            long_node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=long_node_feats,
                edge_attrs=long_edge_attrs,
                edge_feats=long_edge_feats,
                edge_index=data["long_edge_index"],
                first_layer=(i == 0),
            )
            long_node_feats = product(
                node_feats=long_node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            long_node_feats_concat.append(long_node_feats)

        h_short = short_node_feats_concat[-1]
        h_long = long_node_feats_concat[-1]

        node_feats_fused = torch.cat(
            [h_short, h_long],
            dim=-1
        )

        node_energy = self.final_readout(
            node_feats_fused
        ).squeeze(-1)

        energy = scatter_sum(
            node_energy,
            index=data["batch"],
            dim=0,
            dim_size=num_graphs
        )

        total_energy = energy + e0


        forces, virials, stress = get_outputs(
            energy=total_energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress
        )

        return {
            "e0": e0,
            "energy": total_energy,
            "forces": forces,
            "virials": virials,
            "stress": stress,
        }






