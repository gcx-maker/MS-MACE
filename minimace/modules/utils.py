###########################################################################################
# Utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
from scipy.constants import c, e

from minimace.tools.scatter import scatter_mean, scatter_std, scatter_sum
from minimace.graph.batch import Batch

from .blocks import AtomicEnergiesBlock


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()


def compute_forces(
    energy: torch.Tensor, positions: torch.Tensor, training: bool = True
) -> torch.Tensor:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    gradient = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,  # For complete dissociation turn to true
    )[
        0
    ]  # [n_nodes, 3]
    if gradient is None:
        return torch.zeros_like(positions)
    return -1 * gradient


def compute_forces_virials(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = True,
    compute_stress: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    forces, virials = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions, displacement],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,
    )
    stress = torch.zeros_like(displacement)
    if compute_stress and virials is not None:
        cell = cell.view(-1, 3, 3)
        volume = torch.linalg.det(cell).abs().unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
        stress = torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))
    if forces is None:
        forces = torch.zeros_like(positions)
    if virials is None:
        virials = torch.zeros((1, 3, 3))

    return -1 * forces, -1 * virials, stress


def get_symmetric_displacement(
    positions: torch.Tensor,
    unit_shifts: torch.Tensor,
    cell: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    num_graphs: int,
    batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cell is None:
        cell = torch.zeros(
            num_graphs * 3,
            3,
            dtype=positions.dtype,
            device=positions.device,
        )
    sender = edge_index[0]
    displacement = torch.zeros(
        (num_graphs, 3, 3),
        dtype=positions.dtype,
        device=positions.device,
    )
    displacement.requires_grad_(True)
    symmetric_displacement = 0.5 * (
        displacement + displacement.transpose(-1, -2)
    )  # From https://github.com/mir-group/nequip
    positions = positions + torch.einsum(
        "be,bec->bc", positions, symmetric_displacement[batch]
    )
    cell = cell.view(-1, 3, 3)
    cell = cell + torch.matmul(cell, symmetric_displacement)
    shifts = torch.einsum(
        "be,bec->bc",
        unit_shifts,
        cell[batch[sender]],
    )
    return positions, shifts, displacement


def get_outputs(
    energy: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    displacement: Optional[torch.Tensor],
    training: bool = False,
    compute_force: bool = True,
    compute_virials: bool = True,
    compute_stress: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if (compute_virials or compute_stress) and displacement is not None:
        forces, virials, stress = compute_forces_virials(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            compute_stress=compute_stress,
            training=training,
        )
    elif compute_force:
        forces, virials, stress = (
            compute_forces(
                energy=energy,
                positions=positions,
                training=training,
            ),
            None,
            None,
        )
    else:
        forces, virials, stress = (None, None, None)
    return forces, virials, stress


def get_edge_vectors_and_lengths(
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor,
        normalize: bool = False,
        eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths


def compute_avg_num_neighbors(data_loader:torch.utils.data.DataLoader) -> float:
    num_neighbors = []
    for batch in data_loader:
        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )
    return to_numpy(avg_num_neighbors).item()


def compute_statistics(
        data_loader: torch.utils.data.DataLoader,
        atomic_energies: np.ndarray,
) -> Tuple[float, float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

    atom_energy_list = []
    forces_list = []
    num_neighbors = []
    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=0, dim_size=batch.num_graphs
        )[torch.arange(batch.num_graphs)]
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_energy_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )
        forces_list.append(batch.forces)
        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    atom_energies = torch.cat(atom_energy_list, dim=0)
    forces = torch.cat(forces_list, dim=0)

    mean = torch.mean(atom_energies).item()

    rms = torch.sqrt(
        torch.mean(torch.square(forces))
    ).item()

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )
    return to_numpy(avg_num_neighbors).item(), mean, rms


class GraphContext(NamedTuple):
    num_graphs: int
    num_atoms_arange: torch.Tensor
    displacement: Optional[torch.Tensor]
    positions: torch.Tensor
    cell: torch.Tensor
    vectors: Optional[torch.Tensor] = None
    lengths: Optional[torch.Tensor] = None
    short_vectors: Optional[torch.Tensor] = None
    short_lengths: Optional[torch.Tensor] = None
    long_vectors: Optional[torch.Tensor] = None
    long_lengths: Optional[torch.Tensor] = None



def prepare_graph(
        data: Dict[str, torch.Tensor],
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
) -> GraphContext:
    if not (hasattr(torch, "compiler") and torch.compiler.is_compiling()):
        data["positions"].requires_grad_(True)
    positions = data["positions"]
    cell = data["cell"]
    num_atoms_arange = torch.arange(positions.shape[0], device=positions.device)
    num_graphs = int(data["ptr"].numel() - 1)
    displacement = torch.zeros((num_graphs, 3, 3), dtype=positions.dtype, device=positions.device)
    if compute_virials or compute_stress or compute_displacement:
        p, s, displacement = get_symmetric_displacement(
            positions=positions,
            unit_shifts=data["unit_shifts"],
            cell=cell,
            edge_index=data["edge_index"],
            num_graphs=num_graphs,
            batch=data["batch"],
        )
        data["positions"], data["shifts"] = p, s
    vectors, lengths = get_edge_vectors_and_lengths(
        positions=data["positions"],
        edge_index=data["edge_index"],
        shifts=data["shifts"]
    )
    return GraphContext(
        num_graphs=num_graphs,
        num_atoms_arange=num_atoms_arange,
        displacement=displacement,
        positions=positions,
        vectors=vectors,
        lengths=lengths,
        cell=cell
    )


def prepare_ms_graph(
    data: Dict[str, torch.Tensor],
    compute_virials: bool = False,
    compute_stress: bool = False,
    compute_displacement: bool = False,
) -> GraphContext:

    if not (hasattr(torch, "compiler") and torch.compiler.is_compiling()):
        data["positions"].requires_grad_(True)

    positions = data["positions"]
    cell = data["cell"]

    num_atoms_arange = torch.arange(
        positions.shape[0],
        device=positions.device,
    )

    num_graphs = int(data["ptr"].numel() - 1)
    displacement = torch.zeros((num_graphs, 3, 3), dtype=positions.dtype, device=positions.device)
    if compute_virials or compute_stress or compute_displacement:
        p, s, displacement = get_symmetric_displacement(
            positions=positions,
            unit_shifts=data["unit_shifts"],
            cell=cell,
            edge_index=data["edge_index"],
            num_graphs=num_graphs,
            batch=data["batch"],
        )
        data["positions"], data["shifts"] = p, s

    vectors, lengths = get_edge_vectors_and_lengths(
        positions=data["positions"],
        edge_index=data["edge_index"],
        shifts=data["shifts"]
    )
    short_vectors, short_lengths = get_edge_vectors_and_lengths(
        positions=data["positions"],
        edge_index=data["short_edge_index"],
        shifts=data["short_shifts"]
    )

    long_vectors, long_lengths = get_edge_vectors_and_lengths(
        positions=data["positions"],
        edge_index=data["long_edge_index"],
        shifts=data["long_shifts"]
    )

    return GraphContext(
        num_graphs=num_graphs,
        num_atoms_arange=num_atoms_arange,
        displacement=displacement,
        positions=data["positions"],

        vectors=vectors,
        lengths=lengths,

        short_vectors=short_vectors,
        short_lengths=short_lengths,

        long_vectors=long_vectors,
        long_lengths=long_lengths,

        cell=cell,
    )
