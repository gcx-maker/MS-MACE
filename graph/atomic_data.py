from copy import deepcopy
from typing import Optional, Sequence

import torch.utils.data

from minimace.data.read import Frame
from .data import Data
from .utils import (
    AtomicNumberTable,
    atomic_numbers_to_indices,
    to_one_hot,
)
from .neighborhood import get_neighborhood
from .dataloader import DataLoader
from .dataset import Dataset

class MSAtomicData(Data):

    num_graphs: torch.Tensor
    batch: torch.Tensor

    node_attrs: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor

    short_edge_index: torch.Tensor
    short_edge_vectors: torch.Tensor
    short_edge_lengths: torch.Tensor
    short_shifts: torch.Tensor
    short_unit_shifts: torch.Tensor
    short_mask: Optional[torch.Tensor]

    long_edge_index: torch.Tensor
    long_edge_vectors: torch.Tensor
    long_edge_lengths: torch.Tensor
    long_shifts: torch.Tensor
    long_unit_shifts: torch.Tensor
    long_mask: Optional[torch.Tensor]

    cell: torch.Tensor
    forces: torch.Tensor
    energy: torch.Tensor
    energy_weight: torch.Tensor
    forces_weight: torch.Tensor
    stress: torch.Tensor
    stress_weight: torch.Tensor
    pbc: torch.Tensor

    def __init__(
            self,
            short_edge_index,
            short_edge_vectors,
            short_edge_lengths,
            short_shifts,
            short_unit_shifts,

            long_edge_index,
            long_edge_vectors,
            long_edge_lengths,
            long_shifts,
            long_unit_shifts,

            node_attrs,
            edge_index,
            edge_lengths,
            edge_vectors,
            shifts,
            unit_shifts,
            positions,
            cell,
            forces,
            energy,
            energy_weight,
            forces_weight,
            pbc,
            stress,
            stress_weight,
            **extra
    ):
        num_nodes = node_attrs.shape[0]

        data = dict(
            num_nodes=num_nodes,

            short_edge_index=short_edge_index,
            short_edge_vectors=short_edge_vectors,
            short_edge_lengths=short_edge_lengths,
            short_shifts=short_shifts,
            short_unit_shifts=short_unit_shifts,

            long_edge_index=long_edge_index,
            long_edge_vectors=long_edge_vectors,
            long_edge_lengths=long_edge_lengths,
            long_shifts=long_shifts,
            long_unit_shifts=long_unit_shifts,

            node_attrs=node_attrs,
            edge_index=edge_index,
            edge_lengths = edge_lengths,
            edge_vectors=edge_vectors,
            positions=positions,
            shifts=shifts,
            unit_shifts=unit_shifts,

            cell=cell,
            forces=forces,
            energy=energy,
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            pbc=pbc,
            stress=stress,
            stress_weight=stress_weight,
        )

        data.update(extra)
        super().__init__(**data)

    @classmethod
    def from_frame(
            cls,
            frame: Frame,
            z_table: AtomicNumberTable,
            r_short: float,
            r_long: float,
            width: float,
    ):
        edge_index, shifts, unit_shifts, cell = get_neighborhood(
            positions=frame.positions,
            cutoff=r_long,
            pbc=deepcopy(frame.pbc),
            cell=deepcopy(frame.cell)
        )

        positions = torch.tensor(frame.positions, dtype=torch.get_default_dtype())

        edge_index = torch.tensor(edge_index, dtype=torch.long)
        shifts = torch.tensor(shifts, dtype=torch.get_default_dtype())
        unit_shifts = torch.tensor(unit_shifts, dtype=torch.get_default_dtype())

        edge_vectors = positions[edge_index[1]] - positions[edge_index[0]] + shifts
        edge_lengths = torch.norm(edge_vectors, dim=-1)

        short_mask = edge_lengths < r_short + width
        long_mask = edge_lengths >= r_short - width

        short_edge_index = edge_index[:, short_mask]
        short_edge_vectors = edge_vectors[short_mask]
        short_edge_lengths = edge_lengths[short_mask]
        short_shifts = shifts[short_mask]
        short_unit_shifts = unit_shifts[short_mask]

        long_edge_index = edge_index[:, long_mask]
        long_edge_vectors = edge_vectors[long_mask]
        long_edge_lengths = edge_lengths[long_mask]
        long_shifts = shifts[long_mask]
        long_unit_shifts = unit_shifts[long_mask]

        indices = atomic_numbers_to_indices(frame.atomic_numbers, z_table=z_table)

        node_attrs = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table)
        )

        cell = (
            torch.tensor(cell, dtype=torch.get_default_dtype())
            if cell is not None
            else torch.tensor(
                3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()
            ).view(3, 3)
        )

        num_atoms = len(frame.atomic_numbers)

        energy_weight = (
            torch.tensor(
                frame.property_weights.get("energy_weight"), dtype=torch.get_default_dtype()
            )
            if frame.property_weights.get("energy_weight") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        forces_weight = (
            torch.tensor(
                frame.property_weights.get("forces_weight"), dtype=torch.get_default_dtype()
            )
            if frame.property_weights.get("forces_weight") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        stress_weight = (
            torch.tensor(
                frame.property_weights.get("stress_weight"),
                dtype=torch.get_default_dtype(),
            )
            if frame.property_weights.get("stress_weight") is not None
            else torch.tensor(
                0.0,
                dtype=torch.get_default_dtype(),
            )
        )

        stress = (
            torch.tensor(
                frame.stress,
                dtype=torch.get_default_dtype(),
            )
            if frame.stress is not None
            else None
        ).unsqueeze(0)

        forces = (
            torch.tensor(frame.forces, dtype=torch.get_default_dtype())
            if frame.forces is not None else torch.zeros(num_atoms, 3, dtype=torch.get_default_dtype())
        )

        energy = (
            torch.tensor(
                frame.energy, dtype=torch.get_default_dtype()
            )
            if frame.energy is not None else torch.tensor(0.0, dtype=torch.get_default_dtype())
        )

        if frame.pbc is not None:
            pbc = list(bool(pbc_) for pbc_ in frame.pbc)
        else:
            pbc = None
        pbc = (
            torch.tensor([pbc], dtype=torch.bool)
            if pbc is not None
            else torch.tensor([[False, False, False]], dtype=torch.bool)
        )
        positions = torch.tensor(frame.positions, dtype=torch.get_default_dtype())



        return cls(
            short_edge_index=short_edge_index,
            short_edge_vectors=short_edge_vectors,
            short_edge_lengths=short_edge_lengths.unsqueeze(-1),
            short_shifts=short_shifts,
            short_unit_shifts=short_unit_shifts,
            short_mask=short_mask,

            long_edge_index=long_edge_index,
            long_edge_vectors=long_edge_vectors,
            long_edge_lengths=long_edge_lengths.unsqueeze(-1),
            long_shifts=long_shifts,
            long_unit_shifts=long_unit_shifts,
            long_mask=long_mask,

            node_attrs=node_attrs,
            edge_index=edge_index,
            edge_lengths=edge_lengths.unsqueeze(-1),
            edge_vectors=edge_vectors,
            shifts=shifts,
            unit_shifts=unit_shifts,
            positions=positions,
            cell=cell,
            forces=forces,
            energy=energy,
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            pbc=pbc,
            stress=stress,
            stress_weight=stress_weight,
        )


class AtomicData(Data):
    num_graphs: torch.Tensor
    batch: torch.Tensor
    edge_index: torch.Tensor
    node_attrs: torch.Tensor
    edge_vectors: torch.Tensor
    edge_lengths: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor
    unit_shifts: torch.Tensor
    cell: torch.Tensor
    forces: torch.Tensor
    energy: torch.Tensor
    energy_weight: torch.Tensor
    forces_weight: torch.Tensor

    def __init__(
            self,
            edge_index: torch.Tensor,  #[2. n_edges]
            node_attrs: torch.Tensor,   #[n_nodes, n_node_features]
            positions: torch.Tensor,     #[n_nodes, 3]
            shifts: torch.Tensor,           #[n_edges, 3]
            unit_shifts: torch.Tensor,   #[n_edges, 3]
            cell: Optional[torch.Tensor], #[3, 3]
            forces: Optional[torch.Tensor], #[n_nodes, 3]
            energy:Optional[torch.Tensor], #[,]
            energy_weight: Optional[torch.Tensor],  # [,]
            forces_weight: Optional[torch.Tensor],  # [,]
            pbc: Optional[torch.Tensor] = None, #[, 3]
            **extra_data: torch.Tensor,
    ):
        num_nodes = node_attrs.shape[0]

        assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
        assert positions.shape == (num_nodes, 3)
        assert shifts.shape[1] == 3
        assert unit_shifts.shape[1] == 3
        assert energy_weight is None or len(energy_weight.shape) == 0
        assert forces_weight is None or len(forces_weight.shape) == 0
        assert len(node_attrs.shape) == 2
        assert cell is None or cell.shape == (3, 3)
        assert forces is None or forces.shape == (num_nodes, 3)
        assert energy is None or len(energy.shape) == 0
        assert pbc is None or (pbc.shape[-1] == 3 and pbc.dtype == torch.bool)

        data = {
            "num_nodes": num_nodes,
            "edge_index": edge_index,
            "positions": positions,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "cell": cell,
            "node_attrs": node_attrs,
            "forces": forces,
            "energy": energy,
            "pbc": pbc,
            "energy_weight": energy_weight,
            "forces_weight": forces_weight,
        }
        data.update(extra_data)
        super().__init__(**data)

    @classmethod
    def from_frame(
            cls,
            frame: Frame,
            z_table: AtomicNumberTable,
            cutoff: float,
    ) -> "AtomicData":
        edge_index, shifts, unit_shifts, cell = get_neighborhood(
            positions=frame.positions,
            cutoff=cutoff,
            pbc=deepcopy(frame.pbc),
            cell=deepcopy(frame.cell)
        )
        indices = atomic_numbers_to_indices(frame.atomic_numbers, z_table=z_table)
        one_hot = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table)
        )
        cell = (
            torch.tensor(cell, dtype=torch.get_default_dtype())
            if cell is not None
            else torch.tensor(
                3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()
            ).view(3, 3)
        )

        num_atoms = len(frame.atomic_numbers)

        energy_weight = (
            torch.tensor(
                frame.property_weights.get("energy_weight"), dtype=torch.get_default_dtype()
            )
            if frame.property_weights.get("energy_weight") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        forces_weight = (
            torch.tensor(
                frame.property_weights.get("forces_weight"), dtype=torch.get_default_dtype()
            )
            if frame.property_weights.get("forces_weight") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        forces = (
            torch.tensor(frame.forces, dtype=torch.get_default_dtype())
            if frame.forces is not None else torch.zeros(num_atoms, 3, dtype=torch.get_default_dtype())
        )

        energy = (
            torch.tensor(
                frame.energy, dtype=torch.get_default_dtype()
            )
            if frame.energy is not None else torch.tensor(0.0, dtype=torch.get_default_dtype())
        )

        if frame.pbc is not None:
            pbc = list(bool(pbc_) for pbc_ in frame.pbc)
        else:
            pbc = None
        pbc = (
            torch.tensor([pbc], dtype=torch.bool)
            if pbc is not None
            else torch.tensor([[False, False, False]], dtype=torch.bool)
        )
        positions = torch.tensor(frame.positions, dtype=torch.get_default_dtype())

        cls_kwargs = dict(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            positions=positions,
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(unit_shifts, dtype=torch.get_default_dtype()),
            cell=cell,
            node_attrs=one_hot,
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            forces=forces,
            energy=energy,
            pbc=pbc
        )

        return cls(**cls_kwargs)


def get_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle=True,
    drop_last=False,
) -> torch.utils.data.DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


