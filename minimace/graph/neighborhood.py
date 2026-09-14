from typing import Optional, Tuple

import numpy as np
from matscipy.neighbours import neighbour_list


def get_neighborhood(
        positions: np.ndarray, #[num_positions, 3]
        cutoff: float,
        pbc: Optional[Tuple[bool, bool, bool]] = None,
        cell: Optional[np.ndarray] = None, #[3, 3]
        true_self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if pbc is None:
        pbc = (False, False, False)

    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]
    identity = np.identity(3, dtype=float)
    max_positions = np.max(np.absolute(positions)) + 1

    if not pbc_x:
        cell[0, :] = max_positions * 5 * cutoff * identity[0, :]
    if not pbc_y:
        cell[1, :] = max_positions * 5 * cutoff * identity[1, :]
    if not pbc_z:
        cell[2, :] = max_positions * 5 * cutoff * identity[2, :]

    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
    )

    if not true_self_interaction:
        true_self_edge = sender == receiver
        true_self_edge &= np.all(unit_shifts==0, axis=1)
        keep_edge = ~true_self_edge

        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    edge_index = np.stack((sender, receiver))

    shifts = np.dot(unit_shifts, cell)

    return edge_index, shifts, unit_shifts, cell

