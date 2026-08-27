import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
from ase.data import atomic_numbers as ATOMIC_NUMBERS
import re
import json


@dataclass
class Frame:
    atomic_numbers: np.ndarray
    positions: np.ndarray
    property_weights: Dict[str, float]
    energy: Optional[float] = None
    stress: Optional[np.ndarray] = None
    forces: Optional[np.ndarray] = None
    cell: Optional[np.ndarray] = None
    pbc: Optional[np.ndarray] = None


Frames = List[Frame]


def parse_properties(header: str) -> Dict[str, Tuple[int, int]]:
    if "Properties=" not in header:
        return {}

    prop_str = header.split("Properties=")[1].split()[0]
    tokens = prop_str.split(":")

    mapping = {}
    start = 0
    i = 0

    while i < len(tokens):
        name = tokens[i]
        _ = tokens[i + 1]
        ncol = int(tokens[i + 2])

        mapping[name] = (start, start + ncol)

        start += ncol
        i += 3

    return mapping


def read(
    filename: str,
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
    symbols_key: str = "species",
    positions_key: str = "pos",
    stress_key: str = "REF_stress",
    compute_stress: bool = False,
) -> Frames:

    frames: Frames = []

    with open(filename, "r", encoding="utf-8") as f:

        while True:

            line = f.readline()
            if not line:
                break

            try:
                n_atoms = int(line.strip())
            except ValueError:
                break

            header = f.readline().strip()

            # ---------------- cell ----------------
            cell = None
            if 'Lattice="' in header:
                cell_str = header.split('Lattice="')[1].split('"')[0]
                cell = np.asarray(list(map(float, cell_str.split()))).reshape(3, 3)

            # ---------------- pbc ----------------
            pbc = None
            if 'pbc="' in header:
                pbc = np.asarray(
                    [x == "T" for x in header.split('pbc="')[1].split('"')[0].split()],
                    dtype=bool
                )

            # ---------------- energy ----------------
            energy = None
            for token in header.split():
                if token.startswith(f"{energy_key}="):
                    energy = float(token.split("=", 1)[1].strip('"'))
                    break

            stress = None
            if compute_stress:
                m = re.search(
                    rf'{stress_key}="_JSON\s*(.*?)"',
                    header
                )
                if m is not None:
                    stress = np.asarray(
                        json.loads(m.group(1)),
                        dtype=np.float64
                    )
                    if stress.shape == (3, 3):
                        stress = np.array(
                            [
                                stress[0, 0],  # xx
                                stress[1, 1],  # yy
                                stress[2, 2],  # zz
                                stress[1, 2],  # yz
                                stress[0, 2],  # xz
                                stress[0, 1],  # xy
                            ],
                            dtype=np.float64,
                        )
                    # 已经是Voigt
                    elif stress.shape == (6,):
                        pass

                    else:
                        raise ValueError(
                            f"Unsupported stress shape: {stress.shape}"
                        )

            # ---------------- properties ----------------
            prop_map = parse_properties(header)

            pos_slice = prop_map.get(positions_key)
            force_slice = prop_map.get(forces_key)
            symbol_slice = prop_map.get(symbols_key)

            if pos_slice is None:
                raise ValueError(f"Missing positions key: {positions_key}")

            symbols = []
            positions = []
            forces = []
            property_weight = {}

            for _ in range(n_atoms):
                parts = f.readline().split()

                # symbol（默认单列）
                if symbol_slice is not None:
                    s = parts[symbol_slice[0]]
                else:
                    s = parts[0]

                symbols.append(s)

                positions.append(
                    parts[pos_slice[0]:pos_slice[1]]
                )

                if force_slice is not None:
                    forces.append(
                        parts[force_slice[0]:force_slice[1]]
                    )

            atomic_numbers = np.asarray(
                [ATOMIC_NUMBERS[s] for s in symbols],
                dtype=np.int64
            )

            positions = np.asarray(positions, dtype=np.float64)

            forces = (
                np.asarray(forces, dtype=np.float64)
                if force_slice is not None else None
            )

            property_weight["forces_weight"] = 1.0 if forces is not None else 0.0
            property_weight["energy_weight"] = 1.0 if energy is not None else 0.0
            property_weight["stress_weight"] = 1.0 if (compute_stress and stress is not None) else 0.0


            frames.append(
                Frame(
                    atomic_numbers=atomic_numbers,
                    positions=positions,
                    energy=energy,
                    forces=forces,
                    stress=stress,
                    cell=cell,
                    pbc=pbc,
                    property_weights=property_weight
                )
            )

    return frames
