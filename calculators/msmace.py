from ase.calculators.calculator import Calculator
from ase.calculators.calculator import all_changes
from minimace.graph.utils import AtomicNumberTable
from minimace.graph.batch import Batch
from minimace.graph.atomic_data import MSAtomicData
from minimace.data.read import Frame
import numpy as np
import torch


class MSMACECalculator(Calculator):

    implemented_properties = [
        "energy",
        "forces",
        "free_energy",
        "stress",
        "virials"
    ]

    def __init__(
        self,
        model_path,
        device="cpu",
        energy_units_to_eV=1.0,
        length_units_to_A=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.device = device

        self.model = torch.load(
            model_path,
            map_location=device
        )

        self.model.to(device)

        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        self.r_max = float(self.model.r_max)
        self.r_short = float(self.model.r_short)
        self.width = float(self.model.width)

        self.z_table = AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )

        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A

    @staticmethod
    def atoms_to_frame(atoms):
        return Frame(
            atomic_numbers=np.array(atoms.numbers),
            positions=np.array(atoms.positions),

            energy=None,
            forces=None,

            cell=np.array(atoms.cell),

            pbc=np.array(atoms.pbc),

            property_weights={}
        )

    def calculate(
            self,
            atoms=None,
            properties=None,
            system_changes=all_changes,
    ):
        super().calculate(
            atoms,
            properties,
            system_changes,
        )

        frame = self.atoms_to_frame(atoms)

        graph = MSAtomicData.from_frame(
            frame=frame,
            z_table=self.z_table,
            r_short=self.r_short,
            r_long=self.r_max,
            width=self.width,
        )

        batch = Batch.from_data_list([graph])

        batch = batch.to(self.device)

        compute_force = (
                properties is None
                or "forces" in properties
        )

        compute_stress = (
                properties is not None
                and (
                        "stress" in properties
                        or "virials" in properties
                )
        )

        with torch.enable_grad():
            output = self.model(
                batch,
                training=False,
                compute_force=compute_force,
                compute_virials=compute_stress,
                compute_stress=compute_stress,
            )

            energy = (
                output["energy"]
                .detach()
                .cpu()
                .item()
            )
        self.results = {
            "energy": energy,
            "free_energy": energy,
        }

        if compute_force:
            self.results["forces"] = (
                output["forces"]
                .detach()
                .cpu()
                .numpy()
            )

        if compute_stress:

            stress = (
                output["stress"]
                .detach()
                .cpu()
                .numpy()
            )

            virials = (
                output["virials"]
                .detach()
                .cpu()
                .numpy()
            )

            # (1,3,3) -> (3,3)
            if stress.ndim == 3:
                stress = stress[0]

            # (1,3,3) -> (3,3)
            if virials.ndim == 3:
                virials = virials[0]

            # ASE要求Voigt格式:
            # (xx, yy, zz, yz, xz, xy)
            ase_stress = np.array([
                stress[0, 0],
                stress[1, 1],
                stress[2, 2],
                stress[1, 2],
                stress[0, 2],
                stress[0, 1],
            ])

            self.results["stress"] = ase_stress
            self.results["virials"] = virials


