from os.path import dirname, join, realpath

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx

import springcraft


def data_dir():
    return join(dirname(realpath(__file__)), "data")


def load_protein_structure(pdb_id: str) -> struc.AtomArray:
    file_path = join(dirname(realpath(__file__)), "data", pdb_id + ".cif")
    cif_file = pdbx.CIFFile.read(file_path)
    atoms = pdbx.get_structure(cif_file, model=1)
    return atoms[(atoms.atom_name == "CA") & (atoms.element == "C")]  # pyright: ignore[reportReturnType, reportIndexIssue]


def prepare_gnm(pdb_id: str, cutoff: float | int) -> springcraft.GNM:
    ca = load_protein_structure(pdb_id)
    ff = springcraft.InvariantForceField(cutoff)
    return springcraft.GNM(ca, ff)


def prepare_anm(pdb_id: str, cutoff: float | int) -> springcraft.ANM:
    ca = load_protein_structure(pdb_id)
    ff = springcraft.InvariantForceField(cutoff)
    return springcraft.ANM(ca, ff)
