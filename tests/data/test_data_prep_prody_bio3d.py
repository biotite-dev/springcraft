import abc
import os
from os import path

import biotite.structure as struc
import biotite.structure.io as bstio
import biotite.database.rcsb as rcsb
import numpy as np
import prody
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import DataFrame as DataFrame_r
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter

FETCH_PDB_IDS = ["1l2y"]

# Load bio3d in R
bio3d = importr("bio3d")
# Sequence funtion in R
r_seq = robjects.r["seq"]


# Convert AtomArray into bio3d-PDB objects
def aarray_to_bio3d(aarray):
    coords = aarray.coord

    # Resid names
    residue_starts = struc.get_residue_starts(aarray)
    res_names = aarray.res_name[residue_starts]

    xyz_r = robjects.r.matrix(robjects.FloatVector(coords.ravel()), nrow=1)

    assert not any(aarray.hetero)
    # Local converter for Numpy -> R conversion
    with localconverter(robjects.default_converter + numpy2ri.converter):
        # TODO: HETATM should also be included
        #       -> AtomArrays should only contain protein atoms here for now
        type_r = robjects.StrVector(["ATOM"] * len(aarray))
        atom_id_r = robjects.IntVector(np.arange(1, len(aarray) + 1))
        atom_names_r = robjects.StrVector(aarray.atom_name)
        alt_r = robjects.StrVector(["NA"] * len(aarray))
        res_names_r = robjects.StrVector(aarray.res_name)
        chain_r = robjects.StrVector(aarray.chain_id)
        resid_r = robjects.IntVector(aarray.res_id)
        x_r = robjects.IntVector(coords[:, 0])
        y_r = robjects.IntVector(coords[:, 1])
        z_r = robjects.IntVector(coords[:, 2])
        o_r = robjects.IntVector([1] * len(aarray))
        b_r = robjects.IntVector([0] * len(aarray))

    # Create a R dataframe
    # equivalent of res_name is resid in bio3d
    atoms_r = DataFrame_r(
        {
            "type": type_r,
            "eleno": atom_id_r,
            "elety": atom_names_r,
            "alt": alt_r,
            "resid": res_names_r,
            "chain": chain_r,
            "resno": resid_r,
            "x": x_r,
            "y": y_r,
            "z": z_r,
            "o": o_r,
            "b": b_r,
        }
    )

    # Create bio3d PDB -> ListVector (list in R)
    pdb_bio3d = robjects.ListVector(
        {
            "xyz": xyz_r,
            "atom": atoms_r,
            "calpha": robjects.NULL,
        }
    )
    # Create S3 R class object
    pdb_bio3d.rclass = robjects.StrVector(["pdb", "sse"])

    # Get indices of Calpha atoms, as required by bio3d
    ca = aarray[aarray.atom_name == "CA"]
    ca_inds = ca.res_id
    seq_all_atoms = np.arange(aarray.res_id[0], aarray.res_id[-1] + 1)

    # Local converter for Numpy -> R conversion
    with localconverter(robjects.default_converter + numpy2ri.converter):
        pdb_bio3d.rx2["calpha"] = np.isin(seq_all_atoms, ca_inds)

    return pdb_bio3d


file_paths = rcsb.fetch(FETCH_PDB_IDS, format="pdb", target_path="./", overwrite=False)


## Test data-generating functions
# -> Enforce consistent naming of output .csv
def prody_enm_nma(enm_type, structure_path, cutoff_list, output_markers="all"):
    # Check cutoff_list and outputs
    all_types_in_list = all([isinstance(c, (int, float)) for c in cutoff_list])
    if not isinstance(cutoff_list, list) and all_types_in_list:
        raise ValueError(
            "A list of integers/floats is expected as input for 'cutoff_list'."
        )

    accepted_outputs = [
        "hess_kirchhoff",
        "evals",
        "evecs",
        "fluctuations",
        "dcc_norm",
        "dcc_norm_subset",
        "dcc_absolute",
    ]
    # Check output string/list of strings
    outputs = accepted_outputs if output_markers == "all" else output_markers
    if not all([o in accepted_outputs for o in outputs]):
        raise ValueError(
            "Only the following options are accepted for 'output_markers': \n",
            f"{accepted_outputs}",
        )

    # Structure I/O
    in_struc = bstio.load_structure(structure_path, model=1)
    ca = in_struc[
        ((struc.filter_canonical_amino_acids(in_struc)) & (in_struc.atom_name == "CA"))
    ]

    for c in cutoff_list:
        # Generalized ENM/Kirchhoff or Hessian w. pattern matching
        match enm_type:
            # Select either ANM or GNMs
            case "anm":
                prody_enm = prody.ANM()
                prody_enm.buildHessian(ca.coord, gamma=1.0, cutoff=c)
                structure_info_matrix = prody_enm.getHessian()
                first_non_triv_mode = 6
                last_subset_mode = 36
            case "gnm":
                prody_enm = prody.GNM()
                prody_enm.buildKirchhoff(ca.coord, gamma=1.0, cutoff=c)
                structure_info_matrix = prody_enm.getKirchhoff()
                first_non_triv_mode = 1
                last_subset_mode = 16 + 1
            case _:
                raise ValueError(
                    "Only 'gnm' and 'anm' are valid inputs for 'enm_type'."
                )

        # NOTE: Prody only computes non-trivial modes in the standard case
        # First trivial mode -> last mode
        prody_enm.calcModes(n_modes="all", zeros=True)

        strucname = structure_path.split(os.sep)[-1].split(".")[0]

        for o in outputs:
            match o:
                case "hess_kirchhoff":
                    prody_output = structure_info_matrix
                    o = "hessian" if enm_type == "anm" else "kirchhoff"
                case "evals":
                    prody_output = prody_enm.getEigvals()
                case "evecs":
                    prody_output = prody_enm.getEigvecs().T
                case "fluctuations":
                    prody_output = prody.calcSqFlucts(prody_enm)
                case "dcc_norm":
                    prody_output = prody.calcCrossCorr(prody_enm)
                case "dcc_norm_subset":
                    prody_output = prody.calcCrossCorr(
                        prody_enm[first_non_triv_mode:last_subset_mode], norm=True
                    )
                case "dcc_absolute":
                    prody_output = prody.calcCrossCorr(prody_enm, norm=False)
            out_str = f"prody_{enm_type}_{c}_ang_cutoff_{o}_{strucname}.csv.gz"
            np.savetxt(out_str, prody_output, delimiter=",")


def bio3d_anm_nma(structure_path, bio3d_ff, output_markers="all"):
    # Check bio3d_ff and outputs
    accepted_bio3d_ff = ["calpha", "sdenm", "pfanm"]
    if not bio3d_ff in accepted_bio3d_ff:
        raise ValueError(
            "Onle the following bio3d FFs are accepted: \n", f"{accepted_bio3d_ff}"
        )

    accepted_outputs = [
        "masses",
        "hessian",
        "evals_mw",
        "frequencies_mw",
        "fluctuations_non_mw",
        "fluctuations_subset_mw",
        "dcc_mw",
        "dcc_subset_mw",
    ]
    # Check output string/list of strings
    outputs = accepted_outputs if output_markers == "all" else output_markers
    if not all([o in accepted_outputs for o in outputs]):
        raise ValueError(
            "Only the following options are accepted for 'output_markers': \n",
            f"{accepted_outputs}",
        )

    # Get strucname; structure I/O w. bio3d
    strucname = structure_path.split(os.sep)[-1].split(".")[0]
    # Create bio3d NMA; read full PDB
    pdb_bio3d = bio3d.read_pdb(structure_path)
    enm_nma_bio3d = bio3d.nma(pdb=pdb_bio3d, ff=bio3d_ff, mass=True)

    for o in outputs:
        match o:
            case "masses":
                bio3d_output = np.array(enm_nma_bio3d.rx2["mass"])
            case "hessian":
                ## Have to be computed separately; AtomArrays -> bio3d-PDB objects
                # Structure I/O w. biotite
                in_struc = bstio.load_structure(structure_path, model=1)
                ca = in_struc[
                    (
                        (struc.filter_canonical_amino_acids(in_struc))
                        & (in_struc.atom_name == "CA")
                    )
                ]
                ff_bio3d = bio3d.load_enmff(ff=bio3d_ff)
                pdb_bio3d = aarray_to_bio3d(ca)
                bio3d_output = bio3d.build_hessian(
                    pdb_bio3d.rx2("xyz"), pfc_fun=ff_bio3d, pdb=pdb_bio3d
                )
            case "evals_mw":
                bio3d_output = np.array(enm_nma_bio3d.rx2["L"])
            case "frequencies_mw":
                bio3d_output = np.array(enm_nma_bio3d.rx2["frequencies"])
            case "fluctuations_non_mw":
                bio3d_output = np.array(enm_nma_bio3d.rx2["fluctuations"])
            case "fluctuations_subset_mw":
                bio3d_output = np.array(
                    bio3d.fluct_nma(enm_nma_bio3d, mode_inds=r_seq(12, 33))
                )
            case "dcc_mw":
                bio3d_output = np.array(bio3d.dccm(enm_nma_bio3d))
            case "dcc_subset_mw":
                bio3d_output = np.array(bio3d.dccm(enm_nma_bio3d, nmodes=30))
        if not o == "masses":
            out_str = f"bio3d_anm_{bio3d_ff}_ff_{o}_{strucname}.csv.gz"
        else:
            out_str = f"bio3d_mass_{strucname}.csv.gz"
        np.savetxt(out_str, bio3d_output, delimiter=",")


## Generate CSV files
path_1l2y = "1l2y.pdb"
anm_prody_cutoffs = [13]
# ANM - Prody
anm_prody_out = ["evals", "fluctuations", "dcc_norm", "dcc_norm_subset", "dcc_absolute"]
prody_enm_nma("anm", path_1l2y, anm_prody_cutoffs, anm_prody_out)

# GNM - Prody
gnm_prody_cutoffs = [4, 7, 13]
gnm_prody_out = [
    "hess_kirchhoff",
    "evals",
    "evecs",
    "fluctuations",
    "dcc_norm",
    "dcc_norm_subset",
    "dcc_absolute",
]
prody_enm_nma("gnm", path_1l2y, gnm_prody_cutoffs, gnm_prody_out)

# ANMs: sdENM, pfENM and Hinsen/Calpha - bio3d
bio3d_forcefields = ("calpha", "sdenm", "pfanm")

for bff in bio3d_forcefields:
    bio3d_anm_nma(path_1l2y, bff, "all")

# Test Hessians with random coordinates
SEED = [1, 323, 777, 999]
N_ATOMS = 500
BOX_SIZE = 40

cutoffs_for_random_coords = [5, 10, 15]

for s in SEED:
    np.random.seed(SEED)
    coord = np.random.rand(N_ATOMS, 3) * BOX_SIZE
    np.savetxt(f"random_coord_seed_{s}.csv.gz", coord, delimiter=",")

    for c in cutoffs_for_random_coords:
        # Random coord. GNM
        random_gnm = prody.GNM()
        random_gnm.buildKirchhoff(coord, gamma=1.0, cutoff=c)
        random_kirchhoff = random_gnm.getKirchhoff()

        np.savetxt(
            f"prody_gnm_{c}_ang_cutoff_kirchhoff_random_coords_seed_{s}.csv.gz",
            random_kirchhoff,
            delimiter=",",
        )

        # Random coord. ANM
        # -> broken Hessian; degenerate eigvals for low cutoffs
        if c < 10:
            continue
        random_anm = prody.ANM()
        random_anm.buildHessian(coord, gamma=1.0, cutoff=c)
        random_hessian = random_anm.getHessian()

        np.savetxt(
            f"prody_anm_{c}_ang_cutoff_hessian_random_coords_seed_{s}.csv.gz",
            random_hessian,
            delimiter=",",
        )
