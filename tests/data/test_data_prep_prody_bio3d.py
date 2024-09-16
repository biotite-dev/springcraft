import os
from os import path

import biotite.structure.io as bstio
import biotite.database.rcsb as rcsb
import numpy as np
import prody
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

FETCH_PDB_IDS = ["1l2y"]

# Load bio3d in R
bio3d = importr("bio3d")
# Sequence funtion in R
r_seq = robjects.r["seq"]

file_paths = rcsb.fetch(
    FETCH_PDB_IDS, format="pdb", target_path="./", overwrite=False
)

bio3d_forcefields = ("calpha", "sdenm", "pfanm")
bio3d_ff_to_springcraft = dict(
    zip(bio3d_forcefields, ["hinsen", "sdenm", "pfenm"])
)

for file_path in file_paths:
    # Use biotite to read-in structures
    atoms = bstio.load_structure(file_path, model=1)
    ca = atoms[((atoms.atom_name == "CA") & (atoms.element == "C"))]

    strucname = file_path.split(os.sep)[-1].split(".")[0]

    ## ProDy - ANM
    # Pass coordinates to ProDy
    prody_anm = prody.ANM()
    prody_anm.buildHessian(ca.coord, gamma=1.0, cutoff=13)
    
    # Eigenvalues
    prody_anm.calcModes(n_modes="all")
    
    # Fluctuations
    prody_fluc = prody.calcSqFlucts(prody_anm)
    fluc_str = f"prody_anm_13_ang_cutoff_fluctuations_{strucname}.csv"
    np.savetxt(fluc_str, prody_fluc, delimiter=",")
    
    # DCC/Cross-correlations
    prody_dcc = prody.calcCrossCorr(prody_anm, norm=True)
    dcc_str = f"prody_anm_13_ang_cutoff_dcc_norm_{strucname}.csv"
    np.savetxt(dcc_str, prody_dcc, delimiter=",")
    
    # -> Subset: first 30 non-triv. modes
    prody_dcc_norm_subset = prody.calcCrossCorr(prody_anm[0:30], norm=True)
    dcc_str_subset = f"prody_anm_13_ang_cutoff_dcc_norm_subset_{strucname}.csv"
    np.savetxt(dcc_str_subset, prody_dcc_norm_subset, delimiter=",")

    # -> Absolute values
    prody_dcc_absolute = prody.calcCrossCorr(prody_anm[0:], norm=False)
    dcc_str_absolute = (
        f"prody_anm_13_ang_cutoff_dcc_norm_subset_{strucname}.csv"
    )
    np.savetxt(dcc_str_absolute, prody_dcc_absolute, delimiter=",")

    ## ProDy - GNM
    cutoffs_gnm = [4, 7, 13]
    for c in cutoffs_gnm:
        prody_gnm = prody.GNM()

        # Kirchhoff/Eval/Evec
        prody_anm.buildKirchhoff(ca.coord, gamma=1.0, cutoff=c)
        prody_gnm.calcModes("all", zeros=True)
        kirchhoff = prody_anm.getKirchhoff()
        ref_eig_values = prody_gnm.getEigvals()
        ref_eig_vectors = prody_gnm.getEigvecs().T

        np.savetxt(f"prody_gnm_{c}_ang_cutoff_kirchhoff_{strucname}.csv", kirchhoff, delimiter=",")
        np.savetxt(f"prody_gnm_{c}_ang_cutoff_eval_{strucname}.csv", ref_eig_values, delimiter=",")
        np.savetxt(f"prody_gnm_{c}_ang_cutoff_evevs_{strucname}.csv", ref_eig_vectors, delimiter=",")

        # Fluctuations/DCC
        ref_fluc = prody.calcSqFlucts(prody_anm[0:])
        ref_dcc = prody.calcCrossCorr(prody_anm[0:])
        ref_dcc_norm_subset = prody.calcCrossCorr(prody_anm[0:16], norm=True)
        ref_dcc_absolute = prody.calcCrossCorr(prody_anm[0:], norm=False)

        np.savetxt(f"prody_gnm_{c}_ang_cutoff_fluctuations_{strucname}.csv", ref_fluc, delimiter=",")
        np.savetxt(f"prody_gnm_{c}_ang_cutoff_dcc_{strucname}.csv", ref_dcc, delimiter=",")
        np.savetxt(f"prody_gnm_{c}_ang_cutoff_dcc_norm_subset_{strucname}.csv", ref_dcc_norm_subset, delimiter=",")
        np.savetxt(f"prody_gnm_{c}_ang_cutoff_dcc_absolute_{strucname}.csv", ref_dcc_absolute, delimiter=",")

    ## Bio3d
    # Read in PDB file separately with Bio3d -> for mass-weighting
    pdb_bio3d = bio3d.read_pdb(file_path)

    # ANM-NMA for various bio3d forcefields
    for en, bio3d_ff in enumerate(bio3d_forcefields): 
        if en == 0:
            # Save reference mass
            enm_nma_bio3d = bio3d.nma(pdb=pdb_bio3d, ff=bio3d_ff, mass=True)
            bio3d_masses = np.array(enm_nma_bio3d.rx2["mass"])
            np.savetxt(
                f"bio3d_mass_{strucname}.csv", bio3d_masses, delimiter=","
            )

        # Eigenvalues (mass-weighted)
        bio3d_eigval = np.array(enm_nma_bio3d.rx2["L"])
        np.savetxt(
            f"bio3d_{bio3d_ff}_ff_eigenvalues_{strucname}.csv", 
            bio3d_eigval, 
            delimiter=","
        )

        # Mass-weighted frequencies
        bio3d_freq = np.array(enm_nma_bio3d.rx2["frequencies"])
        np.savetxt(
            f"bio3d_{bio3d_ff}_ff_frequencies_mw_{strucname}.csv",
            bio3d_freq,
            delimiter=","
        )

        # Fluctuations
        bio3d_fluc = np.array(enm_nma_bio3d.rx2["fluctuations"])
        np.savetxt(
            f"bio3d_{bio3d_ff}_ff_fluctuations_non_mw_{strucname}.csv",
            bio3d_fluc,
            delimiter=","
        )

        # Fluctuations; first 30 mode subset (apparently mass-weighted!)
        bio3d_fluc_subset = np.array(
            bio3d.fluct_nma(enm_nma_bio3d, mode_inds=r_seq(12,33))
        )
        np.savetxt(
            f"bio3d_{bio3d_ff}_ff_fluctuations_subset_mw_{strucname}.csv",
            bio3d_fluc_subset,
            delimiter=","
        )

        # Mass-weighted DCC/Cross-correlations
        bio3d_dcc = np.array(bio3d.dccm(enm_nma_bio3d))
        np.savetxt(
            f"bio3d_{bio3d_ff}_ff_dcc_mw_{strucname}.csv",
            bio3d_dcc,
            delimiter=","
        )

        # -> 30 mode subset
        bio3d_dcc_subset = np.array(
            bio3d.dccm(enm_nma_bio3d, nmodes=30)
        )
        np.savetxt(
            f"bio3d_{bio3d_ff}_ff_dcc_subset_mw_{strucname}.csv",
            bio3d_dcc_subset,
            delimiter=","
        )
