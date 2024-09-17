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

 
    xyz_r = robjects.r.matrix(robjects.FloatVector(coords.ravel()), 
                             nrow = 1
                            )
    
    assert not any(aarray.hetero)
    # Local converter for Numpy -> R conversion
    with localconverter(robjects.default_converter + numpy2ri.converter):
        # TODO: HETATM should also be included 
        #       -> AtomArrays should only contain protein atoms here for now
        type_r = robjects.StrVector(["ATOM"]*len(aarray)) 
        atom_id_r = robjects.IntVector(np.arange(1, len(aarray) + 1))
        atom_names_r = robjects.StrVector(aarray.atom_name)
        alt_r = robjects.StrVector(["NA"]*len(aarray))
        res_names_r = robjects.StrVector(aarray.res_name)
        chain_r = robjects.StrVector(aarray.chain_id)
        resid_r = robjects.IntVector(aarray.res_id)
        x_r = robjects.IntVector(coords[:,0])
        y_r = robjects.IntVector(coords[:,1])
        z_r = robjects.IntVector(coords[:,2])
        o_r = robjects.IntVector([1]*len(aarray))
        b_r = robjects.IntVector([0]*len(aarray))

    # Create a R dataframe
    # equivalent of res_name is resid in bio3d 
    atoms_r = DataFrame_r({
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
                            "b": b_r
                        })

    # Create bio3d PDB -> ListVector (list in R)
    pdb_bio3d = robjects.ListVector({"xyz": xyz_r, 
                                     "atom": atoms_r,
                                     "calpha": robjects.NULL,
                                     })
    # Create S3 R class object
    pdb_bio3d.rclass = robjects.StrVector(["pdb", "sse"])

    # Get indices of Calpha atoms, as required by bio3d
    ca = aarray[aarray.atom_name=="CA"]
    ca_inds = ca.res_id
    seq_all_atoms = np.arange(aarray.res_id[0], aarray.res_id[-1] + 1)
    
    # Local converter for Numpy -> R conversion
    with localconverter(robjects.default_converter + numpy2ri.converter):
        pdb_bio3d.rx2["calpha"] = np.isin(
                        seq_all_atoms, ca_inds
                        )

    return pdb_bio3d

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
        prody_gnm.buildKirchhoff(ca.coord, gamma=1.0, cutoff=c)
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

        # Hessian -> Compute separately; AtomArray -> bio3d-PDB objects
        ff_bio3d = bio3d.load_enmff(ff=bio3d_ff)
        pdb_bio3d = aarray_to_bio3d(ca)
        ref_hessian = bio3d.build_hessian(
            pdb_bio3d.rx2("xyz"), 
            pfc_fun=ff_bio3d,
            pdb=pdb_bio3d
        )
        np.savetxt(
            f"bio3d_{bio3d_ff}_ff_hessian_{strucname}.csv", 
            ref_hessian, 
            delimiter=","
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
