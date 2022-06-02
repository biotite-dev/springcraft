import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import biotite.structure as struc
import biotite.structure.io.mmtf as mmtf
import biotite.database.rcsb as rcsb
import springcraft
import ammolite


N_FORCE = 1000
TARGET_RES_ID = 120
CLUSTERS = 4
SPHERE_RADIUS = 5


def create_fibonacci_points(n):
    """
    Get an array of approximately equidistant points on a unit sphere
    surface using a golden section spiral.
    """
    phi = (3 - np.sqrt(5)) * np.pi * np.arange(n)
    z = np.linspace(1 - 1.0/n, 1.0/n - 1, n)
    radius = np.sqrt(1 - z*z)
    coord = np.zeros((n, 3))
    coord[:,0] = radius * np.cos(phi)
    coord[:,1] = radius * np.sin(phi)
    coord[:,2] = z
    return coord


mmtf_file = mmtf.MMTFFile.read(rcsb.fetch("1CEX", "mmtf"))
atoms = mmtf.get_structure(mmtf_file, model=1)
ca = atoms[(atoms.atom_name == "CA") & struc.filter_amino_acids(atoms)]
target_index = np.where(ca.res_id == TARGET_RES_ID)[0][0]

ff = springcraft.TabulatedForceField.sd_enm(ca)
anm = springcraft.ANM(ca, ff)
#!#
#np.random.seed(1)
#cov = np.random.rand(*anm.covariance.shape)
#anm.covariance = cov * cov.T
#!#

# Create evenly distributed force vectors
force_vectors = create_fibonacci_points(N_FORCE)
displacements = []
for i, force_vec in enumerate(force_vectors):
    force = np.zeros((ca.array_length(), 3))
    force[target_index] = force_vec
    displacement = anm.linear_response(force)
    # Filter displacement to important residues for function
    displacement = displacement[
        ((ca.res_id >=  80) & (ca.res_id <=  90)) |
        ((ca.res_id >= 179) & (ca.res_id <= 187)) |
        ((ca.res_id >=  42) & (ca.res_id <=  45))
    ]
    displacements.append(displacement)
displacements = np.stack(displacements)


# remove 'artificial' strong amplitude of target and adjacent atoms
displacements[:, target_index-1 : target_index+2, :] = 0
# Reshape into feature vector
displacements = displacements.reshape(N_FORCE, -1)
kmeans = KMeans(n_clusters=CLUSTERS, random_state=0).fit(displacements)
clusters = kmeans.labels_


# Create 'dummy' molecule to display sphere in PyMOL
dummy = struc.AtomArray(N_FORCE)
dummy.chain_id[:] = "D"
dummy.res_id[:] = 1
dummy.res_name[:] = "DUM"
dummy.hetero[:] = True
dummy.atom_name[:] = "DUM"
dummy.element[:] = "C"
dummy.coord = ca.coord[target_index] + force_vectors * SPHERE_RADIUS

ammolite.launch_interactive_pymol()

pymol_protein = ammolite.PyMOLObject.from_structure(ca)
pymol_dummy = ammolite.PyMOLObject.from_structure(dummy)
pymol_dummy.show_as("nb_spheres")
for cluster, color in zip(np.unique(clusters), ["red", "green", "blue", "yellow"]):
    pymol_dummy.color(color, clusters == cluster)
pymol_dummy.set("sphere_scale", 0.5)
