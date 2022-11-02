library(bio3d)

wd <- getwd()
print(wd)

pdb <- read.pdb("1l2y.pdb")
ca <- atom.select(pdb, elety="CA")
coords <- pdb$xyz[ca$xyz]

calpha <- load.enmff(ff="calpha")
sdenm <- load.enmff(ff="sdenm")
pfanm <- load.enmff(ff="pfanm")

# Non-mass-weighted Hessians
hess.calpha <-build.hessian(coords, pfc.fun=calpha, pdb=pdb)
hess.sdenm <- build.hessian(coords, pfc.fun=sdenm, pdb=pdb)
hess.pfenm <- build.hessian(coords, pfc.fun=pfanm, pdb=pdb)

write.csv(hess.calpha,"./hessian_calpha_bio3d.csv", row.names = FALSE)
write.csv(hess.sdenm,"./hessian_sdenm_bio3d.csv", row.names = FALSE)
write.csv(hess.pfenm,"./hessian_pfenm_bio3d.csv", row.names = FALSE)

# Conduct NMA for 1l2y; bio3d masses and mass-weighted Eigenvalues as reference for tests;
# frequencies/fluctuations for comparisons 
# (whole set of non trivial modes as well as for n.-triv. modes 12-33)
# compute DCCs 
# (whole set of non trivial modes as well as for first 30 n.-triv. modes)
## Hinsen/Calpha
nma.calpha <- nma(pdb=pdb, ff="calpha", mass=TRUE)
# Write raw masses/eigenvalues
write.csv(nma.calpha$mass, "./1l2y_bio3d_masses.csv", row.names = FALSE)
write.csv(nma.calpha$L, "./mw_eigenvalues_calpha_bio3d.csv", row.names = FALSE)
# Frequencies
write.csv(nma.calpha$frequencies, "./mw_frequencies_calpha_bio3d.csv", row.names = FALSE)
# Fluctuations
write.csv(nma.calpha$fluctuations, "./mw_fluctuations_calpha_bio3d.csv", row.names = FALSE)
fluct.subset.calpha <- fluct.nma(nma.calpha, mode.inds=seq(12,33))
write.csv(fluct.subset.calpha, "./mw_fluctuations_calpha_subset_bio3d.csv", row.names = FALSE)
# DCC
dccm.calpha <- dccm(nma.calpha)
dccm.subset.calpha <- dccm(nma.calpha, nmodes=30)
write.csv(dccm.calpha,"./dccm_calpha_bio3d.csv", row.names = FALSE)
write.csv(dccm.subset.calpha,"./dccm_calpha_subset_bio3d.csv", row.names = FALSE)

## SDENM
nma.sdenm <- nma(pdb=pdb, ff="sdenm", mass=TRUE)
# Write raw masses/eigenvalues
write.csv(nma.sdenm$L, "./mw_eigenvalues_sdenm_bio3d.csv", row.names = FALSE)
# Frequencies
write.csv(nma.sdenm$frequencies, "./mw_frequencies_sdenm_bio3d.csv", row.names = FALSE)
#Fluctuations
write.csv(nma.sdenm$fluctuations, "./mw_fluctuations_sdenm_bio3d.csv", row.names = FALSE)
fluct.subset.sdenm <- fluct.nma(nma.sdenm, mode.inds=seq(12,33))
write.csv(fluct.subset.sdenm, "./mw_fluctuations_sdenm_subset_bio3d.csv", row.names = FALSE)
# DCC
dccm.sdenm <- dccm(nma.sdenm)
dccm.subset.sdenm <- dccm(nma.sdenm, nmodes=30)
write.csv(dccm.sdenm,"./dccm_sdenm_bio3d.csv", row.names = FALSE)
write.csv(dccm.subset.sdenm,"./dccm_sdenm_subset_bio3d.csv", row.names = FALSE)

## pfENM
nma.pfenm <- nma(pdb=pdb, ff="pfanm", mass=TRUE)
# Write raw masses/eigenvalues
write.csv(nma.pfenm$L, "./mw_eigenvalues_pfenm_bio3d.csv", row.names = FALSE)
# Frequencies
write.csv(nma.pfenm$frequencies, "./mw_frequencies_pfenm_bio3d.csv", row.names = FALSE)
# Fluctuations
write.csv(nma.pfenm$fluctuations, "./mw_fluctuations_pfenm_bio3d.csv", row.names = FALSE)
fluct.subset.pfenm <- fluct.nma(nma.pfenm, mode.inds=seq(12,33))
write.csv(fluct.subset.pfenm, "./mw_fluctuations_pfenm_subset_bio3d.csv", row.names = FALSE)
# DCC
dccm.pfenm <- dccm(nma.pfenm)
dccm.subset.pfenm <- dccm(nma.pfenm, nmodes=30)
write.csv(dccm.pfenm,"./dccm_pfenm_bio3d.csv", row.names = FALSE)
write.csv(dccm.subset.pfenm,"./dccm_pfenm_subset_bio3d.csv", row.names = FALSE)