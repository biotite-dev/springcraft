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
nma.calpha <- nma(pdb=pdb, ff="calpha", mass=TRUE)
write.csv(nma.calpha$mass, "./1l2y_bio3d_masses.csv", row.names = FALSE)
write.csv(nma.calpha$L, "./mw_eigenvalues_calpha_bio3d.csv", row.names = FALSE)
write.csv(nma.calpha$frequencies, "./mw_frequencies_calpha_bio3d.csv", row.names = FALSE)
write.csv(nma.calpha$fluctuations, "./mw_fluctuations_calpha_bio3d.csv", row.names = FALSE)
# Fluctuations computed for modes 12-78
fluct.subset.calpha <- fluct.nma(nma.calpha, mode.inds=seq(12,33))
write.csv(fluct.subset.calpha, "./mw_fluctuations_calpha_subset_bio3d.csv", row.names = FALSE)

# Conduct NMA for 1l2y; bio3d masses and mass-weighted Eigenvalues as reference for tests;
# frequencies/fluctuations for comparisons
nma.sdenm <- nma(pdb=pdb, ff="sdenm", mass=TRUE)
write.csv(nma.sdenm$L, "./mw_eigenvalues_sdenm_bio3d.csv", row.names = FALSE)
write.csv(nma.sdenm$frequencies, "./mw_frequencies_sdenm_bio3d.csv", row.names = FALSE)
write.csv(nma.sdenm$fluctuations, "./mw_fluctuations_sdenm_bio3d.csv", row.names = FALSE)
fluct.subset.sdenm <- fluct.nma(nma.sdenm, mode.inds=seq(12,33))
write.csv(fluct.subset.sdenm, "./mw_fluctuations_sdenm_subset_bio3d.csv", row.names = FALSE)

# Frequencies and fluctuations for pfENM
nma.pfenm <- nma(pdb=pdb, ff="pfanm", mass=TRUE)
write.csv(nma.pfenm$L, "./mw_eigenvalues_pfenm_bio3d.csv", row.names = FALSE)
write.csv(nma.pfenm$frequencies, "./mw_frequencies_pfenm_bio3d.csv", row.names = FALSE)
write.csv(nma.pfenm$fluctuations, "./mw_fluctuations_pfenm_bio3d.csv", row.names = FALSE)
fluct.subset.pfenm <- fluct.nma(nma.pfenm, mode.inds=seq(12,33))
write.csv(fluct.subset.pfenm, "./mw_fluctuations_pfenm_subset_bio3d.csv", row.names = FALSE)