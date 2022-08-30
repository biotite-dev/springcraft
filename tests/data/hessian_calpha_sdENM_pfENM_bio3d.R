library(bio3d)

wd <- getwd()
print(wd)

pdb <- read.pdb("1l2y.pdb")
ca <- atom.select(pdb, elety="CA")
coords <- pdb$xyz[ca$xyz]

calpha <- load.enmff(ff="calpha")
sdenm <- load.enmff(ff="sdenm")
pfanm <- load.enmff(ff="pfanm")

hess.calpha <-build.hessian(coords, pfc.fun=calpha, pdb=pdb)
hess.sdenm <- build.hessian(coords, pfc.fun=sdenm, pdb=pdb)
hess.pfenm <- build.hessian(coords, pfc.fun=pfanm, pdb=pdb)

write.csv(hess.calpha,"./hessian_calpha_bio3d.csv", row.names = FALSE)
write.csv(hess.sdenm,"./hessian_sdenm_bio3d.csv", row.names = FALSE)
write.csv(hess.pfenm,"./hessian_pfenm_bio3d.csv", row.names = FALSE)

# Conduct NMA for 1l2y; bio3d masses and mass-weighted Eigenvalues as reference for tests;
# compare frequencies
nma.calpha <- nma(pdb=pdb, ff="calpha", mass=TRUE)
write.csv(nma.calpha$mass, "./1l2y_bio3d_masses.csv", row.names = FALSE)
write.csv(nma.calpha$L, "./mw_eigenvalues_calpha_bio3d.csv", row.names = FALSE)

# Conduct NMA for 1l2y; bio3d masses and mass-weighted Eigenvalues as reference for tests
nma.sdenm <- nma(pdb=pdb, ff="sdenm", mass=TRUE)
write.csv(nma.sdenm$L, "./mw_eigenvalues_sdenm_bio3d.csv", row.names = FALSE)