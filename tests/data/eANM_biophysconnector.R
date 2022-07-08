library(BioPhysConnectoR)

# Get Hessian for eANM with biophysconnectoR for test purposes
# 1L2Y as test peptide/protein

pdb <- extractPDB("1l2y.pdb")

contacts <- build.contacts(length(pdb$caseq), 13^2, pdb$coords)

# MJ/Keskin parametersets
mj <- as.matrix(read.table(system.file("extdata", "mj1.txt", 
                                        package = "LRTNullModel4")))
ke <- as.matrix(read.table(system.file("extdata", "mj2.txt", 
                                        package = "LRTNullModel4")))


## "Original" parametrization for eANMs
# MJ for intrachain-, Keskin for interchain-contacts
intmat <- build.interact(pdb$caseq, mj1 = mj, mj2 = ke, d = pdb$chains, 
                         alpha = 82)

hessian <- build.hess(cm=contacts$cm, im=intmat, deltas=contacts$deltas)

print(hessian)

# Export as .csv
write.csv(hessian,"./hessian_eANM_BioPhysConnectoR.csv", row.names = FALSE)


## MJ for non-bonded interactions
intmat_mj <- build.interact(pdb$caseq, mj1 = mj, mj2 = mj, d = pdb$chains, 
                         alpha = 82)

hessian_mj <- build.hess(cm=contacts$cm, im=intmat_mj, deltas=contacts$deltas)

print(hessian_mj)

# Export as .csv
write.csv(hessian_mj,"./hessian_eANM_mj_BioPhysConnectoR.csv", row.names = FALSE)

## MJ for non-bonded interactions
intmat_ke <- build.interact(pdb$caseq, mj1 = ke, mj2 = ke, d = pdb$chains, 
                            alpha = 82)

hessian_ke <- build.hess(cm=contacts$cm, im=intmat_ke, deltas=contacts$deltas)

print(hessian_ke)

# Export as .csv
write.csv(hessian_ke,"./hessian_eANM_ke_BioPhysConnectoR.csv", row.names = FALSE)