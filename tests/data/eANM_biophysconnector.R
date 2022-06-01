library(BioPhysConnectoR)

# Get Hessian for eANM with biophysconnectoR for test purposes
# 1L2Y as test peptide/protein

pdb <- extractPDB("1l2y.pdb")

contacts <- build.contacts(length(pdb$caseq), 13^2, pdb$coords)

# MJ for intrachain-, Keskin for interchain-contacts
mj1 <- as.matrix(read.table(system.file("extdata", "mj1.txt", 
                                        package = "LRTNullModel4")))
mj2 <- as.matrix(read.table(system.file("extdata", "mj2.txt", 
                                        package = "LRTNullModel4")))


intmat <- build.interact(pdb$caseq, mj1 = mj1, mj2 = mj2, d = pdb$chains, 
                         alpha = 82)

hessian <- build.hess(cm=contacts$cm, im=intmat, deltas=contacts$deltas)

print(hessian)

# Export as .csv
write.csv(hessian,"./hessian_eANM_BioPhysConnectoR.csv", row.names = FALSE)
