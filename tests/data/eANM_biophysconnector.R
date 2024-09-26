library(BioPhysConnectoR)

# Write compressed csv files
write_compressed_csv <- function(data, out_path, row_names = FALSE) {
    # Open gzfile
    gz_file <- gzfile(out_path, "w")
    write.csv(data, gz_file, row.names = row_names)
    close(gz_file)
}

# Get Hessian for eANM with biophysconnectoR for test purposes
# 1L2Y/7CAL as test peptide/multi-chain protein
pdb_files <- c("1l2y.pdb", "7cal.pdb")

# MJ/Keskin parametersets
mj <- as.matrix(read.table(system.file("extdata", "mj1.txt", 
                                        package = "LRTNullModel4")))
ke <- as.matrix(read.table(system.file("extdata", "mj2.txt", 
                                        package = "LRTNullModel4")))

base_out_filename <- "./biophysconnector_anm_%s_%s_%s.csv.gz"
for (pdb_file in pdb_files) {
    pdb <- extractPDB(pdb_file)
    # [[]] for list elements instead of [] for sublists...
    pdb_name <- strsplit(pdb_file, "\\.")[[1]][1]
    print(pdb_name)
    contacts <- build.contacts(length(pdb$caseq), 13^2, pdb$coords)

    ## "Original" parametrization for eANMs
    # MJ for intrachain-, Keskin for interchain-contacts
    intmat <- build.interact(pdb$caseq, mj1 = mj, mj2 = ke, d = pdb$chains, 
                             alpha = 82)

    hessian <- build.hess(cm=contacts$cm, im=intmat, deltas=contacts$deltas)

    ## MJ + Keskin (standard eANM)
    # Export as .csv
    hess_outname <- sprintf(base_out_filename, "eanm", "hessian", pdb_name)
    print(hess_outname)
    write_compressed_csv(hessian, hess_outname)

    # Compute & export eigenvalues as .csv
    eigen <- get.svd(hessian)
    evals_outname <- sprintf(base_out_filename, "eanm", "evals", pdb_name)
    print(evals_outname)
    write_compressed_csv(eigen$ev, evals_outname)

    # Compute the (predicted) anisotropic B-factors using the covariance matrix
    covmat <- get.cov(contacts$cm, intmat, contacts$deltas)
    bfacs <- get.bfacs(covmat)

    # Export as .csv
    bfacs_outname <- sprintf(base_out_filename, "eanm", "bfacs", pdb_name)
    print(bfacs_outname)
    write_compressed_csv(bfacs, bfacs_outname)

    ## MJ for non-bonded interactions
    intmat_mj <- build.interact(pdb$caseq, mj1 = mj, mj2 = mj, d = pdb$chains, 
                             alpha = 82)

    hessian_mj <- build.hess(cm=contacts$cm, im=intmat_mj, deltas=contacts$deltas)

    # Export as .csv
    hess_outname_mj <- sprintf(base_out_filename, "eanm_mj", "hessian", pdb_name)
    print(hess_outname_mj)
    write_compressed_csv(hessian_mj, hess_outname_mj)

    ## Keskin for non-bonded interactions
    intmat_ke <- build.interact(pdb$caseq, mj1 = ke, mj2 = ke, d = pdb$chains, 
                                alpha = 82)

    hessian_ke <- build.hess(cm=contacts$cm, im=intmat_ke, deltas=contacts$deltas)

    # Export as .csv
    hess_outname_keskin <- sprintf(base_out_filename, "eanm_ke", "hessian", pdb_name)
    print(hess_outname_keskin)
    write_compressed_csv(hessian_ke, hess_outname_keskin)
}