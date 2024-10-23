            rm(list = ls())
            source('CVASL_RELIEF.R')            
            library(MASS)
            library(Matrix)
            options(repos = c(CRAN = "https://cran.r-project.org"))
            install.packages("denoiseR", dependencies = TRUE, quiet = TRUE)
            library(denoiseR)
            install.packages("RcppCNPy", dependencies = TRUE, quiet = TRUE)
            library(RcppCNPy)
            data5 <- npyLoad("dat_var_for_RELIEF5.npy")
            covars5 <- read.csv('bath_and_mod_forRELIEF5.csv')
            covars_only5  <- covars5[,-(1:2)]   
            covars_only_matrix5 <-data.matrix(covars_only5)
            relief.harmonized = relief(
                dat=data5,
                batch=covars5$batch,
                mod=covars_only_matrix5
            )
            outcomes_harmonized5 <- relief.harmonized$dat.relief
            write.csv(outcomes_harmonized5, "relief1_for5_results.csv")

