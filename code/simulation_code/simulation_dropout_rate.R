#### simulate scRNA-seq data with various dropout rates ####

rm(list = ls())
library(splatter)
library(rhdf5)

dropout.rate <- c()
for(i in 1:20) {
  simulate <- function(nGroups=3, nGenes=2500, batchCells=1500, dropout=0) # change dropout to simulate various dropout rates
  {
    if (nGroups > 1) method <- 'groups'
    else             method <- 'single'
    
    group.prob <- rep(1, nGroups) / nGroups
    sim <- splatSimulate(group.prob=group.prob, nGenes=nGenes, batchCells=batchCells,
                         dropout.type="experiment", method=method,
                         seed=100+i, dropout.shape=-1, dropout.mid=dropout)
    
    counts     <- as.data.frame(t(counts(sim)))
    truecounts <- as.data.frame(t(assays(sim)$TrueCounts))
    
    dropout    <- assays(sim)$Dropout
    mode(dropout) <- 'integer'
    
    cellinfo   <- as.data.frame(colData(sim))
    geneinfo   <- as.data.frame(rowData(sim))
    
    list(sim=sim,
         counts=counts,
         cellinfo=cellinfo,
         geneinfo=geneinfo,
         truecounts=truecounts)
  }
  
  sim <- simulate()
  
  simulation <- sim$sim
  counts <- sim$counts
  geneinfo <- sim$geneinfo
  cellinfo <- sim$cellinfo
  truecounts <- sim$truecounts
  
  dropout.rate <- c(dropout.rate, (sum(counts==0)-sum(truecounts==0))/sum(truecounts>0))
  save(counts, geneinfo, cellinfo, truecounts, file=paste("splatter_simulate_data_", i, ".RData", sep=""))
  
  X <- t(counts)
  Y <- as.integer(substring(cellinfo$Group,6))
  Y <- Y-1
  
  h5createFile(paste("./simulation_raw_data/splatter_simulate_data_", i, ".h5", sep=""))
  h5write(X, paste("./simulation_raw_data/splatter_simulate_data_", i, ".h5", sep=""),"X")
  h5write(Y, paste("./simulation_raw_data/splatter_simulate_data_", i, ".h5", sep=""),"Y")
  
  X <- t(truecounts)
  h5createFile(paste("./simulation_true_counts/splatter_simulate_data_", i, ".h5", sep=""))
  h5write(X, paste("./simulation_true_counts/splatter_simulate_data_", i, ".h5", sep=""),"X")
  h5write(Y, paste("./simulation_true_counts/splatter_simulate_data_", i, ".h5", sep=""),"Y")
}

dropout.rate <- data.frame(dropout.rate=dropout.rate)
write.csv(dropout.rate, "simulation_dropout_rate.csv", row.names = F)
