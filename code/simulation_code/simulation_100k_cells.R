rm(list = ls())
library(splatter)
library(rhdf5)

### simulate 100k cells
simulate <- function(nGroups=10, nGenes=3000, batchCells=10^5, dropout=0)
{
  if (nGroups > 1) method <- 'groups'
  else             method <- 'single'
  
  group.prob <- rep(1, nGroups) / nGroups
  sim <- splatSimulate(group.prob=group.prob, nGenes=nGenes, batchCells=batchCells,
                       dropout.type="experiment", method=method,
                       seed=100, dropout.shape=-1, dropout.mid=dropout)
  
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

counts1 <- counts
counts1$group <- cellinfo$Group



dropout.rate <- (sum(counts==0)-sum(truecounts==0))/sum(truecounts>0)
print(dropout.rate)
save(counts, geneinfo, cellinfo, truecounts, dropout.rate, file="splatter_simulate_data_100k.RData")

X <- t(counts)
Y <- as.integer(substring(cellinfo$Group,6))
Y <- Y-1

### randomly downsample from 1k cells to 100k cells
set.seed(0)
idx1 <- sample(10^5, 1000)
X1 <- X[, idx1]
Y1 <- Y[idx1]
h5createFile("splatter_simulate_data_1k.h5")
h5write(X1, "splatter_simulate_data_1k.h5", "X")
h5write(Y1, "splatter_simulate_data_1k.h5", "Y")

set.seed(0)
idx2 <- sample(10^5, 5000)
X2 <- X[, idx2]
Y2 <- Y[idx2]
h5createFile("splatter_simulate_data_5k.h5")
h5write(X2, "splatter_simulate_data_5k.h5", "X")
h5write(Y2, "splatter_simulate_data_5k.h5", "Y")

set.seed(0)
idx3 <- sample(10^5, 1e4)
X3 <- X[, idx3]
Y3 <- Y[idx3]
h5createFile("splatter_simulate_data_10k.h5")
h5write(X3, "splatter_simulate_data_10k.h5", "X")
h5write(Y3, "splatter_simulate_data_10k.h5", "Y")

set.seed(0)
idx4 <- sample(10^5, 5e4)
X4 <- X[, idx4]
Y4 <- Y[idx4]
h5createFile("splatter_simulate_data_50k.h5")
h5write(X4, "splatter_simulate_data_50k.h5", "X")
h5write(Y4, "splatter_simulate_data_50k.h5", "Y")

set.seed(0)
idx5 <- sample(10^5, 25000)
X5 <- X[, idx5]
Y5 <- Y[idx5]
h5createFile("splatter_simulate_data_25k.h5")
h5write(X5, "splatter_simulate_data_25k.h5", "X")
h5write(Y5, "splatter_simulate_data_25k.h5", "Y")

set.seed(0)
idx6 <- sample(10^5, 2500)
X6 <- X[, idx6]
Y6 <- Y[idx6]
h5createFile("splatter_simulate_data_2_5k.h5")
h5write(X6, "splatter_simulate_data_2_5k.h5", "X")
h5write(Y6, "splatter_simulate_data_2_5k.h5", "Y")

set.seed(0)
idx7 <- sample(10^5, 7500)
X7 <- X[, idx7]
Y7 <- Y[idx7]
h5createFile("splatter_simulate_data_7_5k.h5")
h5write(X7, "splatter_simulate_data_7_5k.h5", "X")
h5write(Y7, "splatter_simulate_data_7_5k.h5", "Y")

set.seed(0)
idx8 <- sample(10^5, 75000)
X8 <- X[, idx8]
Y8 <- Y[idx8]
h5createFile("splatter_simulate_data_75k.h5")
h5write(X8, "splatter_simulate_data_75k.h5", "X")
h5write(Y8, "splatter_simulate_data_75k.h5", "Y")




X <- t(truecounts)
h5createFile("splatter_simulate_truedata_100k.h5")
h5write(X, "splatter_simulate_truedata_100k.h5", "X")
h5write(Y, "splatter_simulate_truedata_100k.h5", "Y")

