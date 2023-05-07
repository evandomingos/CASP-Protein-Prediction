dataset <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv", header = TRUE, stringsAsFactors = FALSE)

library(ggplot2)
#remove missing data (none in CASP dataset), create protein set
protein <- na.omit(dataset)
names(protein)



# Load the protein dataset (assuming it's already imported)


vars <- names(protein)
combinations <- combn(vars, 2)
#scatter plot each variables
par(mfrow = c(1, 1)) 
par(ask = TRUE)
for (i in 1:ncol(combinations)) {
  plot(protein[, combinations[1, i]], protein[, combinations[2, i]],
       xlab = combinations[1, i], ylab = combinations[2, i],
       main = paste0("Scatter plot of ", combinations[1, i], " vs. ", combinations[2, i]))
}

# sort by increasing RMSD values
protein_sorted <- protein[order(protein$RMSD),]
# plot RMSD in increasing order
plot(protein_sorted$RMSD, type = "l", xlab = "Residue", ylab = "RMSD", main = "RMSD Sorted in Increasing Order")
abline(a = 0, b = 20.99/45730, col = "purple")
