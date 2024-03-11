# install.packages("factoextra")
# install.packages("ggfortify")
# install.packages("tsne")
# install.packages("umap")

suppressPackageStartupMessages(suppressWarnings({
  library(factoextra)
  library(dplyr)
  library(ggfortify)
  library(tsne)
  library(plotly)
  library(umap)
  library(matrixStats)
  library(reshape)
}))

setwd("/home/data/Git_RNA-Seq-Data-Analysis/Assignment5/")
Expression_Data = read.csv("Expression.txt", header = T, sep = "\t", row.names = 1)

# K-Means clustering
kmeans = kmeans(Expression_Data, centers = 3)
Expression_Data$clusters = kmeans$cluster

options(repr.plot.width = 10, repr.plot.height = 8)
fviz_cluster(kmeans, data = Expression_Data,
             palette = c("#2E9FDF", "#FF0000", "#E7B800"), 
             geom = "point", main = "K-Means Clustering",
             ellipse.type = "convex", pointsize = 3, labelsize = 30,
             ggtheme = theme_bw() +
               theme(axis.text = element_text(size = 14, , face = "bold"),   # Adjust size of axis labels
                     axis.title = element_text(size = 18),  # Adjust size of axis titles
                     plot.title = element_text(size = 20, face = "bold"), # Adjust size of plot title
                     legend.text = element_text(size = 16),  # Adjust size of legend text
                     legend.title = element_text(size = 18)) # Adjust size of legend title
             ) 

#Hierarchical clustering

distance = dist(Expression_Data[,c(1:10)], method = "manhattan")
clusters = hclust(distance, method = "average")

# Assign colors to clusters
cluster_colors <- rainbow(length(unique(cutree(clusters, k = 3))))

# Plot dendrogram
#par(mar = c(5, 6, 4, 2) + 0.1) # Adjust margin for better visualization
options(repr.plot.width = 15, repr.plot.height = 6)
plot(clusters, labels = Expression_Data$Gene,
     main = "Hierarchical Clustering Dendrogram", 
     xlab = "Genes", ylab = "Distance", 
     cex.main = 1.5, cex.lab = 1.5, cex.sub = 1.2, cex = 0.8)

rect.hclust(clusters, k = 3, border = 2:6)  # Add colored rectangles for clusters

#Time point regression

background = filter(Expression_Data , Expression_Data$clusters == 1)
increasing = filter(Expression_Data, Expression_Data$clusters == 3)
decreasing = filter(Expression_Data, Expression_Data$clusters == 2)

background_expres = t(background[,c(1:10)])
increasing_expres = t(increasing[,c(1:10)])
decreasing_expres = t(decreasing[,c(1:10)])

background_slopes = c()
increasing_slopes = c()
decreasing_slopes = c()

time_point = c(1:10)
my_rownames = c(1:12)

for (i in my_rownames){
x = increasing_expres[,i]
a = lm(time_point ~ x )
required = a$coefficients[2]
increasing_slopes = append(increasing_slopes, required)
}

for (i in my_rownames){
x = decreasing_expres[,i]
a = lm(time_point ~ x )
required = a$coefficients[2]
decreasing_slopes = append(decreasing_slopes, required)
}

for (i in my_rownames){
x = background_expres[,i]
a = lm(time_point ~ x )
required = a$coefficients[2]
background_slopes = append(background_slopes, required)
}

print(paste("Mean Slope for the Background Cluster is: " , round(mean(background_slopes), 2))) # = -0.0112971
print(paste("Mean Slope for the Increasing Cluster is: " , round(mean(increasing_slopes),3))) # = 0.1001658
print(paste("Mean Slope for the Decreasing Cluster is: " , round(mean(decreasing_slopes),3))) # = -0.09934661
print(paste("The Background Expression Level is: " , mean(rowMeans(background_expres)))) # = 55.00665

# Selecting the first 10 columns from Expression_Data
X <- Expression_Data[, 1:10]
# Extracting clusters from Expression_Data
clusters <- Expression_Data$clusters

# Standard scaling
X_scaled <- scale(X)

# PCA
pca <- prcomp(X_scaled)

# Extracting PC scores
X_pca <- pca$x

# Unique clusters
unique_clusters <- unique(clusters)

# Plotting PCA
options(repr.plot.width = 10, repr.plot.height = 8)

plot(X_pca[, 1], X_pca[, 2], type = "n",
     xlab = "Principal Component 1",
     ylab = "Principal Component 2",
     main = "Principal Component Analysis",
     xlim = range(X_pca[, 1]), ylim = range(X_pca[, 2]),
     cex.main = 1.5, cex.lab = 1.5, cex.sub = 1.2, cex = 0.8)

for (cluster in unique_clusters) {
    points(X_pca[clusters == cluster, 1],
           X_pca[clusters == cluster, 2],
           col = cluster, pch = 19)
}
legend("topright", legend = paste("Cluster", unique_clusters),
       col = unique_clusters, pch = 19, cex = 0.8, bty = "n")
grid()

## Heatmap
Expression_data2 = Expression_Data[order(Expression_Data$clusters),]
options(repr.plot.width = 14, repr.plot.height = 10)
heatmap(as.matrix(Expression_data2[,1:10]),
        Colv = NA, cexCol = 1.5, cexRow = 1.2, main = "Heatmap") 

## T-SNE

set.seed(0)
tsne <- tsne(Expression_data2[,1:10], initial_dims = 2)

tsne = data.frame(tsne)
pdb = cbind(tsne, Expression_data2$clusters)

options(repr.plot.width = 10, repr.plot.height = 8)
fig = plot_ly(data = pdb, x = ~X1, y = ~X2, type = "scatter",
              mode = 'markers', split = ~ Expression_data2$clusters)
fig

## UMAP

data.umap = umap(Expression_Data[,1:10], n_components = 3, random_state = 3)
layout = data.umap[["layout"]]
layout = data.frame(layout)
final = cbind(layout, Expression_Data$clusters)

options(repr.plot.width = 8, repr.plot.height = 8)
fig3 = plot_ly(final, x = ~X1, y = ~X2, 
               mode = 'markers', type = "scatter",split = ~ Expression_Data$clusters )
fig3

data0 <- read.table("data1.txt", header = FALSE)
data1 <- as.vector(data0$V1)

tables_list = list()
sample_size = 500

# Sample and calculate tables
for (i in 1:10) {
  table_data <- table(sample(data1, sample_size, replace = TRUE))
  tables_list[[i]] <- as.data.frame(table_data)
}
     
Q2_1 <- suppressWarnings(Reduce(function(x, y) merge(x, y, by = "Var1", all = TRUE), tables_list))
rownames(Q2_1) = Q2_1[,1]
row_names_numeric <- as.numeric(rownames(Q2_1))

Q2_1 = Q2_1[,-1]
colnames(Q2_1) = c(1:10)
Q2_1[is.na(Q2_1)] = 0
                       
# Calculate total sum
total_sum <- sum(rowSums(Q2_1))

# Compute mean, variance, and abundance
Q2_1$mean <- rowMeans(Q2_1[,1:10])
Q2_1$variance <- round(rowVars(as.matrix(Q2_1[,1:10])), 4)
Q2_1$abundance <- rowSums(Q2_1[,1:10]) / total_sum
Q2_1 <- Q2_1[order(row_names_numeric), ]
Q2_1
                                
                                
options(repr.plot.width = 10, repr.plot.height = 8)

plot(x = Q2_1$mean, y = Q2_1$variance,
     xlab="Mean of gene", ylab="Variance of gene", 
     main="Mean vs Variance for 500 rows sampled",
     col = "blue", pch = 16, cex = 1.5,
     cex.main = 2.0, cex.lab = 1.5, cex.axis = 1.5)

lines(x = c(0:150), y = c(0:150), col = "red")
abline(lm(Q2_1$variance~Q2_1$mean, data = Q2_1), col = "green")

legend("bottomright",c("Poisson mean = variance", "linear fit regression line", "Mean Vs Variance"),
       col=c("red","green", "blue"), lwd= c(2,2,NA), lty= c(2,1,NA), pch = c(NA,NA,16),cex = 1.3)

data0 <- read.table("data1.txt", header = FALSE)
data1 <- as.vector(data0$V1)

tables_list = list()
sample_size = 5000

# Sample and calculate tables
for (i in 1:10) {
  table_data <- table(sample(data1, sample_size, replace = TRUE))
  tables_list[[i]] <- as.data.frame(table_data)
}
     
Q2_1 <- suppressWarnings(Reduce(function(x, y) merge(x, y, by = "Var1", all = TRUE), tables_list))
rownames(Q2_1) = Q2_1[,1]
row_names_numeric <- as.numeric(rownames(Q2_1))

Q2_1 = Q2_1[,-1]
colnames(Q2_1) = c(1:10)
Q2_1[is.na(Q2_1)] = 0
                       
# Calculate total sum
total_sum <- sum(rowSums(Q2_1))

# Compute mean, variance, and abundance
Q2_1$mean <- rowMeans(Q2_1[,1:10])
Q2_1$variance <- round(rowVars(as.matrix(Q2_1[,1:10])), 4)
Q2_1$abundance <- rowSums(Q2_1[,1:10]) / total_sum
Q2_1 <- Q2_1[order(row_names_numeric), ]
Q2_1
                                
                                
options(repr.plot.width = 10, repr.plot.height = 8)

plot(x = Q2_1$mean, y = Q2_1$variance,
     xlab="Mean of gene", ylab="Variance of gene", 
     main="Mean vs Variance for 5000 rows sampled",
     col = "blue", pch = 16, cex = 1.5,
     cex.main = 2.0, cex.lab = 1.5, cex.axis = 1.5)

lines(x = c(0:1500), y = c(0:1500), col = "red")
abline(lm(Q2_1$variance~Q2_1$mean, data = Q2_1), col = "green")

legend("bottomright",c("Poisson mean = variance", "linear fit regression line", "Mean Vs Variance"),
       col=c("red","green", "blue"), lwd= c(2,2,NA), lty= c(2,1,NA), pch = c(NA,NA,16),cex = 1.3)

data0 <- read.table("data1.txt", header = FALSE)
data1 <- as.vector(data0$V1)

tables_list = list()
sample_size = 50000

# Sample and calculate tables
for (i in 1:10) {
  table_data <- table(sample(data1, sample_size, replace = TRUE))
  tables_list[[i]] <- as.data.frame(table_data)
}
     
Q2_1 <- suppressWarnings(Reduce(function(x, y) merge(x, y, by = "Var1", all = TRUE), tables_list))
rownames(Q2_1) = Q2_1[,1]
row_names_numeric <- as.numeric(rownames(Q2_1))

Q2_1 = Q2_1[,-1]
colnames(Q2_1) = c(1:10)
Q2_1[is.na(Q2_1)] = 0
                       
# Calculate total sum
total_sum <- sum(rowSums(Q2_1))

# Compute mean, variance, and abundance
Q2_1$mean <- rowMeans(Q2_1[,1:10])
Q2_1$variance <- round(rowVars(as.matrix(Q2_1[,1:10])), 4)
Q2_1$abundance <- rowSums(Q2_1[,1:10]) / total_sum
Q2_1 <- Q2_1[order(row_names_numeric), ]
Q2_1
                                
                                
options(repr.plot.width = 10, repr.plot.height = 8)

plot(x = Q2_1$mean, y = Q2_1$variance,
     xlab="Mean of gene", ylab="Variance of gene", 
     main="Mean vs Variance for 50000 rows sampled",
     col = "blue", pch = 16, cex = 1.5,
     cex.main = 2.0, cex.lab = 1.5, cex.axis = 1.5)

lines(x = c(0:15000), y = c(0:15000), col = "red")
abline(lm(Q2_1$variance~Q2_1$mean, data = Q2_1), col = "green")

legend("bottomright",c("Poisson mean = variance", "linear fit regression line", "Mean Vs Variance"),
       col=c("red","green", "blue"), lwd= c(2,2,NA), lty= c(2,1,NA), pch = c(NA,NA,16),cex = 1.3)

#data0 = read.table("data1.txt", header = F)
data1 = as.vector(read.table("data1.txt", header = F)$V1)
data2 = as.vector(read.table("data2.txt", header = F)$V1)

generate_and_merge_tables <- function(data, sample_size, prefix) {
  tables_list <- lapply(1:3, function(i) {
    table_data <- table(sample(data, sample_size, replace = TRUE))
    as.data.frame(table_data)
  })
  
  merged_table <- Reduce(function(x, y) merge(x, y, by = "Var1", all = TRUE), tables_list)
  rownames(merged_table) <- merged_table[, 1]
  row_names_numeric <- as.numeric(rownames(Q2_1))
  merged_table <- merged_table[, -1]
  colnames(merged_table) <- paste0(prefix, letters[1:3])  # Modify column names here
  merged_table[is.na(merged_table)] <- 0
  
  return(merged_table)
}

# Generate and merge tables for data1
Q3a_1 <- generate_and_merge_tables(data1, sample_size, prefix = "1")

# Generate and merge tables for data2
Q3a_2 <- generate_and_merge_tables(data2, sample_size, prefix = "2")

Q3a = merge(Q3a_1, Q3a_2, by = 'row.names')
rownames(Q3a) = Q3a[,1]
Q3a = Q3a[,-1]
Q3a <- Q3a[order(row_names_numeric), ]

# Calculate mean and log2 fold change
Q3a$data1_mean <- rowMeans(Q3a[, 1:3])
Q3a$data2_mean <- rowMeans(Q3a[, 4:6])
Q3a$log2_foldchange <- (log2(Q3a$data2_mean) - log2(Q3a$data1_mean)) / log2(Q3a$data1_mean)

# Perform t-test and calculate p-values
t.test.all.genes <- function(x, s1, s2) {
  x1 <- as.numeric(x[s1])
  x2 <- as.numeric(x[s2])
  t.out <- t.test(x1, x2, alternative = "two.sided", paired = TRUE, conf.level = 0.95)
  return(as.numeric(t.out$p.value))
}
gene_p_values <- round(apply(Q3a, 1, t.test.all.genes, s1 = 1:3, s2 = 4:6), 5)
Q3a <- cbind(Q3a, gene_p_values)
Q3a$significant <- Q3a$gene_p_values < 0.05
Q3a$log_10_negative <- -log10(Q3a$gene_p_values)
                
Q3a


# Plot
plot(y = Q3a$log_10_negative, x = Q3a$log2_foldchange,
     col = "red", pch = 16, main = "Volcano Plot",
     xlab = "Log2 Fold Change", ylab = "-log10(p-value)",
    cex.main = 2.0, cex.lab = 1.5, cex.axis = 1.5, cex = 1.5)

abline(v = c(-0.1, 0.1), col = "blue")
abline(h = -log10(0.05), col = "green")
text(x = Q3a$log2_foldchange, y = Q3a$log_10_negative, labels = rownames(Q3a), pos = 1)

generate_and_merge_tables <- function(data, sample_size, prefix) {
  tables_list <- lapply(1:10, function(i) {
    table_data <- table(sample(data, sample_size, replace = TRUE))
    as.data.frame(table_data)
  })
  
  merged_table <- suppressWarnings(Reduce(function(x, y) merge(x, y, by = "Var1", all = TRUE), tables_list))
  rownames(merged_table) <- merged_table[, 1]
  row_names_numeric <- as.numeric(rownames(Q2_1))
  merged_table <- merged_table[, -1]
  colnames(merged_table) <- paste0(prefix, letters[1:10])  # Modify column names here
  merged_table[is.na(merged_table)] <- 0
  
  return(merged_table)
}

# Generate and merge tables for data1
Q3c_1 <- generate_and_merge_tables(data1, sample_size, prefix = "1")

# Generate and merge tables for data2
Q3c_2 <- generate_and_merge_tables(data2, sample_size, prefix = "2")

Q3c = merge(Q3c_1, Q3c_2, by = 'row.names')
rownames(Q3c) = Q3c[,1]
Q3c = Q3c[,-1]
Q3c <- Q3c[order(row_names_numeric), ]
     
# Calculate mean and log2 fold change
Q3c$data1_mean <- rowMeans(Q3c[, 1:10])
Q3c$data2_mean <- rowMeans(Q3c[, 11:20])
Q3c$log2_foldchange <- (log2(Q3c$data2_mean) - log2(Q3c$data1_mean)) / log2(Q3c$data1_mean)

# Perform t-test and calculate p-values
t.test.all.genes <- function(x, s1, s2) {
  x1 <- as.numeric(x[s1])
  x2 <- as.numeric(x[s2])
  t.out <- t.test(x1, x2, alternative = "two.sided", paired = TRUE, conf.level = 0.95)
  return(as.numeric(t.out$p.value))
}
gene_p_values <- round(apply(Q3c, 1, t.test.all.genes, s1 = 1:10, s2 = 11:20), 5)
Q3c <- cbind(Q3c, gene_p_values)
Q3c$significant <- Q3c$gene_p_values < 0.05
Q3c$log_10_negative <- -log10(Q3c$gene_p_values)

Q3c

# Plot
plot(y = Q3c$log_10_negative, x = Q3c$log2_foldchange,
     col = "red", pch = 16, main = "Volcano Plot",
     xlab = "Log2 Fold Change", ylab = "-log10(p-value)",
    cex.main = 2.0, cex.lab = 1.5, cex.axis = 1.5, cex = 1.5)

abline(v = c(-0.1, 0.1), col = "blue")
abline(h = -log10(0.05), col = "green")
text(x = Q3c$log2_foldchange, y = Q3c$log_10_negative, labels = rownames(Q3c), pos = 1)


