setwd("D:/Archivos/Codigos/clustering espacial")
library(spdep)
library(rgeoda)
library(digest)

rdos <- matrix(nrow = 525)
rdos <- as.data.frame(rdos)
sim2 <- st_read("SDEC/datos/simulaciones/sim2.shp")
w_knn <- knn_weights(sim2,6)
sk <- rgeoda::skater(k = 6, w = w_knn, df = sim2)
rdos['skater'] <- sk$Clusters
redcap <- rgeoda::redcap(k = 6, w = w_knn, df = sim2)
rdos['redcap'] <- redcap$Clusters
schc <- rgeoda::schc(k = 6, w = w_knn, df = sim2)
rdos['schc'] <- schc$Clusters
rdos <- rdos[,2:4]
