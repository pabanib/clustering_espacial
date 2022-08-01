setwd("D:/Archivos/Codigos/clustering espacial")
library(spdep)
library(rgeoda)
library(digest)

covid <- st_read("SDEC/datos/datos/geo_prop.shp")
w_knn <- knn_weights(covid,6)

sk <- skater(7, w_knn, covid)
sk$`The ratio of between to total sum of squares`
sk$`Total within-cluster sum of squares`
sk$`Within-cluster sum of squares`
set.seed(213)


elegir_grupos <- function(datos, metodo, grupos){
  whithins <- numeric()
  for (i in 1:length(grupos)){
    m <- metodo(grupos[i],w_knn, datos)
    #m <- metodo(datos, grupos[i])
    whithins[i] <- mean(m$`Within-cluster sum of squares`)
  }
  return(whithins)}

grupos = 5:30
wss_sk <- elegir_grupos(covid, skater, grupos = grupos)

plot(grupos,wss_sk, type = "b", cex = 1, pch = 21, bg = "green", col = "red", lwd = 1)

wss_redcap <- elegir_grupos(covid, redcap, grupos)
plot(grupos,wss_redcap, type = "b", cex = 1, pch = 21, bg = "green", col = "red", lwd = 1)

wss_schc <- elegir_grupos(covid, schc, grupos)
plot(grupos,wss_schc, type = "b", cex = 1, pch = 21, bg = "green", col = "red", lwd = 1)

sk <- skater(10, w_knn, covid)
sk$`Total within-cluster sum of squares`
sk$`The ratio of between to total sum of squares`

redcap <- redcap(10,w_knn, covid)
redcap$`The ratio of between to total sum of squares`

sch <- schc(13, w_knn, covid)
sch$`The ratio of between to total sum of squares`

plot(st_geometry(covid), col = sk$Clusters)
plot(st_geometry(covid), col = redcap$Clusters+1)
plot(st_geometry(covid), col = sch$Clusters+1)

rdos <- matrix(nrow = 525)
rdos <- as.data.frame(rdos)
rdos['skater'] <- sk$Clusters
rdos['redcap'] <- redcap$Clusters
rdos['schc'] <- sch$Clusters
rdos <- rdos[,2:4]
