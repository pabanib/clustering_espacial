setwd("D:/Archivos/Codigos/clustering espacial")

#install.packages("keras")
#install.packages("tensorflow")

library(keras)
library(tensorflow)
library(spdep)
library(rgeoda)
library(digest)
#library(tensorflow)
#install_keras(tensorflow = "gpu")
#install_keras()

sim2 <- st_read("SDEC/datos/simulaciones/sim2.shp")
#sim2 <- as.data.frame(sim2)
#val1 <- sim2[,1:100]
#val2 <- sim2[,100:200]


#mod <- keras_model_sequential()
inp1 <- layer_input(shape = c(100))
inp2 <- layer_input(shape = c(100))
dense1 <- layer_dense(units = 80, activation = "relu")(inp1)
dense2 <- layer_dense(units = 80, activation = "relu")(inp2)
concat <- layer_concatenate(c(dense1,dense2))
concat <- layer_dense(units = 50, activation = "relu")(concat)
enco <- layer_dense(units = 10,activation = "relu")(concat)
dense <- layer_dense(units = 80, activation = "relu", kernel_regularizer = regularizer_l1(0.01))(enco)
decoder <- layer_dense(units = 200, activation = "relu", kernel_regularizer =  regularizer_l1(0.01))(dense)


autoencoder <- keras_model(inputs = c(inp1,inp2), outputs = decoder)
encoder <- keras_model(inputs = c(inp1,inp2), outputs = enco)

autoencoder %>% compile(optimizer = "adam", loss = "cosine_similarity")



val1 <- as.matrix(val1)
val2 <- as.matrix(val2)

list(val1,val2)

val1 <- runif(500*100)
val1 <- array(val1, c(500,100))
val2 <- runif(500*100)
val2 <- array(val2, c(500, 100))

mnist <- keras::dataset_mnist()

val <- runif(500*200)
val <- array(val, c(500,200))
val1 <- val[,1:100]
val2 <- val[,101:200]

autoencoder %>% fit(list(val1,val2), val, epochs = 20, validation_split = 0.2)

clustering <- R6::R6Class("ClusteringLayer",
                            
                            inherit = KerasLayer,
                            
                            public = list(
                              
                              n_clusters = NULL,
                              weights = NULL,
                              alpha = 1.0,
                              
                              initialize = function(n_cluster, weights = NULL, alpha = 1.0) {
                                self$n_clusters <- n_cluster
                                self$alpha <- alpha
                                self$initial_weights <- weights
                                
                              },
                              
                              build = function(input_shape) {
                                input_dim <- input_shape[2]
                                #self.input_spec = InputSpec
                                self$clusters <- self$add_weight(
                                  name = 'clusters', 
                                  shape = list(self$n_clusters, input_dim),
                                 initialer = 'glorot_uniform'
                                )
                              if(!isnull(self$initial_weights)) {
                                self$set_weights(self$initial_weights)
                                                      }
                              self$built = TRUE
                                
                              },
                              
                              call = function(inputs) {
                                q <- 1/(1+ (k_sum(k_square(k_expand_dims(inputs,axis = 1)-self$clusters), axis = 2)/self$alpha))
                                q <- q^((self.aplha+1)/2)
                                q <- k_transpose(k_transpose(q)/ k_sum(q, axis = 1))
                                return(q)
                              }
                              
                            )
                          )

layer_my_dense <- function(object, n_clusters, weights = NULL,alpha = 1.0,name = NULL, trainable = TRUE) {
  create_layer(clustering, object, list(
    n_cluster = as.integer(n_clusters),
    weights = weights,
    alpha = alpha,
    name = name,
    trainable = trainable
  ))
}

cl <- layer_my_dense(n_cluster = 6)

