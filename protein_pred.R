#Various libraries
library(tidyverse)  
library(caret)
library(tensorflow) 
library(ranger)     #Random forest (more efficient than rf library for higher dimensional data)
library(nnet)  #Neural net
library(keras) 
library(gtable)
library(luz) # High-level interface for torch
library(torchvision) # For datasets and image transformation
library(torchdatasets) # For datasets we are going to use
library(randomForest)
library(zeallot)
library(rpart)
library(rpart.plot)
library(torch)
library(torchvision)
library(xgboost)

# Load CASP 5.9 Set from 
dataset <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv", header = TRUE, stringsAsFactors = FALSE)

# Remove missing data (none in CASP dataset), create protein set
protein <- na.omit(dataset)

#n number of protein residues: 45730 obs
n <- nrow(protein)
# For reproduction
set.seed(7)

# Number of test samples = 1/3 observations
ntest <- trunc(n / 3)
#randomly select test IDs from 
testid <- sample(1:n, ntest)

# Set PyTorch seed
torch_manual_seed(7)

# Scale variables, separate x labels ,y target var. Remove intercepts
x <- scale(model.matrix(protein$RMSD ~ . - 1, data = protein))
y <- protein$RMSD




#Gradient Boosted Tree
# Gradient Boosted Regression Tree with MAE Objective Function
gdboosteds_protein <- xgboost(data = x[-testid,], label = y[-testid], nrounds = 10000, objective = "reg:absoluteerror")
gdboost_test_pred <- predict(gdboosteds_protein, x[testid, ])

# Calculate MSE
mse <- mean((gdboost_test_pred-y[testid])^2)
#19.3254 MSE for set.seed(7), nround = 10000
mse


# Calculate MAE
mae <- mean(abs(gdboost_test_pred-y[testid]))
#2.979239 MAE for set.seed(7), nround = 10000
mae


importance_matrix <- xgb.importance(model = gdboosteds_protein)
importance_matrix
xgb.plot.importance(importance_matrix, col = rgb(124/256,148/256,198/256), xlab = "Importance (0-1)", 
                    ylab = "Feature")


# First Method
# Random Forest regression tree with the obejctive of plotting feature important and purposely overfitting a model
# Very easy to produce good looking tree, extremely overfitted and validation MAE explodes

library(randomForest)


# Split the data into training and testing sets

# *** Un-comment these blocks to run the Random Forest Model. Commented out so the train_id object does not affect Nueral Networks ***

# For reproducibility
set.seed(13)
#train_id <- sample(1:n, floor(n * 0.6), replace = FALSE)
#valid_id <- sample(setdiff(1:n, train_id), floor(n * 0.2), replace = FALSE)
#test_id <- setdiff(1:n, union(train_id, valid_id))


#train_x <- x[train_id, ]
#train_y <- y[train_id]
#valid_x <- x[valid_id, ]
#valid_y <- y[valid_id]
#test_x <- x[test_id, ]
#test_y <- y[test_id]


# Train the random forest regression model
rf_model <- randomForest(train_x, train_y, ntree = 500)

# Predict RMSD for test set using the trained model
pred_valid <- predict(rf_model, valid_x)
pred_test <- predict(rf_model, test_x)

# Calculate mean squared error on the test set
mae_valid <- mean(abs(pred_valid - test_y))
mae_test <- mean(abs(pred_test-test_y))

#MAE 2.5555 on test set with  
# After purposely overfitting as much as possible (try nrounds >>500) this MAE of 2.55 appears to be the absolute minimume error achievable
mae_test

#6.161
mae_valid

# Strict Regression Tree
protein_tree1 <- rpart(RMSD ~ ., data = protein[train_id,], control = rpart.control(minsplit = 1))

# Predict on Training set, goal was to minimize MAE
train_pred_protein_tree1 <- predict(protein_tree1, protein[train_id,])
# Calculate MAE on training set
train_tree1_mae <- mean(abs(train_pred_protein_tree1-protein$RMSD[train_id]))
train_tree1_mae

# Predict on Test Set, Calculate MAE
test_pred_protein_tree1 <- predict(protein_tree1, newdata = protein[test_id, ])

# Calculate MAE for test set
mae_protein_tree1 <- mean(abs(pred_protein_tree1 - y[testid]))
mae_protein_tree1

rpart.plot(protein_tree1)

  

#create NN with four layers, input, 50 node hidden layer, Relu activation, dropout layer 0.4 
modnn <- nn_module(
  initialize = function(input_size) {
    self$hidden <- nn_linear(input_size, 50)
    self$activation1 <- nn_relu()
    self$dropout <- nn_dropout(0.4)
    self$output <- nn_linear(50, 1)
  },
  forward = function(x) {
    ## x %>%
    ##   self$hidden() %>%
    ##   self$activation() %>%
    ##   self$dropout() %>%
    ##   self$output() %>%
    self$output( self$dropout( self$activation1( self$hidden(x) ) ) )
  }
)

#set hyperparameters for nn_module
# MSE loss function
modnn <- set_hparams( setup(modnn ,
                            loss = nn_mse_loss(),
                            optimizer = optim_rmsprop,
                            metrics = list(luz_metric_mae())),
                      input_size = ncol(x))


#training
#takes roughly 6 hours to train at 100 epochs, 210 sec each
fitted <-   fit(modnn,
                data = list(x[-testid, ],
                            matrix(y[-testid], ncol = 1)),
                valid_data = list(x[testid, ],
                                  matrix(y[testid], ncol = 1)),
                epochs = 10)


plot(fitted)
summary(fitted)

fitted$ctx

# Predict RMSD values for test set
test_pred <- predict(fitted, x[testid, ])

#calc MSE
mse <- mean((test_pred-y[testid])^2)
mse
#50.7419 for 100 epoch model

# scatterplot of predicted vs actual RMSD
plot(y[testid], test_pred, xlab = "Actual RMSD (Angstrom)", ylab = "Predicted RMSD (Angstrom)", main = "RELU HL Nueral Network")
abline(a = 0, b = 1, col = "purple")



#best model
#2nd Model, Add elu hidden layer
modnn2 <- nn_module(
  initialize = function(input_size) {
    self$hidden1 <- nn_linear(input_size, 50)
    self$activation1 <- nn_relu()
    self$dropout1 <- nn_dropout(0.4)
    self$hidden2 <- nn_linear(50, 50)
    self$activation2 <- nn_elu()
    self$dropout2 <- nn_dropout(0.4)
   # self$hidden3 <- nn_linear()
    
    self$output <- nn_linear(50, 1)
  },
  forward = function(x) {
    x %>%
      self$hidden1() %>%
      self$activation1() %>%
      self$dropout1() %>%
      self$hidden2() %>%
      self$activation2() %>%
      self$dropout2() %>%
      self$output()
  }
)

modnn2 <- set_hparams( setup(modnn2 ,
                            loss = nn_mse_loss(),
                            optimizer = optim_rmsprop,
                            metrics = list(luz_metric_mae())),
                      input_size = ncol(x))


#training
#210sec per epoch
fitted2 <-   fit(modnn2,
                data = list(x[-testid, ],
                            matrix(y[-testid], ncol = 1)),
                valid_data = list(x[testid, ],
                                  matrix(y[testid], ncol = 1)),
                epochs = 100)

plot(fitted2)
summary(fitted2)

fitted2$ctx

#predict RMSD values for test set
test_pred2 <- predict(fitted2, x[testid, ])
mae_nn2 <- mean(abs(test_pred2 - y))
mae_nn2
plot(y[testid], test_pred2, xlab = "Actual RMSD (Angstrom)", ylab = "Predicted RMSD (Angstrom)", main = "NN2: 2 Layer RELU/ELU")
abline(a = 0, b = 1, col = "purple")







#ModNN3

#add CELU hidden layer
modnn3 <- nn_module(
  initialize = function(input_size) {
    self$hidden1 <- nn_linear(input_size, 50)
    self$activation1 <- nn_relu()
    self$dropout1 <- nn_dropout(0.4)
    self$hidden2 <- nn_linear(50, 50)
    self$activation2 <- nn_elu()
    self$dropout2 <- nn_dropout(0.4)
    self$hidden3 <- nn_linear(50, 50)
    self$activation3 <- nn_celu()
    self$dropout3 <- nn_dropout(0.4)
    # self$hidden3 <- nn_linear()
    
    self$output <- nn_linear(50, 1)
  },
  forward = function(x) {
    x %>%
      self$hidden1() %>%
      self$activation1() %>%
      self$dropout1() %>%
      self$hidden2() %>%
      self$activation2() %>%
      self$dropout2() %>%
      self$hidden3() %>%
      self$activation3() %>%
      self$dropout3() %>% 
      self$output()
  }
)
nn_get_parameters(modnn3)
modnn3 <- set_hparams( setup(modnn3 ,
                             loss = nn_mse_loss(),
                             optimizer = optim_rmsprop,
                             metrics = list(luz_metric_mae())),
                       input_size = ncol(x))


#training
#210sec per epoch
fitted3 <-   fit(modnn3,
                 data = list(x[-testid, ],
                             matrix(y[-testid], ncol = 1)),
                 valid_data = list(x[testid, ],
                                   matrix(y[testid], ncol = 1)),
                 epochs = 100)


matrix(y[-testid], ncol = 1)

list(x[-testid, ],
     matrix(y[-testid], ncol = 1))

plot(fitted3)
summary(fitted3)
fitted3$model
fitted3$ctx



# Predict RMSD values for test set
test_pred3 <- predict(fitted3, x[testid, ])

test_pred3 <- predict(fitted3, x[testid, ])

plot(y[testid], test_pred3, xlab = "Actual RMSD (Angstrom)", ylab = "Predicted RMSD (Angstrom)", main = "NN3: 3 Layer RELU/ELU/CELU")
abline(a = 0, b = 1, col = "purple")

#NN3 MAE calc
length(testid)



mae_nn3 <- mean(abs(test_pred3 - y[testid]))
mae_nn3 



#add binary step (Threshold) Hidden Layer
modnn4 <- nn_module(
  initialize = function(input_size) {
    self$hidden1 <- nn_linear(input_size, 50)
    self$activation1 <- nn_relu()
    self$dropout1 <- nn_dropout(0.4)
    self$hidden2 <- nn_linear(50, 50)
    self$activation2 <- nn_elu()
    self$dropout2 <- nn_dropout(0.4)
    self$hidden3 <- nn_linear(50, 50)
    self$activation3 <- nn_relu()#nn_threshold ( value = 15, inplace = FALSE, threshold = 15)
    self$dropout3 <- nn_dropout(0.4)
    self$hidden4 <- nn_linear(50, 50)
    self$activation4 <- nn_elu()
    self$dropout4 <- nn_dropout(0.4)
    # self$hidden3 <- nn_linear()
    
    self$output <- nn_linear(50, 1)
  },
  forward = function(x) {
    x %>%
      self$hidden1() %>%
      self$activation1() %>%
      self$dropout1() %>%
      self$hidden2() %>%
      self$activation2() %>%
      self$dropout2() %>%
      self$hidden3() %>%
      self$activation3() %>%
      self$dropout3() %>% 
      self$hidden4() %>%
      self$activation4() %>%
      self$dropout4() %>% 
      self$output()
  }
)

modnn4 <- set_hparams( setup(modnn4 ,
                             loss = nn_mse_loss(),
                             optimizer = optim_rmsprop,
                             metrics = list(luz_metric_mae())),
                       input_size = ncol(x))


#training
#215sec per epoch
fitted4 <-   fit(modnn4,
                 data = list(x[-testid, ],
                             matrix(y[-testid], ncol = 1)),
                 valid_data = list(x[testid, ],
                                   matrix(y[testid], ncol = 1)),
                 epochs = 200)

plot(fitted4)
summary(fitted4)

xfitted4$ctx

# Predict RMSD values for test set
test_pred4 <- predict(fitted4, x[testid, ])

test_pred4 <- predict(fitted4, x[testid, ])
plot(y[testid], test_pred4, xlab = "Actual RMSD (Angstrom)", ylab = "Predicted RMSD (Angstrom)", main = "3 Layer RELU/ELU Nueral Network")
abline(a = 0, b = 1, col = "purple")





#stack on a bunch of hidden layers
modnn5 <- nn_module(
  initialize = function(input_size) {
    self$hidden1 <- nn_linear(input_size, 50)
    self$activation1 <- nn_relu()
    self$dropout1 <- nn_dropout(0.5)
    
    self$hidden2 <- nn_linear(50, 50)
    self$activation2 <- nn_elu()
    self$dropout2 <- nn_dropout(0.5)
    
    self$hidden3 <- nn_linear(50, 50)
    self$activation3 <- nn_relu()
    self$dropout3 <- nn_dropout(0.5)
    
    self$hidden4 <- nn_linear(50, 50)
    self$activation4 <- nn_elu()
    self$dropout4 <- nn_dropout(0.5) 
    
    self$hidden5 <- nn_linear(50, 50)
    self$activation5 <- nn_relu()
    self$dropout5 <- nn_dropout(0.5)  
    
    self$hidden6 <- nn_linear(50, 50)
    self$activation6 <- nn_relu()
    self$dropout6 <- nn_dropout(0.5)
    
    self$hidden7 <- nn_linear(50, 50)
    self$activation7 <- nn_elu()
    self$dropout7 <- nn_dropout(0.25) 
    
    self$hidden8 <- nn_linear(50, 50)
    self$activation8 <- nn_elu()
    self$dropout8 <- nn_dropout(0.5) 
    
    self$hidden9 <- nn_linear(50, 50)
    self$activation9 <- nn_elu()
    self$dropout9 <- nn_dropout(0.5) 
    
    self$hidden10 <- nn_linear(50, 50)
    self$activation10 <- nn_relu()
    self$dropout10 <- nn_dropout(0.5) 
    
    self$hidden11 <- nn_linear(50, 50)
    self$activation11 <- nn_relu()
    self$dropout11 <- nn_dropout(0.5) 
    
    self$hidden12 <- nn_linear(50, 50)
    self$activation12 <- nn_relu()
    self$dropout12 <- nn_dropout(0.5)
    
    self$output <- nn_linear(50, 1)
  },
  forward = function(x) {
    x %>%
      self$hidden1() %>%
      self$activation1() %>%
      self$dropout1() %>%
      
      self$hidden2() %>%
      self$activation2() %>%
      self$dropout2() %>%
      
      self$hidden3() %>%
      self$activation3() %>%
      self$dropout3() %>% 
      
      self$hidden4() %>%
      self$activation4() %>%
      self$dropout4() %>%
      
      self$hidden5() %>%
      self$activation5() %>%
      self$dropout5() %>%
      
      self$hidden6() %>%
      self$activation6() %>%
      self$dropout6() %>%
      
      self$hidden7() %>%
      self$activation7() %>%
      self$dropout7() %>%
      
      self$hidden8() %>%
      self$activation8() %>%
      self$dropout8() %>%
      
      self$hidden9() %>%
      self$activation9() %>%
      self$dropout9() %>%
      
      self$hidden10() %>%
      self$activation10() %>%
      self$dropout10() %>%
      
      self$hidden11() %>%
      self$activation11() %>%
      self$dropout11() %>%
      
      self$hidden12() %>%
      self$activation12() %>%
      self$dropout12() %>%
      
      self$output()
  }
)

modnn5 <- set_hparams( setup(modnn5 ,
                             loss = nn_mse_loss(),
                             optimizer = optim_rmsprop,
                             metrics = list(luz_metric_mae())),
                       input_size = ncol(x))


#training
#210sec per epoch
fitted5 <-   fit(modnn5,
                 data = list(x[-testid, ],
                             matrix(y[-testid], ncol = 1)),
                 valid_data = list(x[testid, ],
                                   matrix(y[testid], ncol = 1)),
                 epochs = 200)

plot(fitted5)
summary(fitted5)

xfitted2$ctx

# Predict RMSD values for test set
test_pred3 <- predict(fitted3, x[testid, ])

test_pred3 <- predict(fitted3, x[testid, ])
plot(y[testid], test_pred3, xlab = "Actual RMSD (Angstrom)", ylab = "Predicted RMSD (Angstrom)")
abline(a = 0, b = 1, col = "purple")








modnn4 <- nn_module(
  initialize = function(input_size) {
    self$hidden1 <- nn_linear(input_size, 50)
    
    # Use different activation functions for each feature
    self$activation1 <- nn_relu()  # F1, F2, F3, F5, F6, F7, F9
    self$activation2 <- nn_sigmoid()  # F4
    self$activation3 <- nn_linear()  # F8
    
    self$dropout1 <- nn_dropout(0.4)
    self$hidden2 <- nn_linear(50, 50)
    self$dropout2 <- nn_dropout(0.4)
    
    self$output <- nn_linear(50, 1)
  },
  forward = function(x) {
    # Split the input tensor based on the feature indices
    x1 <- x[, 1:7]  # F1, F2, F3, F4, F5, F6, F7
    x2 <- x[, 8]  # F8
    x3 <- x[, 9]  # F9
    
    # Forward pass for each feature with the corresponding activation function
    out1 <- x1 %>%
      self$hidden1() %>%
      self$activation1() %>%
      self$dropout1() %>%
      self$hidden2() %>%
      self$activation2() %>%
      self$dropout2()
    
    out2 <- x2 %>%
      self$hidden1() %>%
      self$activation3()  # Use linear activation function for F8
    
    out3 <- x3 %>%
      self$hidden1() %>%
      self$activation1()  # Use ReLU activation function for F9
    
    # Concatenate the output tensors from each feature and pass through the output layer
    torch$cat(list(out1, out2, out3), dim = 2) %>%
      self$output()
  }
)

modnn4_setup <- setup(modnn4 ,
                      loss = nn_mse_loss(),
                      optimizer = optim_rmsprop,
                      metrics = list(luz_metric_mae()))

modnn4 <- set_hparams(modnn4_setup, input_size = ncol(x))

#training
#210sec per epoch
fitted4 <- fit(modnn4,
               data = list(x[-testid, ],
                           matrix(y[-testid], ncol = 1)),
               valid_data = list(x[testid, ],
                                 matrix(y[testid], ncol = 1)),
               epochs = 100)
