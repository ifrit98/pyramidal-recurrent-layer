library(tensorflow)
library(keras)
library(magrittr)
source('pyramidal-recurrent-block.R')


# Data Preparation --------------------------------------------------------

batch_size <- 1024
num_classes <- 10
epochs <- 40

# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Redimension
x_train <- array_reshape(x_train, c(nrow(x_train), 784, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 784, 1))

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)



# Define & Train Model -------------------------------------------------
input <- layer_input(shape = c(784, 1))

# Must specify batch dimension!
output <- input %>% 
  layer_pyramidal_recurrent_block(units = 16, 
                                  batch_size = batch_size) %>% 
  {layer_global_max_pooling_1d(.[[1]])} %>% 
  layer_dense(units = num_classes, activation = 'softmax')

# Compile the model
(model <- build_and_compile(input, output))

# Train the model
model %>% fit(x_train, y_train,
              batch_size = batch_size,
              epochs = epochs,
              verbose = 1,
              validation_data= list(x_test, y_test)
)
