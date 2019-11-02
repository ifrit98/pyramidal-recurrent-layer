

PyramidalRecurrentBlock <- 
  R6::R6Class(
    "PyramidalRecurrentBlock",
    
    inherit = KerasLayer,
    
    public = list(
      units = NULL,
      num_layers = NULL,
      cell_type = NULL,
      activation = NULL,
      kernel_initializer = NULL,
      kernel_regularizer = NULL,
      recurrent_activation = NULL,
      recurrent_initializer = NULL,
      recurrent_regularizer = NULL,
      bias_initializer = NULL,
      bias_regularizer = NULL,
      recurrent_dropout= NULL,
      cell = NULL,
      layers = NULL,
      
      initialize = function(
        units,
        num_layers,
        cell_type,
        activation,
        kernel_initializer,
        kernel_regularizer,
        recurrent_activation,
        recurrent_initializer,
        recurrent_regularizer,
        bias_initializer,
        bias_regularizer,
        recurrent_dropout
        
      ) {
        self$units <-  units
        self$num_layers <- num_layers
        self$cell_type <- cell_type
        self$activation <- activation
        self$kernel_initializer <- kernel_initializer
        self$kernel_regularizer <- kernel_regularizer
        self$recurrent_activation <- recurrent_activation
        self$recurrent_initializer <- recurrent_initializer
        self$recurrent_regularizer <- recurrent_regularizer
        self$bias_initializer <- bias_initializer
        self$bias_regularizer <- bias_regularizer
        self$recurrent_dropout <- recurrent_dropout
      },
      
      pad_sequence = function(output) {
        shape           <- output$shape
        batch_size      <- shape[0]
        sequence_length <- shape[1]
        num_units       <- shape[-1]
        
        padding <- list(c(0L, 0L),
                        c(0L, tf$math$floormod(sequence_length, 2L)),
                        c(0L, 0L))
        
        output <- tf$pad(output, padding)
        
        concat_output <- tf$reshape(output,
                                    c(batch_size, -1L,
                                      tf$math$multiply(num_units, 2L)))
        concat_output
      },
      
      build = function(input_shape) {
        
        stopifnot(self$cell_type %in% c("LSTM", "GRU", "RNN"))
        
        self$cell <- switch(
          self$cell_type,
          "LSTM" = layer_lstm,
          "GRU"  = later_gru,
          "RNN"  = layer_simple_rnn
        )
        
        self$layers <- map(
          .x = vector(mode = "list", length = self$num_layers),
          .f = function(x) bidirectional(layer = self$cell(
            units = self$units,
            recurrent_dropout = self$recurrent_dropout,
            return_sequences = TRUE,
            return_state = TRUE,
          ))
        )
      },
      
      
      call = function(x, mask = NULL) {
        output <- x
        
        i <- 0L
        for (layer in self$layers) {
          c(output,
            hidden_forward,
            context_forward,
            hidden_backward,
            context_backward) %<-% layer(output)
          
          output <- tf$concat(output, -1L)
          state  <- tf$concat(
            list(context_forward, context_backward), -1L)
          
          if (i > 0L)
            output <- self$pad_sequence(output)
          
          i <- i + 1L
        }
        
        c(output, state)
      },
      
      compute_output_shape = function(input_shape) {
        output_shape <-
          list(input_shape[[1]],
               as.integer(input_shape[[2]] %/% 2L ^ (self$num_layers - 1L)),
               input_shape[[3]])
        
        output_shape
      }
      
    )
  )


layer_pyramidal_recurrent_block <-
  function(object,
           units,
           num_layers = 3,
           cell_type = 'LSTM',
           activation = 'tanh',
           kernel_initializer = 'glorot_normal',
           kernel_regularizer = NULL,
           recurrent_activation = 'sigmoid',
           recurrent_initializer = 'orthogonal',
           recurrent_regularizer = NULL,
           bias_initializer = 'zeros',
           bias_regularizer = NULL,
           recurrent_dropout = 0.0,
           name = NULL,
           trainable = TRUE) {
    create_layer(
      PyramidalRecurrentBlock,
      object,
      list(
        units = as.integer(units),
        num_layers = as.integer(num_layers),
        cell_type = toupper(as.character(cell_type)),
        activation = tf$keras$activations$get(activation),
        kernel_initializer = tf$keras$initializers$get(kernel_initializer),
        kernel_regularizer = tf$keras$regularizers$get(kernel_regularizer),
        recurrent_activation = tf$keras$activations$get(recurrent_activation),
        recurrent_initializer = tf$keras$initializers$get(recurrent_initializer),
        recurrent_regularizer = tf$keras$regularizers$get(recurrent_regularizer),
        bias_initializer = tf$keras$initializers$get(bias_initializer),
        bias_regularizer = tf$keras$regularizers$get(bias_regularizer),
        recurrent_dropout = recurrent_dropout,
        name = name,
        trainable = trainable
      )
    )
  }
