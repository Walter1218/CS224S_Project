# Dimensions
num_input_features 	= 39
encoder_hidden_size	= 100
decoder_hidden_size	= 200
embedding_dim 		= 120
vocab_size			= 30

# Hyperparameters
lr 					= 1e-3
dropout_p 			= 0.6
clip_gradients		= False
clip_val          	= 10

# Training Settings
num_epochs 			= 100
batch_size 			= 16
max_in_len 			= 500
max_out_len 		= 200
max_grad_norm 		= 10
print_every			= 50

num_beams 			= 25
