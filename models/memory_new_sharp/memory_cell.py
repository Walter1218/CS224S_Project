import tensorflow as tf
import numpy as np

class MyMemCell(tf.nn.rnn_cell.RNNCell):
    """Wrapper around our GRU cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, num_units, memory, cell, config):
        # super(MyAttCell, self).__init__(num_units=num_units)
        self.memory = memory
        # print 'My memory shape', memory.get_shape()
        self.cell = cell
        self.config = config
        self.Wc = tf.get_variable('WcAtt', shape=(2*self.config.decoder_hidden_size + memory.get_shape().as_list()[-1],\
                                self.config.decoder_hidden_size), \
                                initializer=tf.contrib.layers.xavier_initializer())
        self.W_ers = tf.get_variable('WErs', shape=(self.config.decoder_hidden_size,\
                                self.config.decoder_hidden_size), \
                                initializer=tf.contrib.layers.xavier_initializer())
        self.W_add = tf.get_variable('WAdd', shape=(self.config.decoder_hidden_size,\
                                self.config.decoder_hidden_size), \
                                initializer=tf.contrib.layers.xavier_initializer())

        self.w_rg = tf.get_variable('w_rg', shape=(self.config.decoder_hidden_size, 1), \
                                initializer=tf.contrib.layers.xavier_initializer())

        self.w_wg = tf.get_variable('w_wg', shape=(self.config.decoder_hidden_size, 1), \
                                initializer=tf.contrib.layers.xavier_initializer())


    @property
    def state_size(self):

        # Prev output attention vector, w_r, w_w
        sizes = [self.config.decoder_hidden_size, self.config.num_cells, self.config.num_cells]

        # Memory bank dimension
        sizes += [tf.TensorShape([tf.Dimension(self.config.num_cells), tf.Dimension(self.config.decoder_hidden_size)])]

        # The previous hidden vector for each layer
        for i in range(self.config.num_layers):
            sizes = [self.config.decoder_hidden_size] + sizes

        print 'Sizes is', sizes
        # Convert into tuple
        return tuple(sizes)

    @property
    def output_size(self):
        return self.config.decoder_hidden_size


    # Reads from the external memory bank
    def read_memory(self, cell_output, M_B_prev, w_r_prev):

        # Compute scores
        output = tf.expand_dims(cell_output, 1)
        scores = tf.matmul(output, M_B_prev, transpose_b=True)
        scores = tf.squeeze(scores, [1])

        # Compute probability distribution over memory cells
        w_rt = tf.nn.softmax(logits = scores)

        # Use gate to keep around information from last timestep
       # gate = tf.sigmoid(tf.matmul(cell_output, self.w_rg))
       # w_rt = (gate*w_r_prev) + (1 - gate)*(w_r_tilde)

        # Compute the final weighted vector from the memory bank
        w_rt_expanded = tf.expand_dims(w_rt, 1)
        r_t = tf.matmul(w_rt_expanded, M_B_prev)
        r_t = tf.squeeze(r_t, [1])

        # Return final vector, and w_rt to pass along to next step
        return r_t, w_rt 


    def write_memory(self, s_t, M_B_prev, w_w_prev):

        # Follow a similar procedure as with reading to first
        # calculate a new w_wt
        output = tf.expand_dims(s_t, 1)
        scores = tf.matmul(output, M_B_prev, transpose_b=True)
        scores = tf.squeeze(scores, [1])
        w_wt = tf.nn.softmax(logits = scores)
        #gate = tf.sigmoid(tf.matmul(s_t, self.w_wg))

        # w_wt has size [Batch Size, Num Cells]
        #w_wt = (gate*w_w_prev) + (1 - gate)*(w_w_tilde)

        # ERASE first

        #e_t has shape [Batch Size, Decoder Size]
        e_t = tf.sigmoid(tf.matmul(s_t, self.W_ers))

        # Should have shape [Batch Size, Num Cells, Decoder Size]
        mult_matrix = 1 - tf.batch_matmul(tf.expand_dims(w_wt, 2), tf.expand_dims(e_t, 1))
        
        # Element-wise multiply
        M_tilde = M_B_prev * mult_matrix

        # Then ADD
        add_t = tf.sigmoid(tf.matmul(s_t, self.W_add))

        # Matrix for addition
        add_matrix = tf.batch_matmul(tf.expand_dims(w_wt, 2), tf.expand_dims(add_t, 1))

        # Compute new memory bank
        M_new = M_tilde + add_matrix

        return M_new, w_wt      

    # State should contain previous GRU state, prev final attention vector, w_r, w_w, external memory
    def __call__(self, inputs, state, scope=None):

        # Extract information from previous state
        prev_states, prev_attention, w_r_prev, w_w_prev, M_B_prev = (state[:self.config.num_layers], state[-4], state[-3], state[-2], state[-1])
        
        # Read from the memory bank first using the prev output
        r_t, w_rt = self.read_memory(prev_attention, M_B_prev, w_r_prev)

        # Use the memory vector and concatenate with input
        new_inputs = tf.concat(1, [inputs, r_t])

        # Get the new GRU state
        cell_output, new_state = self.cell(new_inputs, prev_states, scope)

        # Using the vector output from the cell, perform attention
        output = tf.expand_dims(cell_output, 1)
        scores = tf.matmul(output, self.memory, transpose_b=True)
        scores = tf.squeeze(scores, [1])
        probs = tf.nn.softmax(logits = scores)

        probs = tf.expand_dims(probs, 1)
        context = tf.matmul(probs, self.memory)
        s_t = tf.squeeze(context, [1])

        # The output will be tanh(W_c dot [s_t; r_t])
        concat = tf.concat(1, [s_t, r_t, cell_output])
        attention_vec = tf.tanh(tf.matmul(concat, self.Wc))

        # Write the output vector to memory
        M_B, w_wt = self.write_memory(attention_vec, M_B_prev, w_w_prev)

        # Form the next state
        next_state = tuple(list(new_state) + [attention_vec, w_rt, w_wt, M_B])

        return attention_vec, next_state

