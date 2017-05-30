import tensorflow as tf
import numpy as np
import config

class MyAttCell(tf.nn.rnn_cell.GRUCell):
    """Wrapper around our GRU cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, num_units, memory):
        super(MyAttCell, self).__init__(num_units=num_units)
        self.memory = memory
        self.Wc = tf.get_variable('WcAtt', shape=(config.decoder_hidden_size + 2*config.encoder_hidden_size,\
                                config.decoder_hidden_size), \
                                initializer=tf.contrib.layers.xavier_initializer())

    def __call__(self, inputs, state, scope=None):
        """Updates the state using the previous @state and @inputs.
        Remember the GRU equations are:

        z_t = sigmoid(x_t U_z + h_{t-1} W_z + b_z)
        r_t = sigmoid(x_t U_r + h_{t-1} W_r + b_r)
        o_t = tanh(x_t U_o + r_t * h_{t-1} W_o + b_o)
        h_t = z_t * h_{t-1} + (1 - z_t) * o_t

        TODO: In the code below, implement an GRU cell using @inputs
        (x_t above) and the state (h_{t-1} above).
            - Define W_r, U_r, b_r, W_z, U_z, b_z and W_o, U_o, b_o to
              be variables of the apporiate shape using the
              `tf.get_variable' functions.
            - Compute z, r, o and @new_state (h_t) defined above
        Tips:
            - Remember to initialize your matrices using the xavier
              initialization as before.
        Args:
            inputs: is the input vector of size [None, self.input_size]
            state: is the previous state vector of size [None, self.state_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector.
        """
        cell_output, new_state = super(MyAttCell, self).__call__(inputs, state, scope)

        output = tf.expand_dims(cell_output, 1)
        scores = tf.matmul(output, self.memory, transpose_b=True)
        scores = tf.squeeze(scores, 1)
        probs = tf.nn.softmax(logits = scores)

        probs = tf.expand_dims(probs, 1)
        context = tf.matmul(probs, self.memory)
        context = tf.squeeze(context, [1])

        concat = tf.concat(1, [cell_output, context])
        final_output = tf.tanh(tf.matmul(concat, self.Wc))

        return final_output, new_state

