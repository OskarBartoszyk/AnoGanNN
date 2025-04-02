import numpy as np 
import activationFunctions as af


import numpy as np
import activationFunctions as af


class LSTM:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        # Forget gate parameters
        self.Wf = np.random.randn(output_size, input_size + output_size)
        self.bf = np.zeros(output_size)

        # Input gate parameters
        self.Wi = np.random.randn(output_size, input_size + output_size)
        self.bi = np.zeros(output_size)

        # Candidate memory cell parameters
        self.Wc = np.random.randn(output_size, input_size + output_size)
        self.bc = np.zeros(output_size)

        # Output gate parameters
        self.Wo = np.random.randn(output_size, input_size + output_size)
        self.bo = np.zeros(output_size)

        # Cell state and hidden state
        self.ct = np.zeros(output_size)
        self.ht = np.zeros(output_size)

    def forward(self, xt):
        # Concatenate input and previous hidden state
        concat = np.concatenate((xt, self.ht))

        # Forget gate
        ft = af.sigmoid(np.dot(self.Wf, concat) + self.bf)

        # Input gate
        it = af.sigmoid(np.dot(self.Wi, concat) + self.bi)
        c_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)

        # Update cell state
        self.ct = ft * self.ct + it * c_tilde

        # Output gate
        ot = af.sigmoid(np.dot(self.Wo, concat) + self.bo)
        self.ht = ot * np.tanh(self.ct)

        return self.ht