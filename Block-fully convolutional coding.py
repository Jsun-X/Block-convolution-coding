import random
import numpy as np


class Neural_network_coder(object):
    def __init__(self, input_size, hidden_size, output_size):
        self.data_matrix = []
        self.data_matrix_transpose = []
        # self.random_num = None
        self.block_num = None
        self.row_data = None
        self.input_data = None
        self.block_matrix = []
        self.input_data_matrix = []
        self.output_data = []
        self.encoded_block_matrix = []
        self.temporary_matrix = []
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_input_hidden = np.random.rand(hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden_output = np.random.rand(output_size)

    @staticmethod
    def sigmoid(x):
        # Define the activation function
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        # Forward propagation
        hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_input_hidden
        hidden_output = self.sigmoid(hidden_input)
        output = np.dot(hidden_output, self.weights_hidden_output) + self.bias_hidden_output
        return output

    def generate_random_binary(self):
        if self.input_data is not None:
            string = ''.join([str(random.randint(0, 1)) for _ in range(self.input_data)])
        else:
            print("No input data available")
        return string

    def generate_data(self):
        for i in range(0, self.block_num):
            for j in range(0, self.input_data):
                if self.input_data is not None:
                    random_num = ''.join([str(random.randint(0, 1)) for _ in range(self.input_data)])
                    with open("data.txt", "a") as file:
                        file.write(random_num + '\n')
                else:
                    print("No input data available")

    def code(self):
        with open("data.txt", "r") as f:
            for line in f:
                self.data_matrix.append(line)  # Remove newline characters

            # Block the data.
            for i in range(0, self.block_num):
                for j in range(0, self.input_data):
                    self.row_data = self.data_matrix[i * self.input_data + j]
                    self.block_matrix.append(self.row_data)
                matrix_list = [self.block_matrix for _ in range(self.block_num)]

            # Full convolution of partitioned data.
            for matrix in matrix_list:
                for current_input in matrix:
                    # Create the neural network instance
                    input_size = 32  # Input layer size
                    hidden_size = 64  # Hidden layer size
                    output_size = 32  # Output layer size

                    model = Neural_network_coder(input_size, hidden_size, output_size)

                    # Define input data
                    self.input_data_matrix = [int(num) for num in current_input if num.isdigit()]

                    # The neural network model is invoked for forward propagation
                    self.output_data = model.forward(self.input_data_matrix)
                    self.encoded_block_matrix.append(str(bin(int(num))) for num in self.output_data)
                # encode_block_matrix_list = [self.encoded_block_matrix for _ in range(self.block_num)]
            with open("Output_data.txt", "w") as file:
                for encoded_block in self.encoded_block_matrix:
                    file.write(next(encoded_block) + '\n')


def main():
    block_convolutional_encoder = Neural_network_coder(32, 64, 32)
    block_convolutional_encoder.input_data = 32
    block_convolutional_encoder.block_num = random.randint(5, 15)

    block_convolutional_encoder.generate_data()
    block_convolutional_encoder.code()


if __name__ == "__main__":
    main()
