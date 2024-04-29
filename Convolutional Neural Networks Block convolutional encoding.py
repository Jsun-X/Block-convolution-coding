import random
import numpy as np

# Clear and close input.txt and output_data.txt before running, and adjust the convolution exposure as needed.
class Neural_network_coder(object):
    def __init__(self):
        self.data_matrix = []
        self.data_matrix_transpose = []
        self.block_num = None
        self.row_data = None
        self.input_data = None
        self.k = None
        self.block_matrix = []
        self.input_data_matrix = []
        self.output_data_matrix = []
        self.output_data = []
        self.encoded_block_matrix = []
        self.temporary_matrix = []

    @staticmethod
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
                    with open("input.txt", "a") as file:
                        file.write(random_num + '\n')
                else:
                    print("No input data available")

    def Convolutional_encoding_computation(input):
        k = 3
        # 卷积窗口为k。
        output = [[0]*(len(input[0])-k + 1) for _ in range(len(input)-k + 1)]
        #  output.reshape(self.input_data-self.k, self.input_data-self.k)
        for i in range(len(input)-k+1):
            for j in range(len(input[0])-k+1):
                for n in range(k):
                    for m in range(k):
                        if i + n < len(input) and j + m < len(input[0]):
                            output[i][j] += input[i + n][j + m]
        return output


    def code(self):
        with open("input.txt", "r") as f:
            input_data_matrix = [[0] * 64 for _ in range(64)]
            for line in f:
                self.data_matrix.append(line)  # Remove newline characters

            # Block the data.
            for i in range(0, self.block_num):
                for j in range(0, self.input_data):
                    self.row_data = self.data_matrix[i * self.input_data + j]
                    # A string together.
                    for x in range(len(self.row_data)):
                        if self.row_data[x].isdigit():
                            num = self.row_data[x]
                            data = str(num)
                            # print(data + " " + str(j) + " " + str(x))
                    # data = str(num for num in self.row_data if num.isdigit())
                    #self.block_matrix.append(data)  # Save in a string.
                    # matrix_list = [self.block_matrix for _ in range(self.block_num)]

            # Full convolution of partitioned data.
            #         data_list = list(data)  # 将字符串转换为列表
            #         data_value = next(iter(data_list))  # 使用迭代器对象获取下一个值
                            input_data_matrix[j][x] = int(data)
                            # print(input_data_matrix)
                        #else:
                            # print("There is a symbol that are not numeric")
            # input_data_matrix.reshape(self.input_data, self.input_data)
            # for matrix in matrix_list:
                    # One by one.
                    # string_row = data   #self.block_matrix[i*self.block_num+j]
                    # for num in string_row:
                    #     if num.isdigit():
                    #         num_1 = int(num)
                    #         input_data_matrix[j].append(num_1)
                    #         print(input_data_matrix)
                    #     else:
                    #         print("There is a symbol that are not numeric")

            # Define input data.
            # for i in range(0, self.block_num):
            #     # model =
            #     self.input_data_matrix = self.input_data_matrix_+str(i)
                # Note whether the matrix needs to be transposed.

                # The neural network model is invoked for forward propagation
                output_data = Neural_network_coder.Convolutional_encoding_computation(input_data_matrix)
                for a in range(len(output_data)):
                    num_1 = output_data[a]
                    for n in num_1:
                        num_2 = bin(n)
                        print(str(num_2)+" ")
                        # num_3 = iter(num_2)
                        # print(num_3)
                    # string = str(num_1)
                    # self.encoded_block_matrix.append(string)
                    # encode_block_matrix_list = [self.encoded_block_matrix for _ in range(self.block_num)]
                        with open("Output_data.txt", "a") as file:
                            # for encoded_block in self.encoded_block_matrix:
                            file.write(str(num_2) + '\n')



def main():
    block_convolutional_encoder = Neural_network_coder()
    block_convolutional_encoder.input_data = 64
    # block_convolutional_encoder.k = 2
    block_convolutional_encoder.block_num = random.randint(5, 15)

    block_convolutional_encoder.generate_data()
    block_convolutional_encoder.code()


if __name__ == "__main__":
    main()
