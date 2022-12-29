import game_of_life_interface
import numpy as np
import matplotlib.pyplot as plt


class GameOfLife(game_of_life_interface.GameOfLife):
    def __init__(self, size_of_board, board_start_mode, rules, rle, pattern_position):
        self.size = size_of_board
        if pattern_position == '' or pattern_position == 0:
            self.position = (0, 0)
        else:
            self.position = pattern_position
        self.rle = rle
        if board_start_mode < 1 or board_start_mode > 4:  # default starting position is 1 if start mode <1 or >4
            board_start_mode = 1
        if rle != "":
            board_start_mode = 0  # zeroize the board if rle is inserted
        if board_start_mode == 1:
            self.board = self.start_mode_1()  # call function that create random array with 0.5 p
        elif board_start_mode == 2:
            self.board = self.start_mode_2()  # call function that create random array with probability p = 0.8 for
            # alive state
        elif board_start_mode == 3:
            self.board = self.start_mode_3()  # call function that create random array with probability p= 0.2 for
            # alive state
        elif board_start_mode == 4:
            self.board = self.start_mode_4()  # call to glider gun mode function
        elif rle != '':
            self.board = self.convert_to_matrix()

        rules_lists = rules.split("/")  # rules formatting
        self.born_list = [int(x) for x in list(rules_lists[0][1:])]  # born integer
        self.survives_list = [int(x) for x in list(rules_lists[1][1:])]  # survives integer

    def update(self):
        neighbors_count = (np.pad(self.board, ((0, 0), (1, 0)), mode='wrap')[:, :-1]
                           + np.pad(self.board, ((0, 0), (0, 1)), mode='wrap')[:, 1:]
                           + np.pad(self.board, ((1, 0), (0, 0)), mode='wrap')[:-1, :]
                           + np.pad(self.board, ((0, 1), (0, 0)), mode='wrap')[1:, :]
                           + np.pad(self.board, ((1, 0), (1, 0)), mode='wrap')[:-1, :-1]
                           + np.pad(self.board, ((1, 0), (0, 1)), mode='wrap')[:-1, 1:]
                           + np.pad(self.board, ((0, 1), (1, 0)), mode='wrap')[1:, :-1]
                           + np.pad(self.board, ((0, 1), (0, 1)), mode='wrap')[1:, 1:]) / 255  # create 2D array with
        # Numpy that every cell value is the numbers of the neighborhood in the self.board
        # by counting and dividing by 255

        # run on the board and update it
        for i in range(0, self.board.shape[0]):
            for j in range(0, self.board.shape[0]):
                if self.board[i, j] == 0:  # if a cell needs to be revived - we revive it by the rules
                    if neighbors_count[i, j] in self.born_list:
                        self.board[i, j] = 255
                else:  # if a cell needs to die/survive - we kill/keep it by the rules
                    if neighbors_count[i, j] in self.survives_list:
                        self.board[i, j] = 255
                    else:
                        self.board[i, j] = 0

    def save_board_to_file(self, file_name):
        plt.imsave(file_name, self.board)  # save the board to file

    def display_board(self):
        return plt.show(plt.matshow(self.board))  # display the board

    def return_board(self):
        return self.board.tolist()  # return as a list

    def transform_rle_to_matrix(self, rle):  # Transform rle to matrix with size compatible to the given pattern
        # without importance to the board size
        row = []
        matrix = []
        if rle[0] == 'o':
            row[0] = 255
        elif rle[0] == 'b':
            row[0] = 0
        for rle_index in range(1, len(rle)):
            if rle[rle_index] == 'o':
                if rle[rle_index - 1].isdigit() and rle[rle_index - 2].isdigit():
                    for counter in range(0, int(rle[rle_index - 2:rle_index])):
                        row.append(255)
                elif rle[rle_index - 1].isdigit():
                    for counter in range(0, int(rle[rle_index - 1])):
                        row.append(255)
                else:
                    row.append(255)
            elif rle[rle_index] == 'b':
                if rle[rle_index - 1].isdigit() and rle[rle_index - 2].isdigit():
                    for counter in range(0, int(rle[rle_index - 2:rle_index])):
                        row.append(0);
                elif rle[rle_index - 1].isdigit():
                    for counter in range(0, int(rle[rle_index - 1])):
                        row.append(0);
                else:
                    row.append(0);

            elif rle[rle_index] == '$' and rle[rle_index - 1].isdigit() and rle[rle_index - 2].isdigit():
                zeros_list = []
                matrix.append(row)
                row = []
                for i in range(0, len(matrix[0])):
                    zeros_list.append(0)
                for j in range(0, int(rle[rle_index - 2:rle_index]) - 1):
                    matrix.append(zeros_list)
            elif rle[rle_index] == '$' and rle[rle_index - 1].isdigit():
                zeros_list = []
                matrix.append(row)
                row = []
                for i in range(0, len(matrix[0])):
                    zeros_list.append(0)
                for j in range(0, int(rle[rle_index - 1]) - 1):
                    matrix.append(zeros_list)
            elif rle[rle_index] == '$':
                matrix.append(row)
                row = []
            elif rle[rle_index] == '!':
                for i in range(len(row), len(matrix[0])):
                    row.append(0)
                matrix.append(row)
        MaxLength = max(len(i) for i in matrix)  # Fill up all raw with zeros if needed
        for j in range(0, len(matrix)):
            while len(matrix[j]) < MaxLength:
                matrix[j].append(0)

        return matrix  # return the matrix as a two dimensional list

    def convert_to_matrix(self):
        matrix = np.zeros((self.size, self.size),
                          dtype=int)  # rle to matrix function (gets the rle and return the full board)
        x = self.position[1]
        y = self.position[0]
        if self.rle[0] == 'o':
            matrix[y, x] = 255
            x += 1
        elif self.rle[0] == 'b':
            matrix[y, x] = 0
            x += 1
        for rle_index in range(1, len(self.rle)):
            if self.rle[rle_index] == 'o':
                if self.rle[rle_index - 1].isdigit() and self.rle[rle_index - 2].isdigit():
                    matrix[y, x:x + int(self.rle[rle_index - 2:rle_index])] = 255
                    x += int(self.rle[rle_index - 2:rle_index])
                elif self.rle[rle_index - 1].isdigit():
                    matrix[y, x:x + int(self.rle[rle_index - 1])] = 255
                    x += int(self.rle[rle_index - 1])
                else:
                    matrix[y, x] = 255
                    x += 1
            elif self.rle[rle_index] == 'b':
                if self.rle[rle_index - 1].isdigit() and self.rle[rle_index - 2].isdigit():
                    x += int(self.rle[rle_index - 2:rle_index])
                elif self.rle[rle_index - 1].isdigit():
                    x += int(self.rle[rle_index - 1])
                else:
                    x += 1
            elif self.rle[rle_index] == '$' and self.rle[rle_index - 1].isdigit() and self.rle[rle_index - 2].isdigit():
                y += int(self.rle[rle_index - 2:rle_index])
                x = self.position[1]
            elif self.rle[rle_index] == '$' and self.rle[rle_index - 1].isdigit():
                y += int(self.rle[rle_index - 1])
                x = self.position[1]
            elif self.rle[rle_index] == '$':
                y += 1
                x = self.position[1]
        return matrix  # return the matrix by rle code with full size as given by the Size_of_board

    def start_mode_1(self):
        return np.random.choice([0, 255], (self.size, self.size), p=[0.5, 0.5])

    def start_mode_2(self):
        return np.random.choice([0, 255], (self.size, self.size), p=[0.2, 0.8])

    def start_mode_3(self):
        return np.random.choice([0, 255], (self.size, self.size), p=[0.8, 0.2])

    def start_mode_4(self):
        board_mode_4 = np.zeros((self.size, self.size), dtype=int)  # create zero board
        board_mode_4[14, 11] = 255
        board_mode_4[15, 11] = 255
        board_mode_4[14, 10] = 255
        board_mode_4[15, 10] = 255
        board_mode_4[14, 20] = 255
        board_mode_4[15, 20] = 255
        board_mode_4[16, 20] = 255
        board_mode_4[13, 21] = 255
        board_mode_4[17, 21] = 255
        board_mode_4[12, 22] = 255
        board_mode_4[18, 22] = 255
        board_mode_4[12, 23] = 255
        board_mode_4[18, 23] = 255
        board_mode_4[15, 24] = 255
        board_mode_4[13, 25] = 255
        board_mode_4[17, 25] = 255
        board_mode_4[14:17, 26] = 255
        board_mode_4[15, 27] = 255
        board_mode_4[12:15, 30] = 255
        board_mode_4[12:15, 31] = 255
        board_mode_4[11, 32] = 255
        board_mode_4[15, 32] = 255
        board_mode_4[10:12, 34] = 255
        board_mode_4[15:17, 34] = 255
        board_mode_4[12:14, 44] = 255
        board_mode_4[12:14, 45] = 255
        return board_mode_4


if __name__ == '__main__':  # You should keep this line for our auto-grading code.
    print('write your tests here')  # don't forget to indent your code here!
