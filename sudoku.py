import numpy as np
import time

class Sudoku:
    def __init__(self, sudoku) -> None:
        self.sudoku = np.array(sudoku, dtype=int)

    def solve(self):
        self.backtrack()
        return self.sudoku
    
    def backtrack(self):
        if self.is_solved():
            return True
        row, col = self.find_empty()
        for num in range(1,10):
            if self.is_valid(row, col, num):
                temp_sudoku = self.sudoku.copy()
                self.sudoku[row][col] = num
                if self.backtrack():
                    return True
                self.sudoku = temp_sudoku # Copy the sudoku in this step, so if it needs to get back to n step it conserves the empty cells in the n step.
        return False
    
    def is_solved(self):
        return np.sum(self.sudoku == 0) == 0
    
    def find_empty(self):
        empty_coord = np.where(self.sudoku == 0)
        return empty_coord[0][0], empty_coord[1][0]
    
    def is_valid(self, row, col, num):
        # check if it is in row
        if num in self.sudoku[row, :]:
            return False
        # check if it is in column
        if num in self.sudoku[:,col]:
            return False
        # check if it is in 3x3 grid
        row_start = row - row % 3
        col_start = col - col % 3
        reg_grid = self.sudoku[row_start:(row_start+3),col_start:(col_start+3)]
        if num in reg_grid:
            return False
        # if it is not in row, column or grid
        return True

if __name__ == "__main__":
    board = [
        [7,8,0,4,0,0,1,2,0],
        [6,0,0,0,7,5,0,0,9],
        [0,0,0,6,0,1,0,7,8],
        [0,0,7,0,4,0,2,6,0],
        [0,0,1,0,5,0,9,3,0],
        [9,0,4,0,6,0,0,0,5],
        [0,7,0,3,0,0,0,1,2],
        [1,2,0,0,0,7,4,0,0],
        [0,4,9,2,0,6,0,0,7]
    ]

    sudoku = Sudoku(board)
    time1 = time.time()
    print(sudoku.solve())
    time2 = time.time()
    print(f"Time: {time2-time1} seconds")