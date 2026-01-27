class Solution:
    def largestMagicSquare(self, grid: list[list[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        
        for square_size in range(min(rows, cols), 1, -1):
            for i in range(rows - square_size + 1):
                for j in range(cols - square_size + 1):
                    target = sum(grid[i][j:j+square_size])
                    magic = True
                    
                    # Check all rows
                    for r in range(i, i + square_size):
                        if sum(grid[r][j:j+square_size]) != target:
                            magic = False
                            break
                    if not magic:
                        continue
                    
                    # Check all columns
                    for c in range(j, j + square_size):
                        if sum(grid[r][c] for r in range(i, i+square_size)) != target:
                            magic = False
                            break
                    if not magic:
                        continue
                    
                    # Check diagonals
                    if sum(grid[i+k][j+k] for k in range(square_size)) != target:
                        continue
                    if sum(grid[i+k][j+square_size-1-k] for k in range(square_size)) != target:
                        continue
                    
                    # Found a magic square
                    return square_size
        
        return 1

