def is_safe(board, row, col, n):
    
    for i in range(col):                                        # queens presence in same row
        if board[row][i] == 1:
            return False
  
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):    # queens presence in upper diagonal on the left side
        if board[i][j] == 1:
            return False
 
    for i, j in zip(range(row, n, 1), range(col, -1, -1)):      # queens presence in lower diagonal on the left side
        if board[i][j] == 1:
            return False

    return True

def print_board(board):
    for row in board:
        print(" ".join(map(str, row)))
    print()

def solve_n_queens(board, col, n):
    if col == n:
        print_board(board)
        return

    for i in range(n):
        if is_safe(board, i, col, n):
            board[i][col] = 1
            solve_n_queens(board, col + 1, n)
            board[i][col] = 0  # Backtrack

if __name__ == "__main__":
    n = 8  
    board = [[0] * n for _ in range(n)]

    board[0][1] = 1

    print("Initial board with the first queen placed:")
    print_board(board)

    print("Solutions for the n-Queens problem:")
    solve_n_queens(board, 1, n)
