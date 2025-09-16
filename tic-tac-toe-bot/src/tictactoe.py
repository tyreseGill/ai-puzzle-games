import math
from copy import deepcopy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # Determines how many spaces have been occupied on the board
    filled_spaces = [ board[row][col] for row in range(3) for col in range(3) if board[row][col] != EMPTY]
    num_spaces_filled = len(filled_spaces)

    # Since player X plays first, player O must play during every even turn
    if num_spaces_filled % 2 == 1:
        return "O"
    # Player X plays during every odd turn
    else:
        return "X"
    

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    current_player = player(board)

    if action not in actions(board):
        raise Exception(f"Player {current_player} attempted to perform an invalid move!")
    
    row, col = action
    updated_board = deepcopy(board)
    updated_board[row][col] = current_player

    return updated_board



def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Stores each line in the 3x3 board that can result in a win
    winning_lines = []

    # Adds the unique occupants for each row
    for row in range(3):
        row_occupants = { board[row][col] for col in range(3) }
        winning_lines.append(row_occupants)
    
    # Adds the unique occupants for each column
    for col in range(3):
        col_occupants = { board[row][col] for row in range(3) }
        winning_lines.append(col_occupants)
    
    # Adds the unique occupants for each diagonal line
    diagonal_occupants_1 = { board[index][index] for index in range(3) }
    diagonal_occupants_2 = { board[2 - index][index] for index in range(3) }
    winning_lines.append(diagonal_occupants_1)
    winning_lines.append(diagonal_occupants_2)
    
    # Checks each winning line for the presence of a three in a row by a player
    for winning_line in winning_lines:
        # We only check the winning lines that consist of the same three occupants
        if len(winning_line) == 1:
            occupant = winning_line.pop()
            if occupant == "X" or occupant == "O":
                return occupant
    
    # There must be no winner in for the current board configuration
    return None
    


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # Neither player has a three in a row
    if utility(board) == 0:
        # There still exist unfilled spaces
        if actions(board):
            return False
        # No moves are left
        else:
            return True
    # One of the players must have a three in a row
    else:
        return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    winning_player = winner(board)
    # Player X wins
    if winning_player == "X":
        return 1
    # Player O wins
    elif winning_player == "O":
        return -1
    # Neither player wins
    else:
        return 0


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    empty_spaces = { (row, col)  for row in range(3) for col in range(3) if board[row][col] == EMPTY }
    return empty_spaces


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    # A helper function that returns the score and its associated action
    def minimax_value(board, alpha, beta):
        # Determines next player
        current_player = player(board)

        best_action = None  # Starting at the initial state, no action can be classified as the best

        # Returns the final score associated with the end state
        if terminal(board):
            return utility(board), []
        
        # Continues exploring the game tree to look for the best score and action
        # Assigns initial value of best_score based on minimizing/maximizing behavior
        best_score = float("-inf") if current_player == "X" else float("inf")
        for action in actions(board):  # Iterates through each (i, j) move
            new_board = result(board, action)  # Assigns the state created from making a specific action  
            new_score, _ = minimax_value(new_board, alpha, beta)  # Recursively calls itself until it hits terminal state and returns score

            # Calculates the best score and action based on the current player
            if current_player == "X":  # Maximizer
                # Updates score and action if better from maximizer perspective
                if new_score > best_score:
                    best_score = new_score
                    best_action = action
                alpha = max(best_score, alpha)  # Updates alpha if score is higher
                if alpha >= beta:  # Prunes remaining branch by breaking loop
                    break
            else:  # Minimizer
                # Updates score and action if better from minimizer perspective
                if new_score < best_score:
                    best_score = new_score
                    best_action = action
                beta = min(best_score, beta)  # Updates beta if score is lower
                if alpha >= beta: # Prunes remaining branches by breaking loop
                    break

        # After evaluating all possible actions (or returning early from pruning), we obtain
        # the best score and the action associated with that score
        return best_score, best_action
    
    # The pygame logic is reliant only on the move (i, j) made by Minimax, extract and return just that
    _, action = minimax_value(board, alpha=float("-inf"), beta=float("inf"))

    return action


