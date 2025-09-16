import random
import copy
import heapq
#import requests  # Needed if fetching Sudoku from an API
import time

# ========================
# Step 1: Load Sudoku Puzzle
# ========================

def load_sudoku():
    """
    Returns a randomly chosen Sudoku puzzle from predefined puzzles.
    """
    puzzles = [
        # Puzzle 1
        [
            [5, 3, 0, 0, 7, 0, 0, 0, 0], 
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ],
        # Puzzle 2
        [
            [0, 0, 3, 0, 2, 0, 6, 0, 0], 
            [9, 0, 0, 3, 0, 5, 0, 0, 1],
            [0, 0, 1, 8, 0, 6, 4, 0, 0],
            [0, 0, 8, 1, 0, 2, 9, 0, 0],
            [7, 0, 0, 0, 0, 0, 0, 0, 8],
            [0, 0, 6, 7, 0, 8, 2, 0, 0],
            [0, 0, 2, 6, 0, 9, 5, 0, 0],
            [8, 0, 0, 2, 0, 3, 0, 0, 9],
            [0, 0, 5, 0, 1, 0, 3, 0, 0]
        ]
    ]
    return random.choice(puzzles)

def load_sudoku_from_file(filename="sudoku_puzzles.txt"):
    """
    Loads Sudoku puzzles from a text file and returns a random one.
    """
    # Write your code here
    with open(filename, 'r') as file:
        lines = file.readlines()
        puzzles = []
        individual_puzzle = []

        for line in lines:
            line = line.strip()  # Gets rid of white space
            # Skips lines used to separate puzzles
            if line == "":
                continue
            # Pieces together a single puzzle line-by-line
            list_of_ints = [ int(char) for char in line if char != " " ]
            individual_puzzle.append(list_of_ints)
            # Checks to see if an individual puzzle has been fully compiled
            if len(individual_puzzle) == 9:
                puzzles.append(individual_puzzle)  # Compiles list of puzzles
                individual_puzzle = []  # Emptied to store next puzzle
    
    # Returns random puzzle if any exist
    if puzzles:
        return random.choice(puzzles)
    else:
        return None


# ========================
# Step 2: Validity Check Functions
# ========================

def is_valid(grid, row, col, num):
    """
    Returns True if placing num at (row, col) is valid according to Sudoku rules.
    """
    # For a move to be considered valid, all the num values must not already exist in the provided row, column, and 3x3 subgrid.
    # If any one of the conditional checks returns True, then placing the num at the provided (row, column) must be invalid.
    return not(in_row(grid, row, num) | in_col(grid, col, num) | in_box(grid, row, col, num))


def in_row(grid, row, num):
    """
    Checks if num is already present in the row.
    """
    # Write your code here
    return num in grid[row]


def in_col(grid, col, num):
    """
    Checks if num is already present in the column.
    """
    # Write your code here
    return any(num == row[col] for row in grid)


def in_box(grid, row, col, num):
    """
    Checks if num is already present in the 3x3 subgrid.
    """
    # Write your code here
    # Assigns the starting row and column numbers for the 3x3 subgrid of concern
    box_row = (row // 3) * 3
    box_col = (col // 3) * 3

    # Obtains all non-zero values in the 3x3 subgrid
    subgrid_values = { 
        grid[r][c]
        for r in range(box_row, box_row + 3)
        for c in range(box_col, box_col + 3)}  - {0}

    # Checks if the given num value exists in the 3x3 subgrid
    return any( num == subgrid_val for subgrid_val in subgrid_values )


# ========================
# Step 3: Successor Function
# ========================

def get_successors_at_cell(grid, row, col):
    """
    Returns all valid states for the given empty cell.
    """
    valid_states = []
    for digit in range(1, 10):
        grid_copy = copy.deepcopy(grid)  # Creates an independent copy of the grid for modification
        # Checks if using a given digit to plug the empty cell would be valid or not
        if is_valid(grid_copy, row, col, digit):
            # If valid, we fill in the space with the given digit and add it to the list of valid successor states
            grid_copy[row][col] = digit
            valid_states.append(grid_copy)
    return valid_states


def get_successors(grid, A_search = False, MCV = True):
    """
    Finds the first empty cell (for non-A* algorithms) and returns all valid states with one number filled.
    """
    # Write your code here

    # Uses a generator to produce the locations of all empty cells and fetches the location of the first
    row, col = next(  # Unpacks row and column values of next location housing an empty cell
    (row, col)  # Stores the location of cell on Sudoku grid as a tuple
    for row in range(0, 9)
    for col in range(0, 9)
    if grid[row][col] == 0 )  # Only stores the locations of empty cells

    if A_search == False:
        valid_states = get_successors_at_cell(grid, row, col)
    else:
        if MCV == True:
            # Successor states for most-constrained cell
            valid_states, _, _, _ = most_constrained_variable(grid)
        else:
            # Successor states for least-constrained cell
            valid_states, _ = least_constraining_value(grid, row, col)
    
    return valid_states


# ========================
# Step 4: Goal Check
# ========================

def is_goal(grid):
    """
    Returns True if the grid is a valid solved Sudoku.
    """
    # Write your code here
    for row in grid:  # Iterates through each row
        if has_duplicates(row):
            return False

    for col in range(9):  # Iterates through each column
        # Generates list of the column-values and checks it for duplicates
        if has_duplicates([grid[row][col] for row in range(9)]):
            return False

    # Iterates through each 3x3 subgrid
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            # Generates list of values for each subgrid
            box = {
                grid[row][col]
                for row in range(box_row, box_row + 3)
                for col in range(box_col, box_col + 3)
            } - {0}
            # Checks for duplicates in a given 3x3 subgrid
            if has_duplicates(box):
                return False

    # Returns True if and only if no empty cells exist in the grid
    return all(0 not in row for row in grid)


def has_duplicates(lst):
    """
    Returns True if there are duplicate numbers (excluding 0s).
    """
    # Write your code here
    seen = {}
    
    # Iterates through each digit in the provided list
    for num in lst:
        # Checks if a digit is not an empty cell
        if num != 0:
            # If a value is returned from the fetch, then a duplicate is present
            if seen.get(num):
                return True
            # Otherwise, we add the unique digit for reference
            seen[num] = True
            
    # The provided list must not have any duplicates
    return False
    

# ========================
# Step 5: A* Search Algorithm
# ========================

def heuristic(grid):
    """
    Returns the number of empty cells (lower values are better).
    """
    # Write your code here
    # Counts the number of empty spaces by taking the summation of True (1) and False (0) values
    return sum( grid[row][col] == 0 for row in range(0, 9) for col in range(0, 9) )                   


def most_constrained_variable(grid):
    """
    Returns the successor states, number of legal moves (row, col) of the cell.
    """
    best_cell = None
    min_options = 10  # Keeps track of how many digits can be substituted for the first empty space
    for row in range(0, 9):
        for col in range(0, 9):
            if grid[row][col] == 0:
                # Removes duplicate values and obtains list of non-zero values that can be used to make a valid move
                legal_moves = [ digit for digit in range(1, 10) if is_valid(grid, row, col, digit) ]
                # Grabs the number of legal moves 
                num_legal_moves = len(legal_moves)
                # Updates the value associated with the most constrained cell found
                if min_options > num_legal_moves:
                    min_options = num_legal_moves
                    best_cell = (row, col)
                # Acts as an early return to cut down on unnecessary iterations and time
                if min_options == 1:
                    break
        # Ensures that we break out of the second outer loop when the best cell is already found
        if min_options == 1:
            break

    # A most constrained cell exists
    if best_cell:
        row, col = best_cell
        return get_successors_at_cell(grid, row, col), min_options, row, col
    # No constrained cells exist
    else:
        return [], 0, 0, 0
    
    
def least_constraining_value(grid, row, col):
    """
    Given a cell at (row, col), returns a sorted list of successors and values that constrain future moves the least.
    """
    # Get all legal moves for the current cell
    legal_moves = [ digit for digit in range(1, 10) if is_valid(grid, row, col, digit) ]

    # Track constraint scores for each move
    move_constraints = {}

    # Find successors of the current cell
    successors = get_successors_at_cell(grid, row, col)

    # Calculate the constraint score for each legal move
    for digit in legal_moves:
        constrained_count = 0
        # Tallies up number of invalid moves using a digit for a given cell
        if grid[row][col] == 0 and not is_valid(grid, row, col, digit):
            constrained_count += 1
        # Adds key-value pair associating each digit with the number of invalid moves t
        move_constraints[digit] = constrained_count
        
    # Sorts moves based on the least constraint
    sorted_moves = sorted(move_constraints.items(), key=lambda item: item[1])
    # Extracts least constraining values
    least_constraining_moves = [ digit for digit, _ in sorted_moves ]

    return successors, least_constraining_moves


def hill_climbing(grid):
    pass


def genetic_algorithm(grid):
    pass


def a_star_sudoku(grid, mcv):
    """
    Solves Sudoku using A* search.
    """
    steps = 0
    start_time = time.time()
    # Priority queue to keep track of the grid state and its associated heuristic as (heuristic_value, grid)
    pq = []
    heapq.heappush(pq, (heuristic(grid), grid))

    while pq:
        # Retrieves the current state of the next-best state in the priority queue
        _, current_state = heapq.heappop(pq)

        # Check to see if current state is a solution
        if is_goal(current_state):
            print(f'Summary of A* Search:\n\tNo. of Steps: {steps}\n\tTime Elapsed: {time.time() - start_time:.3f} secs')
            return current_state

        successors = get_successors(current_state, A_search = True, MCV = mcv)

        # Calculates heuristic for each successor, and adds (heuristic, state) to priority queue
        for successor_grid in successors:
            heuristic_value = heuristic(successor_grid)
            heapq.heappush(pq, ( heuristic_value + 1, successor_grid ))
            
        steps += 1

    

# ========================
# Step 6: Simulated Annealing (SA)
# ========================
def simulated_annealing(grid, max_iter=10_000, temp=1.0, cooling_rate=0.99):
    """
    Solves Sudoku using SA search.
    """
    # Write your code here
    start_time = time.time()
    current_state = grid
    for iter in range(1, max_iter + 1):        
        temp *= cooling_rate  # Cools down temperature over course of iterations
        valid_states = get_successors(current_state)

        # True when SA reaches a deadend
        if valid_states == []:
            print(f'Summary of Simulated Annealing:\n\tNo. of Steps: {iter}\n\tTime Elapsed: {time.time() - start_time:.3f} secs')
            return current_state
        
        # Randomly selects a successor state for consideration calculates the energy gain from selecting it
        next_state = random.choice(valid_states)
        energy_difference = heuristic(next_state) - heuristic(current_state)

        # True if the successor state is an improvement based on heuristic, confirm as succeeding state
        if energy_difference > 0:
            current_state = next_state
        # Energy gain is negative, the bad state may or may not be confirmed with some probability
        else:
            import math
            random_value = random.random()
            probability = math.e ** (energy_difference / temp)
            if random_value < probability:
                current_state = next_state
        
        # Returns the last state if (1) maximum # of iterations is hit, (2) goal state is found, or (3) the temperature reaches "zero"
        if (iter == max_iter) or (is_goal(current_state)) or (temp < 1e-6):
            print(f'Summary of Simulated Annealing:\n\tNo. of Steps: {iter}\n\tTime Elapsed: {time.time() - start_time:.3f} secs')
            return current_state
    # Max iterations are reached and state has "cooled and solidified"
    return current_state


def simulated_annealing_with_backtracking(grid, max_iter=10_000, temp=1.0, cooling_rate=0.99):
    """
    Solves Sudoku using SA search.
    """
    # Write your code here
    start_time = time.time()
    current_state = copy.deepcopy(grid)

    for iter in range(1, max_iter + 1):
        next_state = current_state
        temp *= cooling_rate  # Cools down the temperature over time
        
        # Incorporates backtracking by finding the most constrained variable
        _, _, row, col = most_constrained_variable(current_state)
        
        # If the puzzle is solved, return the solution
        if is_goal(current_state):
            print(f'Summary of Simulated Annealing w/ Backtracking:'
                  f'\n\tNo. of Steps: {iter}'
                  f'\n\tTime Elapsed: {time.time() - start_time:.3f} secs')
            return current_state
        
        # Generates a list of valid digits for the current cell
        valid_digits = [ digit for digit in range(1, 10) if is_valid(current_state, row, col, digit) ]
        
        # If no valid digits exist, returns the current state (dead-end)
        if not valid_digits:
            return current_state

        # Chooses the next digit to fill successor state with
        chosen_digit = random.choice(valid_digits)
        next_state[row][col] = chosen_digit
        
        # Evaluates the "energy difference" to determine whether moving to a fiven 
        energy_difference = heuristic(next_state) - heuristic(current_state)
        
        # Accepts the next state based on energy difference and temperature
        if energy_difference > 0:  # If energy difference is positive, accept the move
            current_state[row][col] = chosen_digit
        # If energy loss occurs, accept the move with some probability
        else:
            import math
            random_value = random.random()
            probability = math.e ** (energy_difference / temp)
            if random_value < probability:
                current_state[row][col] = chosen_digit
        
        # Halts if the maximum iterations, goal state, or temperature threshold is reached
        if (iter == max_iter) or (is_goal(current_state)) or (temp < 1e-6):
            print(f'Summary of Simulated Annealing w/ Backtracking:'
                  f'\n\tNo. of Steps: {iter}'
                  f'\n\tTime Elapsed: {time.time() - start_time:.3f} secs')
            return current_state

    # Max iterations are reached and state has "cooled and solidified"
    return current_state
    

def depth_first_search(grid):
    """
    Solves Sudoku using DFS.
    """
    steps = 0
    start_time = time.time()
    stack = []
    stack.append(grid)

    while stack:
        # Retrieves the last successor state (treats the list like a stack)
        current_state = stack.pop()
        # Check to see if current state is equivalent to the goal state
        if is_goal(current_state):
            print(f'Summary of Depth-First Search:'
                  f'\n\tNo. of Steps: {steps}'
                  f'\n\tTime Elapsed: {time.time() - start_time:.3f} secs')
            return current_state
        # Adds each successor to the stack
        for successor_state in get_successors(current_state):
            stack.append(successor_state)

        steps += 1

    print_sudoku(current_state)
    #return None  # No Solution


def breadth_first_search(grid):
    """
    Solves Sudoku using BFS.
    """
    steps = 0
    start_time = time.time()
    queue = []
    queue.append(grid)

    while queue:
        # Removes and assigns the first successor state (treats the list like a FIFO queue)
        current_state = queue.pop(0)

        # Check to see if current state is equivalent to the goal state
        if is_goal(current_state):
            print(f'Summary of Breadth-First Search:\n\tNo. of Steps: {steps}\n\tTime Elapsed: {time.time() - start_time:.3f} secs')
            return current_state
        
        # Adds each successor to queue
        for successor_state in get_successors(current_state):
            queue.append(successor_state)

        steps += 1

    print_sudoku(current_state)        
    #return None  # No solution

# ========================
# Step 7: Running and Testing
# ========================

def print_sudoku(grid):
    """
    Prints the Sudoku grid in a readable format.
    """
    for row in grid:
        print(" ".join(str(num) if num != 0 else "." for num in row))
    print("* * * * * * * * * * * * * * * * * * *")

# Load a Sudoku puzzle (Choose one option)
# sudoku_grid = load_sudoku()  # Option 1: Hardcoded
sudoku_grid = load_sudoku_from_file()  # Option 2: From file


print("Loaded Sudoku Puzzle:")
print_sudoku(sudoku_grid)

# Solve using A* w/ MCV
solved_grid = a_star_sudoku(sudoku_grid, mcv = True)

if solved_grid:
    print("\nSolved Sudoku (A* w/ MCV):")
    print_sudoku(solved_grid)
else:
    print("No solution found.")

# Solve using A* w/ LCV
solved_grid = a_star_sudoku(sudoku_grid, mcv = False)

if solved_grid:
    print("\nSolved Sudoku (A* w/ LCV):")
    print_sudoku(solved_grid)
else:
    print("No solution found.")

# Solve using SA
solved_sa = simulated_annealing(sudoku_grid)

if solved_sa:
    print("\nSolved Sudoku (Simulated Annealing):")
    print_sudoku(solved_sa)
else:
    print("No solution found.")

# Solve using SA w/ backtracking
solved_sa_with_bt = simulated_annealing_with_backtracking(sudoku_grid)

if solved_sa_with_bt:
    print("\nSolved Sudoku (Simulated Annealing):")
    print_sudoku(solved_sa_with_bt)
else:
    print("No solution found.")

# Solve using DFS
solved_dfs = depth_first_search(sudoku_grid)

if solved_dfs:
    print("\nSolved Sudoku (Depth-First Search):")
    print_sudoku(solved_dfs)
else:
    print("No solution found.")

# Solve using BFS
solved_bfs = breadth_first_search(sudoku_grid)

if solved_bfs:
    print("\nSolved Sudoku (Breadth-First Search):")
    print_sudoku(solved_bfs)
else:
    print("No solution found.")

