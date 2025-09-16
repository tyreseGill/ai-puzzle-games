import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        # If we know X neighbors are mines and there exist X unknown neighbors, those neighbors must be mines
        if self.count == len(self.cells):
            return self.cells
        # We cannot infer if any of the neighboring cells are mines
        else:
            return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        # None of the neighboring cells are mines
        if self.count == 0:
            return self.cells
        # At least one of the neighbors is a mine, we cannot deduce which neighbors to be safe or not
        else:
            return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)


    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        # Adds a clicked cell and marks it as safe
        self.moves_made.add(cell)
        self.mark_safe(cell)

        # Generates a list of the current cell's neighboring cells
        row, col = cell
        rows = [ r for r in [row - 1, row, row + 1] if r in range(self.height) ]
        cols = [ c for c in [col - 1, col, col + 1] if c in range(self.height) ]
        # Adds every cell within a one cell vicinity of the examined cell (excluding itself)
        neighbors = set(
            (r, c) for r,c in itertools.product(rows, cols)
            if (r,c) != cell
        )

        # Creates new fact based on current cell and add it to knowledge base
        fact = Sentence(neighbors, count)
        self.knowledge.append(
            fact
        )

        changed = True  # Flag to indicate a change to the knowledge base and deduction

        # So long as changes occur in the AI knowledge base, it may be possible to make further inferences
        while changed:
            changed = False  # Initially set to false. If changed, then deduction occured
            # Iterates through knowledge base to infer the status of each fact's unknown neighboring cells
            for fact in self.knowledge:
                # Ensures newly registered safe cells are marked
                safes = fact.known_safes()
                if safes:
                    changed = True  # New inferences may be derived
                    for cell in safes.copy():
                        self.mark_safe(cell)
                # Ensures newly registered mines are marked
                mines = fact.known_mines()
                if mines:
                    changed = True  # New inferences may be derived
                    for cell in mines.copy():
                        self.mark_mine(cell) 
            
            # Iterates through knowledge base to see if new inferences can be drawn from existing facts
            for fact1 in self.knowledge.copy():
                for fact2 in self.knowledge.copy():
                    # No new inference can be derived from the same fact
                    if fact1 == fact2:
                        continue
                    # If fact 1 is a subset of fact 2, a new inference is created to replace them
                    if fact2.cells.issubset(fact1.cells):
                        changed = True  # New inferences may be derived from the new inference
                        # Creates and adds new inference based on existing facts
                        self.knowledge.append( 
                            Sentence(fact1.cells - fact2.cells,
                                    fact1.count - fact2.count)
                                    )
                        # Removes old inferences
                        if fact1 in self.knowledge:
                            self.knowledge.remove(fact1)
                        self.knowledge.remove(fact2)

            # Removes empty facts in the knowledge base as cells are marked as safe/mines
            for fact in self.knowledge.copy():
                if fact.cells == set():
                    self.knowledge.remove(fact)
            
            # Performs cleanup and ensures that redundant deductions don't occur
            for safe_cell in self.safes:
                for fact in self.knowledge.copy():
                    if safe_cell in fact.cells:
                        changed = True  # New inferences may be derived
                        fact.mark_safe(safe_cell)
            for mine in self.mines:
                for fact in self.knowledge.copy():
                    if mine in fact.cells:
                        changed = True  # New inferences may be derived
                        fact.mark_mine(mine)


    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        # Safe moves yet to be made
        poss_moves = self.safes - self.moves_made

        # There exists at least one "safe" move
        if poss_moves:
            return list(poss_moves)[0]
        # There are no safe moves left to choose from
        else:
            return None
        
    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        # Cells left to choose from
        cells_left = set((row, col) for row in range(self.height) for col in range(self.width)) - self.moves_made - self.mines

        # Prevents the window from crashing/closing out
        if len(cells_left) == 0:
            print("You won!")
            exit()

        # Cells that appear in facts with their risk counts
        risky_cells = { cell: fact.count for fact in self.knowledge for cell in fact.cells }

        # Cells with no current risk knowledge (uninformed cells)
        uninformed_cells = [cell for cell in cells_left if cell not in risky_cells]

        # Forces AI to prefer uninformed cells over cells known to be within proximity to a mine
        if uninformed_cells:
            choice = random.choice(uninformed_cells)
            return choice

        # If no uninformed moves left, picks least risky move
        if risky_cells:
            ordered_risky_cells = [k for k, v in sorted(risky_cells.items(), key=lambda item: item[1])]
            choice = ordered_risky_cells[0]
            return choice

        # Fallback: pick a random move from what's left (shouldn't happen)
        choice = random.choice(list(cells_left))
        return choice