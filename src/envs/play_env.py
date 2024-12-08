import numpy as np


class TicTacToeGame:
    """
    A simple Tic-Tac-Toe game engine.

    Features:
    - Maintains the game board and enforces the rules.
    - Allows two players (agents) to play against each other.
    - Supports displaying the board for debugging or visualization.

    Attributes:
        board (np.ndarray): The current state of the board (1D array of size 9).
        verbose (bool): Whether to display the board after each move.
        win_indices (np.ndarray): Predefined winning combinations for Tic-Tac-Toe.
    """

    def __init__(self):
        """
        Initialize the game board and setup win conditions.
        """
        self.board = np.zeros(9, dtype=int)
        self.verbose = True
        self.win_indices = np.array([
            [0, 1, 2],  # Horizontal rows
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],  # Vertical columns
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],  # Diagonals
            [2, 4, 6],
        ])

    def _display_board(self):
        """
        Display the current state of the board using symbols.

        Symbols:
            - "." for empty cells
            - "X" for player 1
            - "O" for player -1
        """
        if not self.verbose:
            return
        symbols = {0: ".", 1: "X", -1: "O"}
        for i in range(0, 9, 3):
            print(" ".join(symbols[self.board[j]] for j in range(i, i + 3)))
        print()

    def _reset_game(self):
        """
        Reset the game board to its initial empty state.
        """
        self.board = np.zeros(9, dtype=int)

    def _check_winner(self):
        """
        Check for a winner or draw.

        Returns:
            int: 1 if player X wins, -1 if player O wins, 0 if it's a draw, None otherwise.
        """
        if np.any(np.all(self.board[self.win_indices] == np.ones(3), axis=1)):
            return 1  # Player X wins
        if np.any(np.all(self.board[self.win_indices] == -np.ones(3), axis=1)):
            return -1  # Player O wins
        if np.all(self.board != 0):  # No empty cells, draw
            return 0
        return None  # Game continues

    def play(self, player_x, player_o, verbose=False):
        """
        Play a game between two players (agents).

        Args:
            player_x (object): An object with a `make_move` method for player X.
            player_o (object): An object with a `make_move` method for player O.
            verbose (bool): Whether to display the board after each move.

        Returns:
            int: 1 if player X wins, -1 if player O wins, 0 if it's a draw.
        """
        self._reset_game()
        self.verbose = verbose
        current_player = 1  # Player X starts
        players = {1: player_x, -1: player_o}
        self._display_board()

        while True:
            # Ask the current player to make a move
            move = players[current_player].make_move(self.board.copy(), current_player)
            self.board[move] = current_player
            self._display_board()

            # Check for a winner or draw
            winner = self._check_winner()
            if winner is not None:
                return winner

            # Switch to the other player
            current_player *= -1