import numpy as np


class TicTacToeEnv:
    """
    A Tic-Tac-Toe environment for training reinforcement learning agents.

    Features:
    - Resets the board for a new game.
    - Validates and processes player actions.
    - Checks for win, draw, or ongoing game states.

    Attributes:
        board (np.ndarray): A 1D array representing the current state of the board.
        done (bool): Indicates whether the game is finished.
        win_r (float): Reward for winning the game.
        draw_r (float): Reward for a draw.
        loose_r (float): Reward for losing the game.
        invalid_r (float): Penalty for making an invalid move.
        move_r (float): Reward for a valid move during the game.
        win_indices (np.ndarray): Array of winning combinations for Tic-Tac-Toe.
    """

    def __init__(self, rewards: tuple[float, float, float, float, float] = (1, 0.5, -1, -1, -0.01)):
        """
        Initialize the environment with an empty board, predefined win conditions, and custom rewards.

        Args:
            rewards (tuple): A tuple of rewards for different game outcomes and actions:
                - win_r (float): Reward for winning the game.
                - draw_r (float): Reward for a draw.
                - loose_r (float): Penalty for losing the game.
                - invalid_r (float): Penalty for making an invalid move.
                - move_r (float): Reward for a valid move during the game.
                Defaults to (1, 0.5, -1, -1, -0.01).
        """
        self.board = np.zeros(9, dtype=int)
        self.done = False
        self.win_r = rewards[0]
        self.draw_r = rewards[1]
        self.loose_r = rewards[2]
        self.invalid_r = rewards[3]
        self.move_r = rewards[4]
        self.win_indices = np.array([
            [0, 1, 2],  # Top row
            [3, 4, 5],  # Middle row
            [6, 7, 8],  # Bottom row
            [0, 3, 6],  # Left column
            [1, 4, 7],  # Middle column
            [2, 5, 8],  # Right column
            [0, 4, 8],  # Diagonal top-left to bottom-right
            [2, 4, 6],  # Diagonal top-right to bottom-left
        ])

    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new game.

        Returns:
            np.ndarray: A fresh, empty board (1D array with all zeros).
        """
        self.board = np.zeros(9, dtype=int)
        self.done = False
        return self.board

    def _check_win(self) -> int:
        """
        Check for a winner or draw in the current board state.

        Returns:
            Optional[int]:
                - 1 if player 1 (X) wins.
                - -1 if player -1 (O) wins.
                - 0 if the game is a draw.
                - None if the game is still ongoing.
        """
        for indices in self.win_indices:
            if np.all(self.board[indices] == 1):  # Player 1 wins
                return 1
            if np.all(self.board[indices] == -1):  # Player -1 wins
                return -1

        if np.all(self.board != 0):  # Draw
            return 0

        return None  # Game is ongoing

    def step(self, action: int, player: int) -> tuple[np.ndarray, float, bool]:
        """
        Apply the player's action and update the game state.

        Args:
            action (int): The index (0-8) of the cell where the player wants to play.
            player (int): The player making the move (1 for X, -1 for O).

        Returns:
            tuple:
                - np.ndarray: The updated board (1D array of size 9).
                - float: The reward for the action.
                - bool: Whether the game is finished.
        """
        # check if game was finished by opponent
        if action == -1:
            winner = self._check_win()
            if winner == player:  # Current player wins
                reward = self.win_r
                self.done = True
            elif winner == -player:  # Opponent wins (shouldn't happen if logic is consistent)
                reward = self.loose_r
                self.done = True
            elif winner == 0:  # Draw
                reward = self.draw_r
                self.done = True
            return self.board, reward, self.done
        
        if self.board[action] != 0 or self.done:
            reward = self.invalid_r
            self.done = True
            return self.board, reward, self.done

        self.board[action] = player

        winner = self._check_win()
        if winner == player:  # Current player wins
            reward = self.win_r
            self.done = True
        elif winner == -player:  # Opponent wins (shouldn't happen if logic is consistent)
            reward = self.loose_r
            self.done = True
        elif winner == 0:  # Draw
            reward = self.draw_r
            self.done = True
        else:  # Game is still ongoing
            reward = self.move_r

        return self.board, reward, self.done
