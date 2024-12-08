from copy import deepcopy


import numpy as np
import torch


class RandomPlayer:
    """A player that selects moves randomly from available options."""
    
    @staticmethod
    def make_move(board, player):
        """
        Make a random valid move.

        Args:
            board (np.ndarray): The current game board.
            player (int): The current player (1 or -1).

        Returns:
            int: The selected action (index of the board).
        """
        valid_actions = np.where(board == 0)[0]
        return np.random.choice(valid_actions)


class DeepPlayer:
    """A player that uses a trained deep learning model to make moves."""
    
    def __init__(self, model, device='cpu'):
        """
        Initialize the DeepPlayer with a pre-trained model.

        Args:
            model (torch.nn.Module): Pre-trained PyTorch model.
            device (str): The device to load the model on ('cpu' or 'cuda').
        """
        self.device = device
        self.model = deepcopy(model)
        self.model.to(device)
        self.model.eval()
        
    
    def make_move(self, board, player):
        """
        Use the model to decide the best move.

        Args:
            board (np.ndarray): The current game board.
            player (int): The current player (1 or -1).

        Returns:
            int: The selected action (index of the board).
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")
        
        board_scaled = board * player  # Align the board perspective for the player
        board_tensor = torch.tensor(board_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            q_values = self.model(board_tensor)

        available_moves = np.where(board == 0)[0]
        best_move = max(available_moves, key=lambda x: q_values[x].item())  # Select the move with the highest Q-value

        return best_move


class CLIPlayer:
    """A player that makes moves via a command-line interface."""
    
    @staticmethod
    def make_move(board, player):
        """
        Prompt the user to input their move via CLI.

        Args:
            board (np.ndarray): The current game board.
            player (int): The current player (1 or -1).

        Returns:
            int: The selected action (index of the board).
        """
        symbols = {0: '.', 1: 'X', -1: 'O'}
        print("Current Board:")
        for i in range(0, 9, 3):
            print(" ".join(symbols[board[j]] for j in range(i, i + 3)))

        valid_actions = np.where(board == 0)[0]
        while True:
            try:
                move = int(input(f"Player {'X' if player == 1 else 'O'}, choose your move (0-8): "))
                if move in valid_actions:
                    return move
                print(f"Invalid move. Valid moves are: {valid_actions}")
            except ValueError:
                print("Invalid input. Please enter an integer between 0 and 8.")
