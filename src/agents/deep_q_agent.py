import random
import copy
from collections import deque


import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


class DeepQAgent:
    """
    A Deep Q-Learning Agent for reinforcement learning tasks.
    
    Attributes:
        model (torch.nn.Module): The primary model used for Q-value predictions.
        target_model (torch.nn.Module): The target model for stable Q-value updates.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        memory (collections.deque): Experience replay buffer.
        loss_hist (list): History of loss values for tracking training performance.
        device (torch.device): The device used for computations (CPU or GPU).
        action_size (int): Number of possible actions.
        mask_invalid (bool): Whether to mask invalid actions (e.g., occupied cells in Tic-Tac-Toe).
        lr (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Probability of taking a random action (exploration rate).
        epsilon_min (float): Minimum value of epsilon for exploration.
        epsilon_decay (float): Decay rate for epsilon per step.
        memory_size (int): Size of the memory buffer for experience replay.
    """

    def __init__(self,
                model: torch.nn.Module,
                mask_invalid: bool = True,
                lr: float = 1e-5,
                gamma: float = 0.95,
                epsilon: float = 1.0,
                epsilon_min: float = 0.01,
                epsilon_decay: float = 0.997,
                memory_size: int = 10000,
                device: torch.device = None):
        """
        Initialize the DeepQAgent.

        Args:
            model (torch.nn.Module): The neural network model for Q-value predictions.
            mask_invalid (bool, optional): Whether to mask invalid actions. Defaults to True.
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-5.
            gamma (float, optional): Discount factor for future rewards. Defaults to 0.95.
            epsilon (float, optional): Initial exploration rate. Defaults to 1.0.
            epsilon_min (float, optional): Minimum exploration rate. Defaults to 0.01.
            epsilon_decay (float, optional): Decay rate for epsilon per step. Defaults to 0.997.
            memory_size (int, optional): Size of the replay buffer. Defaults to 10000.
            device (torch.device, optional): Device for computations (CPU or GPU). Defaults to None, 
                                                which selects CUDA if available, otherwise CPU.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.target_model = copy.deepcopy(model)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # Target model is used only for inference
        self.action_size = 9
        self.mask_invalid = mask_invalid
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = deque(maxlen=memory_size)  # Replay buffer for storing experiences
        self.loss_hist = []


    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience tuple in the replay buffer.
        
        Args:
            state (array-like): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (array-like): The state after taking the action.
            done (bool): Whether the episode ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Select an action using an epsilon-greedy policy.
        
        Args:
            state (array-like): The current state.
        
        Returns:
            int: The selected action.
        """
        valid_actions = np.where(state == 0)[0]
        # Opponent end the game
        if valid_actions.size == 0:
            return -1
        # Disable masking
        if not self.mask_invalid:
            valid_actions = np.arange(9)
            
        # Exploration: Choose a random valid action
        if np.random.rand() <= self.epsilon:
            return np.random.choice(valid_actions)

        # Exploitation: Choose action with the highest Q-value
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.model(state_tensor).squeeze(0)
        masked_q_values = torch.full_like(q_values, -float('inf'))  # Mask invalid actions
        masked_q_values[valid_actions] = q_values[valid_actions]

        return torch.argmax(masked_q_values).item()

    def replay(self, batch_size):
        """
        Train the model using a batch of experiences from the replay buffer.
        
        Args:
            batch_size (int): The number of experiences to sample for training.
        """
        if len(self.memory) < batch_size:
            return  # Not enough experiences in the replay buffer

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert data to tensors
        states = np.array(states)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Compute target Q-values
        targets = self.model(states).detach()
        next_q_values = self.target_model(next_states).detach()
        max_next_q_values = torch.max(next_q_values, dim=1)[0]
        targets[range(batch_size), actions] = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Train the model
        self.optimizer.zero_grad()
        outputs = self.model(states)
        loss = F.smooth_l1_loss(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.loss_hist.append(loss.item())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        """
        Update the target model to match the current model.
        """
        self.target_model.load_state_dict(self.model.state_dict())