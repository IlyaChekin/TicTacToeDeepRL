![:)](images\cat.png)
# TicTacToeDeepRL

TicTacToeDeepRL is a reinforcement learning (RL) implementation for the game of Tic-Tac-Toe. The project includes the following components:

- **Environments**: Custom RL environments for training agents to play Tic-Tac-Toe.
- **Visualization and Testing**: Tools to visualize games and test trained agents.
- **Agents**: An RL agent for training and various pre-built agents for testing and visualization.

---

## Project Structure

The project has a simple structure:

- **`src/`**: Contains all the necessary components, including environments, agents, and utilities.
- **`example.ipynb`**: A Jupyter Notebook in the root directory that demonstrates training an RL agent.
- **`requirements.txt`**: Lists all the required dependencies for the project.

---

## Installation

Follow these steps to set up the project:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/TicTacToeDeepRL.git
cd TicTacToeDeepRL
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate       # On Linux/Mac
venv\Scripts\activate          # On Windows
```

### 3. Install Dependencies
Install all required libraries listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Usage

1. Open the `example.ipynb` notebook in Jupyter Notebook or JupyterLab.
2. Run the cells to train an RL agent for Tic-Tac-Toe.
3. Test the trained agent using the provided environments and visualization tools.

---

## Requirements

- Python >= 3.9
- Libraries (installed via `requirements.txt`):
  - `numpy`
  - `torch >= 2.5.1`
  - `matplotlib >= 3.9.2`

---

## Contributing

Feel free to contribute to the project by:
- Adding new features or agents.
- Improving the environments or training algorithms.
- Reporting issues or suggesting improvements.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Happy Learning and Exploring! 🎉