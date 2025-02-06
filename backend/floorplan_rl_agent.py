import random

class FloorplanRLAgent:
    """
    A simple multi-armed bandit agent that learns which mutation rate
    produces the desired floorplan style. In this example, the agent has a
    discrete set of mutation rate actions. After each user selection,
    the agent updates its Q-values using a basic Q-learning update.
    """

    def __init__(self, actions=None, alpha=0.1, epsilon=0.2):
        # Possible mutation rates (arms)
        if actions is None:
            self.actions = [0.05, 0.1, 0.15, 0.2]
        else:
            self.actions = actions
        # Q-values (initialized to 0 for each action)
        self.q_table = {a: 0.0 for a in self.actions}
        self.alpha = alpha    # learning rate
        self.epsilon = epsilon  # exploration probability

    def choose_action(self):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        # Else, choose the action with highest Q-value (break ties randomly)
        max_val = max(self.q_table.values())
        best_actions = [a for a, q in self.q_table.items() if q == max_val]
        return random.choice(best_actions)

    def update(self, action, reward):
        """
        Update Q-value for the given action with the received reward.
        (In this simple bandit, there is no next state.)
        """
        current_q = self.q_table[action]
        self.q_table[action] = current_q + self.alpha * (reward - current_q)

    def get_mutation_rate(self):
        """Return the current best mutation rate (greedy action)."""
        return max(self.q_table, key=self.q_table.get)

    def __str__(self):
        return f"RL Agent Q-Table: {self.q_table}"
