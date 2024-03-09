import numpy as np


class EpsilonGreedyQLAgent:
    def __init__(self, name, n_actions, epsilon=0.1, alpha=0.1, gamma=0.99):
        """
        Initialize the epsilon-greedy Q-learning agent.
        :param name: Name of the agent (for identification).
        :param n_actions: Number of actions available to the agent.
        :param epsilon: Epsilon value for epsilon-greedy action selection.
        :param alpha: Learning rate.
        :param gamma: Discount factor for future rewards.
        """
        self.name = name
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}
        self.episode_transitions = []  # List to store transitions

    def encode_obs(self, obs):
        return str(obs)

    def train_policy(self, encoded_obs):
        """
        Selects an action based on the epsilon-greedy policy.
        :param encoded_obs: The current observation encoded as a string.
        :return: The action to take.
        """
        if encoded_obs not in self.q_table:
            self.q_table[encoded_obs] = np.zeros(self.n_actions)

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[encoded_obs])

    def play_normal(self, encoded_obs, verbose=False):
        """
        Selects an action based on the greedy policy.
        :param encoded_obs: The current observation encoded as a string.
        :return: The action to take.
        """
        if encoded_obs not in self.q_table:
            self.q_table[encoded_obs] = np.zeros(self.n_actions)
        if verbose:
            print(self.q_table[encoded_obs])
        # if all values are 0, then stand still
        if np.all(self.q_table[encoded_obs] == 0):
            return 4
        return np.argmax(self.q_table[encoded_obs])

    def store_transition(self, encoded_obs, action, reward):
        """
        Stores the transition (state, action, reward) in the episode buffer.
        :param encoded_obs: The current observation encoded as a string.
        :param action: The action taken.
        :param reward: The reward received.
        """
        self.episode_transitions.append((encoded_obs, action, reward))

    def update_end_of_episode(self):
        """
        Updates the Q-values at the end of the episode using the stored transitions.
        """
        total_reward = 0
        for encoded_obs, action, reward in reversed(self.episode_transitions):
            if encoded_obs not in self.q_table:
                self.q_table[encoded_obs] = np.zeros(self.n_actions)
            total_reward = reward + self.gamma * total_reward
            self.q_table[encoded_obs][action] += self.alpha * (total_reward - self.q_table[encoded_obs][action])

        # Clear the episode buffer after updating
        self.episode_transitions.clear()

    def set_epsilon(self, epsilon):
        """
        Set the epsilon value for epsilon-greedy action selection.
        :param epsilon: The new epsilon value.
        """
        self.epsilon = epsilon
