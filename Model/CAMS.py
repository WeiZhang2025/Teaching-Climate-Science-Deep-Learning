import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
from collections import deque


class CAMSAgent:
    """Deep Q-Learning agent for Climate Adaptation and Mitigation Strategy (CAMS)."""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """Create a neural network model for decision-making."""
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action based on exploration-exploitation strategy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """Train the model using past experiences."""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]), verbose=0)[0])
            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action] = target
            self.model.fit(np.array([state]), np.array([target_f[0]]), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    # Example: State size = 4 (temperature, emissions, policy cost, adaptation level)
    # Action size = 3 (mitigation action, adaptation action, mixed strategy)
    state_size = 4
    action_size = 3
    agent = CAMSAgent(state_size, action_size)

    # Simulated Training Loop
    episodes = 1000
    for e in range(episodes):
        state = np.random.rand(state_size)  # Random initial state
        for time in range(200):  # Max steps per episode
            action = agent.act(state)
            next_state = np.random.rand(state_size)  # Simulate next state
            reward = np.random.rand()  # Simulated reward
            done = time == 199  # End of episode
            agent.remember(state, action, reward, next_state, done)
            state = next_state

        agent.replay(32)  # Train with batch size of 32

    # Save trained model
    agent.model.save("models/cams_model.h5")
    print("CAMS Model saved successfully.")
