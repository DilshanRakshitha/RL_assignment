import gymnasium as gym
import numpy as np
import pickle

# Initialize the CartPole environment
env = gym.make('CartPole-v1')

# Set the Q-learning parameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1  
num_episodes = 1000
max_steps = 100

# Initialize the Q-table
num_bins = 10
state_bins = [np.linspace(-4.8, 4.8, num_bins),
              np.linspace(-4, 4, num_bins),
              np.linspace(-0.418, 0.418, num_bins),
              np.linspace(-4, 4, num_bins)]
q_table = np.zeros([num_bins] * 4 + [env.action_space.n])

# Function to discretize continuous state values into bins
def discretize_state(state):
    state_adj = np.clip(state, [-4.8, -4, -0.418, -4], [4.8, 4, 0.418, 4])
    discrete_state = [np.digitize(state_adj[i], state_bins[i]) - 1 for i in range(len(state))]
    return tuple(discrete_state)

# Training loop
for episode in range(num_episodes):
    state = discretize_state(env.reset()[0])
    total_reward = 0

    for step in range(max_steps):
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Take the action in the environment
        next_state_raw, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state_raw)
        
        # Update Q-value
        best_next_action = np.argmax(q_table[next_state])
        q_table[state][action] += learning_rate * (reward + discount_factor * q_table[next_state][best_next_action] - q_table[state][action])
        
        # Update the state
        state = next_state
        total_reward += reward

        if done:
            break

    
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Save the Q-table to a file
with open('q_table_cartpole.pkl', 'wb') as f:
    pickle.dump(q_table, f)

print("Training finished and Q-table saved!")

env.close()
