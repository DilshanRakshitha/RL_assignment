import gymnasium as gym
import numpy as np
import pickle

# Initialize the CartPole environment
env = gym.make('CartPole-v1', render_mode="human")

# Load the saved Q-table from the training script
with open('q_table_cartpole.pkl', 'rb') as f:
    q_table = pickle.load(f)

num_bins = 10
state_bins = [np.linspace(-4.8, 4.8, num_bins),
              np.linspace(-4, 4, num_bins),
              np.linspace(-0.418, 0.418, num_bins),
              np.linspace(-4, 4, num_bins)]

def discretize_state(state):
    state_adj = np.clip(state, [-4.8, -4, -0.418, -4], [4.8, 4, 0.418, 4])
    discrete_state = [np.digitize(state_adj[i], state_bins[i]) - 1 for i in range(len(state))]
    return tuple(discrete_state)

num_test_episodes = 10
max_steps = 200

for episode in range(num_test_episodes):
    state = discretize_state(env.reset()[0])
    total_reward = 0

    for step in range(max_steps):
        # Select the action with the highest Q-value
        action = np.argmax(q_table[state])

        # Take the action in the environment
        next_state_raw, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state_raw)

        # Update the state and accumulate reward
        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Test Episode {episode + 1}: Total Reward = {total_reward}")

# Close the environment
env.close()
