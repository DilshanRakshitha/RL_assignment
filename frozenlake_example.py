import numpy as np
import gymnasium as gym

# Create the FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Parameters
alpha = 0.8     # Learning rate
gamma = 0.95    # Discount factor
epsilon = 1.0   # Exploration rate
epsilon_decay = 0.99
min_epsilon = 0.01
num_episodes = 1000
max_steps = 100  # Max steps per episode

# Initialize Q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Q-Learning algorithm
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    steps = 0
    
    while not done and steps < max_steps:
        # Epsilon-greedy policy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit learned values
        
        # Take action
        next_state, reward, done, _, _ = env.step(action)
        
        # Update Q-value using the Q-Learning formula
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        
        # Move to the next state
        state = next_state
        steps += 1

    # Decay epsilon to reduce exploration over time
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Print progress every 100 episodes
    if episode % 100 == 0:
        print(f"Episode: {episode}, Epsilon: {epsilon:.4f}")

# Testing the trained agent
state, _ = env.reset()
done = False
steps = 0

print("\nTrained agent's performance:")
while not done and steps < max_steps:
    action = np.argmax(q_table[state])
    next_state, reward, done, _, _ = env.step(action)
    env.render()
    state = next_state
    steps += 1

env.close()
