import json
import matplotlib.pyplot as plt

with open('eg_eval_rewards_1.json', 'r') as f:
    eval_rewards = json.load(f)

# Extract episodes and rewards into separate lists for plotting
episodes, rewards = zip(*eval_rewards)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(episodes, rewards, marker='o', linestyle='-', color='b')
plt.title('Training Progress')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True)
plt.show()
