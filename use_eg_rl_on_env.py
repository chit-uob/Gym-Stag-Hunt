import dill
import json
from gym_stag_hunt.envs.pettingzoo.hunt import ZooHuntEnvironment
from zoo_hunt_env_editor import *
from proposed_agent import ProposedAgent
from epsilon_greedy_ql_agent import EpsilonGreedyQLAgent

EPISODES = 5000
TIME_STEP = 20
DECAY_EPISODE = 20
DECAY_RATE = 0.01
EVAL_EPISODES = 10
eval_rewards = []

rl_agent = EpsilonGreedyQLAgent('player_1', 5, epsilon=0.9, alpha=0.1, gamma=0.5)

def encode_obs(obs):
    return str(obs)


def decay_epsilon(epsilon_value, decay_rate):
    return epsilon_value * (1 - decay_rate)


def eval_agent(env, human_agent, rl_agent):
    total_reward = 0
    observation, reward, done, info = env.step({'player_0': 4, 'player_1': 4})
    agent_0_obs = observation['player_0']
    agent_1_obs = observation['player_1']
    for time_step in range(TIME_STEP):
        human_agent_action = human_agent.choose_action(agent_0_obs)
        rl_agent_action = rl_agent.play_normal(encode_obs(agent_1_obs))
        observation, reward, done, info = env.step(
            {'player_0': human_agent_action, 'player_1': rl_agent_action})
        new_agent_0_obs = observation['player_0']
        new_agent_1_obs = observation['player_1']
        agent_0_obs = new_agent_0_obs
        agent_1_obs = new_agent_1_obs
        total_reward += reward['player_1']
        if total_reward >= 5:
            return time_step
    return 20
    # return total_reward


for episode in range(EPISODES):
    env = both_far_from_plant_stag_in_mid()
    observation, reward, done, info = env.step({'player_0': 4, 'player_1': 4})
    agent_0_obs = observation['player_0']
    agent_1_obs = observation['player_1']
    proposed_agent = ProposedAgent(get_player_0_position(env), 1, 0, 0, 0, 0, 0)
    if episode % 500 == 0:
        print(episode, rl_agent.epsilon)
    for time_step in range(TIME_STEP):
        proposed_agent_action = proposed_agent.choose_action(agent_0_obs)
        rl_agent_action = rl_agent.train_policy(encode_obs(agent_1_obs))
        observation, reward, done, info = env.step(
            {'player_0': proposed_agent_action, 'player_1': rl_agent_action})
        new_agent_0_obs = observation['player_0']
        new_agent_1_obs = observation['player_1']
        proposed_agent.update_parameters(agent_0_obs, new_agent_0_obs)
        agent_1_reward = reward['player_1']
        rl_agent.store_transition(encode_obs(agent_1_obs), rl_agent_action, agent_1_reward)
        agent_0_obs = new_agent_0_obs
        agent_1_obs = new_agent_1_obs
    rl_agent.update_end_of_episode()
    if episode % DECAY_EPISODE == 0:
        epsilon = decay_epsilon(rl_agent.epsilon, DECAY_RATE)
        rl_agent.set_epsilon(epsilon)
    if episode % EVAL_EPISODES == 0:
        env = both_far_from_plant_stag_in_mid()
        human_agent = ProposedAgent(get_player_0_position(env), 1, 0, 0, 0, 0, 0)
        eval_reward = eval_agent(env, human_agent, rl_agent)
        eval_rewards.append((episode, eval_reward))


with open('eg_q_table_1.pkl', 'wb') as f:
    dill.dump(rl_agent, f, dill.HIGHEST_PROTOCOL)

with open('eg_eval_rewards_1.json', 'w') as f:
    json.dump(eval_rewards, f)
