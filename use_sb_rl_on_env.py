import dill
from gym_stag_hunt.envs.pettingzoo.hunt import ZooHuntEnvironment
from zoo_hunt_env_editor import *
from proposed_agent import ProposedAgent
from ucb_marl_agent import MARL_Comm


EPISODES = 50000
TIME_STEP = 20

rl_agent = MARL_Comm('player_1', 1, EPISODES, TIME_STEP, 0)


def encode_obs(obs):
    return str(obs)


for episode in range(EPISODES):
    env = both_far_from_plant_stag_in_mid()
    observation, reward, done, info = env.step({'player_0': 4, 'player_1': 4})
    agent_0_obs = observation['player_0']
    agent_1_obs = observation['player_1']
    proposed_agent = ProposedAgent(get_player_0_position(env), 1, 0, 0, 0, 0, 0)
    if episode % 500 == 0:
        print(episode)
    for time_step in range(1, TIME_STEP+1):
        proposed_agent_action = proposed_agent.choose_action(agent_0_obs)
        rl_agent_action = rl_agent.policy(encode_obs(agent_1_obs), time_step)
        observation, reward, done, info = env.step(
            {'player_0': proposed_agent_action, 'player_1': rl_agent_action})
        new_agent_0_obs = observation['player_0']
        new_agent_1_obs = observation['player_1']
        proposed_agent.update_parameters(agent_0_obs, new_agent_0_obs)
        agent_1_reward = reward['player_1']
        if agent_1_reward == 0:
            agent_1_reward = -1
        rl_agent.update(episode, time_step, encode_obs(agent_1_obs), encode_obs(new_agent_1_obs), rl_agent_action, agent_1_reward)
        rl_agent.update_values(episode, time_step)
        agent_0_obs = new_agent_0_obs
        agent_1_obs = new_agent_1_obs


with open('q_table.pkl', 'wb') as f:
    dill.dump(rl_agent, f, dill.HIGHEST_PROTOCOL)