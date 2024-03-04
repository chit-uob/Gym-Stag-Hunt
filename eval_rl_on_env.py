import time

from gym_stag_hunt.envs.pettingzoo.hunt import ZooHuntEnvironment
from zoo_hunt_env_editor import *
from proposed_agent import ProposedAgent
from ucb_marl_agent import MARL_Comm

TIME_STEP = 20

import dill
with open('q_table.pkl', 'rb') as f:
    rl_agent = dill.load(f)


def encode_obs(obs):
    return str(obs)


env = both_far_from_plant_stag_in_mid()
observation, reward, done, info = env.step({'player_0': 4, 'player_1': 4})
env.render(mode="human")
agent_0_obs = observation['player_0']
agent_1_obs = observation['player_1']
proposed_agent = ProposedAgent(get_player_0_position(env), 1, 1, 0, 1, 1, 0)
for time_step in range(1, TIME_STEP+1):
    proposed_agent_action = proposed_agent.choose_action(agent_0_obs)
    rl_agent_action = rl_agent.play_normal(encode_obs(agent_1_obs), time_step)
    observation, reward, done, info = env.step(
        {'player_0': proposed_agent_action, 'player_1': rl_agent_action})
    new_agent_0_obs = observation['player_0']
    new_agent_1_obs = observation['player_1']
    proposed_agent.update_parameters(agent_0_obs, new_agent_0_obs)
    agent_1_reward = reward['player_1']
    agent_0_obs = new_agent_0_obs
    agent_1_obs = new_agent_1_obs
    env.render(mode="human")
    time.sleep(0.5)

