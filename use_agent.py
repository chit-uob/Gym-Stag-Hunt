import time
from gym_stag_hunt.envs.pettingzoo.hunt import ZooHuntEnvironment
from zoo_hunt_env_editor import *
from proposed_agent import ProposedAgent

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAND = 4

wasd_to_action = {
    "w": UP,
    "a": LEFT,
    "s": DOWN,
    "d": RIGHT,
    "q": STAND
}

# Initialize the environment with your parameters
env = choose_from_stag_or_plant()
proposed_agent = ProposedAgent(get_player_0_position(env), 1, 1, 1, 1, 1, 0)
agent_0_obs = env.observe("player_0")
agent_1_obs = env.observe("player_1")
env.render(mode="human")

for _ in range(100):
    proposed_agent_action = proposed_agent.choose_action(agent_0_obs)
    human_action = ""
    while human_action == "":
        human_action = input("Enter action (w: up, a: left, s: down, d: right, q: stand): ")
    print(f"Proposed agent action: {proposed_agent_action}, Human action: {human_action}")
    observation, reward, done, info = env.step({'player_0': proposed_agent_action, 'player_1': wasd_to_action[human_action]})
    new_agent_obs = observation['player_0']
    print(observation)
    proposed_agent.update_parameters(agent_0_obs, new_agent_obs)
    agent_0_obs = new_agent_obs
    env.render(mode="human")