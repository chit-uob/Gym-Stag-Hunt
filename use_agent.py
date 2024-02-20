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
}

# Initialize the environment with your parameters
env = ZooHuntEnvironment(
    obs_type="coords",
    enable_multiagent=True,
    stag_follows=False,
    run_away_after_maul=False,
    forage_quantity=2,
    stag_reward=5,
    forage_reward=1,
    mauling_punishment=-5,
)

# Now you can use the environment to reset, step, render, etc.
obs = env.reset()
set_stag_coord(env, 1, 1)
disable_movement_for_stag(env)
set_plant_positions(env, [(3, 3), (2, 2)])
proposed_agent = ProposedAgent((0, 0), 1, 1, 1)
old_agent_obs = env.env.game.get_observation()
env.render(mode="human")

for _ in range(10):
    proposed_agent_action = proposed_agent.choose_action(old_agent_obs)
    human_action = input("Enter action (w: up, a: left, s: down, d: right): ")
    print(f"Proposed agent action: {proposed_agent_action}, Human action: {human_action}")
    observation, reward, done, info = env.step({'player_0': proposed_agent_action, 'player_1': wasd_to_action[human_action]})
    new_agent_obs = observation['player_0']
    proposed_agent.update_parameters(old_agent_obs, new_agent_obs)
    old_agent_obs = new_agent_obs
    env.render(mode="human")
    time.sleep(1)