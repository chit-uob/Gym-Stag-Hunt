import random
import time

from gym_stag_hunt.envs.pettingzoo.hunt import ZooHuntEnvironment
from zoo_hunt_env_editor import *

# Initialize the environment with your parameters
env = ZooHuntEnvironment(
    grid_size=(5, 5),
    screen_size=(600, 600),
    obs_type="coords",
    enable_multiagent=True,
    opponent_policy="random",
    load_renderer=False,
    stag_follows=False,
    run_away_after_maul=False,
    forage_quantity=2,
    stag_reward=5,
    forage_reward=1,
    mauling_punishment=-5,
)

# Now you can use the environment to reset, step, render, etc.
obs = env.reset()
print(obs)
set_stag_coord(env, 1, 1)
disable_movement_for_stag(env)
set_plant_positions(env, [(3, 3), (2, 2)])

for _ in range(100):
  time.sleep(0.1)
  # env.render(mode="human")
  observation, reward, done, info = env.step({'1': 1, '2': 0})
  print(observation)