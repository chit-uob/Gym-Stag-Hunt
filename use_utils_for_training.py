from utils_for_training import train_agent, plot_eval_results, play_agent, play_agent_vs_human
from epsilon_greedy_ql_agent import EpsilonGreedyQLAgent
from zoo_hunt_env_editor import *
from find_out_strategy import FindOutStrategyAgent

ALWAYS_STAG = (1,0,0,0,0,0)
ALWAYS_PLANT = (0,1,0,0,0,0)
TIT_FOR_TAT = (1,1,0,1,1,0)

find_out_strategy_agent = FindOutStrategyAgent('both_far_from_plant_stag_in_mid')
play_agent_vs_human(both_far_from_plant_stag_in_mid, ALWAYS_STAG, find_out_strategy_agent, load_renderer=False)