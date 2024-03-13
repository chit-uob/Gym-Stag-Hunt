from utils_for_training import train_agent, plot_eval_results, play_agent
from epsilon_greedy_ql_agent import EpsilonGreedyQLAgent
from zoo_hunt_env_editor import both_far_from_plant_stag_in_mid, stag_on_the_side_plant_in_mid, choose_from_stag_or_plant

TYPE_OF_ENVS = [both_far_from_plant_stag_in_mid, stag_on_the_side_plant_in_mid, choose_from_stag_or_plant]
TYPE_OF_SETTINGS = [(1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (1, 1, 0, 1, 1, 0)]
TYPE_OF_SETTINGS_NAMES = ["always_go_to_stag", "always_go_to_plant", "tit_for_tat"]

for env_type in TYPE_OF_ENVS:
    for setting_type, settings_name in zip(TYPE_OF_SETTINGS, TYPE_OF_SETTINGS_NAMES):
        print(env_type, setting_type, settings_name)
