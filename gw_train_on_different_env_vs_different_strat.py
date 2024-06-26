from gw_utils_for_training import train_agent, plot_eval_results, play_agent, plot_eval_results_compare
from gw_epsilon_greedy_ql_agent import EpsilonGreedyQLAgent
from gw_zoo_hunt_env_editor import both_far_from_plant_stag_in_mid, stag_on_the_side_plant_in_mid, choose_from_stag_or_plant

TYPE_OF_ENVS = [both_far_from_plant_stag_in_mid, stag_on_the_side_plant_in_mid, choose_from_stag_or_plant]
TYPE_OF_SETTINGS = [(1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (1, 1, 0, 1, 1, 0)]
TYPE_OF_SETTINGS_NAMES = ["always_go_to_stag", "always_go_to_plant", "tit_for_tat"]

for env_type in TYPE_OF_ENVS:
    for setting_type, settings_name in zip(TYPE_OF_SETTINGS, TYPE_OF_SETTINGS_NAMES):
        rl_agent = EpsilonGreedyQLAgent('player_1', 5, epsilon=0.9, alpha=0.1, gamma=0.9)
        file_prefix = f"training_results/{env_type.__name__}_{settings_name}_added_reward_"
        file_prefix_2 = f"training_results/{env_type.__name__}_{settings_name}_"
        print(file_prefix)
        # train_agent(env_type, setting_type, rl_agent,
        #             need_eval=True, rl_agent_filename=file_prefix+"table.pkl",
        #             eval_result_filename=file_prefix+"result.json")
        print("finished training")
        plot_eval_results_compare(file_prefix+"result.json", file_prefix_2+"result.json", file_prefix[17:])
        # play_agent(env_type, setting_type, file_prefix+"table.pkl", load_renderer=False)
