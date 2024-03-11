from utils_for_training import train_agent, plot_eval_results, play_agent
from epsilon_greedy_ql_agent import EpsilonGreedyQLAgent
from zoo_hunt_env_editor import both_far_from_plant_stag_in_mid

# rl_agent = EpsilonGreedyQLAgent('player_1', 5, epsilon=0.9, alpha=0.1, gamma=0.5)
# train_agent(both_far_from_plant_stag_in_mid, (1,0,0,0,0,0), rl_agent,
#             need_eval=True, eval_episodes=10, rl_agent_filename='training_results/eg_q_table_1.pkl',
#             eval_result_filename='training_results/eg_eval_result_1.json')
#
# plot_eval_results('training_results/eg_eval_result_1.json')
play_agent(both_far_from_plant_stag_in_mid, (1,0,0,0,0,0), 'training_results/eg_q_table_1.pkl')