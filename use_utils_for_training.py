from utils_for_training import train_agent, plot_eval_results, play_agent
from epsilon_greedy_ql_agent import EpsilonGreedyQLAgent
from zoo_hunt_env_editor import *

rl_agent = EpsilonGreedyQLAgent('player_1', 5, epsilon=0.9, alpha=0.1, gamma=0.9)
train_agent(choose_from_stag_or_plant, (1,0,0,0,0,0), rl_agent,
            episodes=1000,
            need_eval=True, rl_agent_filename='training_results/eg_q_table_1.pkl',
            eval_result_filename='training_results/eg_eval_result_1.json')

plot_eval_results('training_results/eg_eval_result_1.json', 'results')
play_agent(choose_from_stag_or_plant, (1,0,0,0,0,0), 'training_results/eg_q_table_1.pkl', frame_interval=0.5)