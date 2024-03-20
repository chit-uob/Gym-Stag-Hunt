import time
from gw_epsilon_greedy_ql_agent import EpsilonGreedyQLAgent
from gw_proposed_agent import ProposedAgent
from gw_zoo_hunt_env_editor import get_player_0_position
import json
import dill
import matplotlib.pyplot as plt
import pandas as pd


def encode_obs(obs):
    return str(obs)


def decay_epsilon(epsilon_value, decay_rate):
    return epsilon_value * decay_rate


def eval_agent(env, human_agent_settings, rl_agent, total_time_step=20):
    """
    Evaluate the agent's performance.
    :param env: The environment to evaluate the agent in.
    :param human_agent: The human agent to play against.
    :param rl_agent: The agent to evaluate.
    :param total_time_step: The number of time steps to evaluate the agent for.
    :param eval_type: turn_until_reward or total_reward
    """
    total_reward = 0
    observation, reward, done, info = env.step({'player_0': 4, 'player_1': 4})
    agent_0_obs = observation['player_0']
    agent_1_obs = observation['player_1']
    human_agent = ProposedAgent(get_player_0_position(env), *human_agent_settings)
    for time_step in range(total_time_step):
        human_agent_action = human_agent.choose_action(agent_0_obs)
        rl_agent_action = rl_agent.play_normal(encode_obs(agent_1_obs))
        observation, reward, done, info = env.step(
            {'player_0': human_agent_action, 'player_1': rl_agent_action})
        new_agent_0_obs = observation['player_0']
        new_agent_1_obs = observation['player_1']
        human_agent.update_parameters(agent_0_obs, new_agent_0_obs)
        agent_0_obs = new_agent_0_obs
        agent_1_obs = new_agent_1_obs
        total_reward += reward['player_1']
        if done['player_1']:
            return total_reward, time_step
    return total_reward, total_time_step


def train_agent(env_generator, human_agent_settings, rl_agent,
                episodes=5000, total_time_step=20, progress_print=500,
                decay_count=300, decay_rate=0.991,
                need_eval=False, eval_episodes=5,
                rl_agent_filename='eg_q_table.pkl',
                eval_result_filename='eg_eval_result.json'):
    decay_episode = episodes // decay_count
    eval_results = []

    for episode in range(episodes):
        env = env_generator()
        observation, reward, done, info = env.step({'player_0': 4, 'player_1': 4})
        agent_0_obs = observation['player_0']
        agent_1_obs = observation['player_1']
        proposed_agent = ProposedAgent(get_player_0_position(env), *human_agent_settings)
        for time_step in range(total_time_step):
            proposed_agent_action = proposed_agent.choose_action(agent_0_obs)
            rl_agent_action = rl_agent.train_policy(encode_obs(agent_1_obs))
            observation, reward, done, info = env.step(
                {'player_0': proposed_agent_action, 'player_1': rl_agent_action})
            new_agent_0_obs = observation['player_0']
            new_agent_1_obs = observation['player_1']
            proposed_agent.update_parameters(agent_0_obs, new_agent_0_obs)
            agent_1_reward = reward['player_1']
            rl_agent.store_transition(encode_obs(agent_1_obs), rl_agent_action, agent_1_reward)
            agent_0_obs = new_agent_0_obs
            agent_1_obs = new_agent_1_obs
            if done['player_1']:
                break
        rl_agent.update_end_of_episode()
        if progress_print and episode % progress_print == 0:
            print(episode, rl_agent.epsilon)
        if episode % decay_episode == 0:
            rl_agent.epsilon = decay_epsilon(rl_agent.epsilon, decay_rate)
        if need_eval and episode % eval_episodes == 0:
            env_for_eval = env_generator()
            eval_result = eval_agent(env_for_eval, human_agent_settings, rl_agent, total_time_step)
            eval_results.append((episode, *eval_result))

    with open(rl_agent_filename, 'wb') as f:
        dill.dump(rl_agent, f)
        print(f"Agent saved to {rl_agent_filename}")

    if need_eval:
        with open(eval_result_filename, 'w') as f:
            json.dump(eval_results, f)
            print(f"Evaluation results saved to {eval_result_filename}")


def plot_eval_results(eval_result_filename, plot_title):
    with open(eval_result_filename, 'r') as f:
        eval_results = json.load(f)
    episodes, rewards, turn_until_reward = zip(*eval_results)
    # print(episodes)
    # print(rewards)
    # print(turn_until_reward)
    # how_good = [reward * 20-turn for reward, turn in zip(rewards, turn_until_reward)]
    window = 20

    turn_until_reward_series = pd.Series(turn_until_reward)
    turn_until_reward_smoothed = turn_until_reward_series.rolling(window=window, min_periods=1).mean()

    plt.plot(episodes, turn_until_reward_smoothed)
    plt.title(plot_title)
    plt.xlabel("Episodes")
    plt.ylabel("Turn until reward")
    plt.show()

    rewards_series = pd.Series(rewards)
    rewards_smoothed = rewards_series.rolling(window=window, min_periods=1).mean()

    plt.plot(episodes, rewards_smoothed)
    plt.title(plot_title)
    plt.xlabel("Episodes")
    plt.ylabel("Total reward")
    plt.show()


def plot_eval_results_compare(eval_result_filename1, eval_result_filename2, plot_title):
    # Load data from the first file
    with open(eval_result_filename1, 'r') as f:
        eval_results1 = json.load(f)
    episodes1, rewards1, turn_until_reward1 = zip(*eval_results1)

    # Load data from the second file
    with open(eval_result_filename2, 'r') as f:
        eval_results2 = json.load(f)
    episodes2, rewards2, turn_until_reward2 = zip(*eval_results2)

    window = 20

    # Plot turn_until_reward
    turn_until_reward_series1 = pd.Series(turn_until_reward1)
    turn_until_reward_smoothed1 = turn_until_reward_series1.rolling(window=window, min_periods=1).mean()

    turn_until_reward_series2 = pd.Series(turn_until_reward2)
    turn_until_reward_smoothed2 = turn_until_reward_series2.rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(episodes1, turn_until_reward_smoothed1, 'r-', label=eval_result_filename1)
    plt.plot(episodes2, turn_until_reward_smoothed2, 'b-', label=eval_result_filename2)
    plt.title(f"{plot_title} - Turn until reward")
    plt.xlabel("Episodes")
    plt.ylabel("Turn until reward")
    plt.legend()
    plt.show()

    # Plot rewards
    rewards_series1 = pd.Series(rewards1)
    rewards_smoothed1 = rewards_series1.rolling(window=window, min_periods=1).mean()

    rewards_series2 = pd.Series(rewards2)
    rewards_smoothed2 = rewards_series2.rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(episodes1, rewards_smoothed1, 'r-', label=eval_result_filename1)
    plt.plot(episodes2, rewards_smoothed2, 'b-', label=eval_result_filename2)
    plt.title(f"{plot_title} - Total reward")
    plt.xlabel("Episodes")
    plt.ylabel("Total reward")
    plt.legend()
    plt.show()


def play_agent(env_generator, human_agent_settings, rl_agent_file_name,
               total_time_step=20, frame_interval=0.5, load_renderer=True):
    with open(rl_agent_file_name, 'rb') as f:
        rl_agent = dill.load(f)
    env = env_generator(load_renderer=load_renderer)
    observation, reward, done, info = env.step({'player_0': 4, 'player_1': 4})
    env.render(mode="human")
    time.sleep(frame_interval)
    agent_0_obs = observation['player_0']
    agent_1_obs = observation['player_1']
    human_agent = ProposedAgent(get_player_0_position(env), *human_agent_settings)
    for time_step in range(total_time_step):
        human_agent_action = human_agent.choose_action(agent_0_obs)
        rl_agent_action = rl_agent.play_normal(encode_obs(agent_1_obs))
        observation, reward, done, info = env.step(
            {'player_0': human_agent_action, 'player_1': rl_agent_action})
        new_agent_0_obs = observation['player_0']
        new_agent_1_obs = observation['player_1']
        human_agent.update_parameters(agent_0_obs, new_agent_0_obs)
        agent_0_obs = new_agent_0_obs
        agent_1_obs = new_agent_1_obs
        env.render(mode="human")
        time.sleep(frame_interval)
        if done['player_1']:
            break


def play_agent_vs_human(env_generator, human_agent_settings, rl_agent,
               total_time_step=20, frame_interval=0.5, load_renderer=True):
    env = env_generator(load_renderer=load_renderer)
    observation, reward, done, info = env.step({'player_0': 4, 'player_1': 4})
    env.render(mode="human")
    time.sleep(frame_interval)
    agent_0_obs = observation['player_0']
    agent_1_obs = observation['player_1']
    human_agent = ProposedAgent(get_player_0_position(env), *human_agent_settings)
    for time_step in range(total_time_step):
        human_agent_action = human_agent.choose_action(agent_0_obs)
        rl_agent_action = rl_agent.play_move(agent_1_obs)
        observation, reward, done, info = env.step(
            {'player_0': human_agent_action, 'player_1': rl_agent_action})
        new_agent_0_obs = observation['player_0']
        new_agent_1_obs = observation['player_1']
        human_agent.update_parameters(agent_0_obs, new_agent_0_obs)
        rl_agent.receive_feedback(agent_1_obs, new_agent_1_obs)
        agent_0_obs = new_agent_0_obs
        agent_1_obs = new_agent_1_obs
        env.render(mode="human")
        time.sleep(frame_interval)