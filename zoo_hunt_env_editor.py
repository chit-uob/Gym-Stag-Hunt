import numpy as np
from gym_stag_hunt.envs.pettingzoo.hunt import ZooHuntEnvironment
from gym_stag_hunt.src.utils import respawn_plants


def get_player_0_position(env):
    """
    Get the position of player 0 in the ZooHuntEnvironment.
    :param env: ZooHuntEnvironment
    :return: (x, y) coordinates
    """
    return env.env.game.A_AGENT


def set_stag_coord(env, x, y):
    """
    Set the position of the stag in the ZooHuntEnvironment.
    :param env: ZooHuntEnvironment
    :param x: x-coordinate
    :param y: y-coordinate
    """
    env.env.game.STAG = np.array([x, y])


def disable_movement_for_stag(env):
    """
    Disable the movement of the stag in the ZooHuntEnvironment.
    :param env: ZooHuntEnvironment
    """
    env.env.game._move_stag = lambda: True


def set_plant_positions(env, plant_positions):
    """
    Set the positions of the plant in the ZooHuntEnvironment.
    :param env: ZooHuntEnvironment
    :param plant_positions: list of (x, y) tuples
    """
    env.env.game.PLANTS = [np.array(pos) for pos in plant_positions]




def set_player_0_position(env, x, y):
    """
    Set the position of player 0 in the ZooHuntEnvironment.
    :param env: ZooHuntEnvironment
    :param x: x-coordinate
    :param y: y-coordinate
    """
    env.env.game.A_AGENT = np.array([x, y])


def set_player_1_position(env, x, y):
    """
    Set the position of player 1 in the ZooHuntEnvironment.
    :param env: ZooHuntEnvironment
    :param x: x-coordinate
    :param y: y-coordinate
    """
    env.env.game.B_AGENT = np.array([x, y])


def get_basic_env(grid_size=10, load_renderer=False):
    env = ZooHuntEnvironment(
        obs_type="coords",
        enable_multiagent=True,
        stag_follows=False,
        run_away_after_maul=False,
        forage_quantity=2,
        stag_reward=5,
        forage_reward=2,
        mauling_punishment=-1,
        load_renderer=load_renderer,
        grid_size=(grid_size, grid_size),
        respawn_plants=False,
        respawn_stag=False,
        move_closer_reward=True,
    )
    env.reset()
    return env


def both_close_to_stag(load_renderer=False):
    env = get_basic_env(load_renderer=load_renderer)
    set_player_0_position(env, 4, 5)
    set_player_1_position(env, 6, 5)
    set_stag_coord(env, 5, 5)
    disable_movement_for_stag(env)
    set_plant_positions(env, [(0, 1), (9, 8)])
    return env


def both_close_to_plant_stag_in_mid(load_renderer=False):
    env = get_basic_env(load_renderer=load_renderer)
    set_player_0_position(env, 0, 0)
    set_player_1_position(env, 9, 9)
    set_stag_coord(env, 5, 5)
    disable_movement_for_stag(env)
    set_plant_positions(env, [(0, 1), (9, 8)])
    return env


def we_close_to_plant_stag_in_mid(load_renderer=False):
    env = get_basic_env(load_renderer=load_renderer)
    set_player_0_position(env, 0, 0)
    set_player_1_position(env, 9, 9)
    set_stag_coord(env, 5, 5)
    disable_movement_for_stag(env)
    set_plant_positions(env, [(0, 1), (0, 8)])
    return env


def they_close_to_plant_stag_in_mid(load_renderer=False):
    env = get_basic_env(load_renderer=load_renderer)
    set_player_0_position(env, 0, 0)
    set_player_1_position(env, 9, 9)
    set_stag_coord(env, 5, 5)
    disable_movement_for_stag(env)
    set_plant_positions(env, [(9, 1), (9, 8)])
    return env


def both_far_from_plant_stag_in_mid(load_renderer=False):
    env = get_basic_env(load_renderer=load_renderer)
    set_player_0_position(env, 0, 0)
    set_player_1_position(env, 9, 9)
    set_stag_coord(env, 4, 4)
    disable_movement_for_stag(env)
    set_plant_positions(env, [(9, 0), (0, 9)])
    return env


def stag_on_the_side_plant_in_mid(load_renderer=False):
    env = get_basic_env(load_renderer=load_renderer)
    set_player_0_position(env, 0, 0)
    set_player_1_position(env, 9, 9)
    set_stag_coord(env, 9, 0)
    disable_movement_for_stag(env)
    set_plant_positions(env, [(4, 4), (0, 9)])
    return env


def choose_from_stag_or_plant(load_renderer=False):
    env = get_basic_env(load_renderer=load_renderer)
    set_player_0_position(env, 2, 2)
    set_player_1_position(env, 7, 7)
    set_stag_coord(env, 4, 5)
    disable_movement_for_stag(env)
    set_plant_positions(env, [(0, 0), (9, 9)])
    return env
