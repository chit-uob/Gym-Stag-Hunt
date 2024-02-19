import numpy as np

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