from itertools import product
from random import choice
from sys import stdout

from numpy import all, full, zeros, uint8

symbol_dict = {"hunt": ("S", "P"), "harvest": ("p", "P"), "escalation": "M"}

A_AGENT = 0  # base
B_AGENT = 1

STAG = 2  # hunt
PLANT = 3

Y_PLANT = 2  # harvest
M_PLANT = 3

MARK = 2  # escalation


def print_matrix(obs, game, grid_size):
    if game == "escalation":
        matrix = full((grid_size[0], grid_size[1], 3), False, dtype=bool)
    else:
        matrix = full((grid_size[0], grid_size[1], 4), False, dtype=bool)

    if game == "hunt":
        a, b, stag = (obs[0], obs[1]), (obs[2], obs[3]), (obs[4], obs[5])
        matrix[a[1]][a[0]][A_AGENT] = True
        matrix[b[1]][b[0]][B_AGENT] = True
        if stag != (255, 255):
            matrix[stag[1]][stag[0]][STAG] = True
        for i in range(6, len(obs), 2):
            plant = obs[i], obs[i + 1]
            if plant != (255, 255):
                matrix[plant[1]][plant[0]][PLANT] = True

    elif game == "harvest":
        a, b = (obs[0], obs[1]), (obs[2], obs[3])
        matrix[a[0]][a[1]][A_AGENT] = True
        matrix[b[0]][b[1]][B_AGENT] = True

        for i in range(4, len(obs), 3):
            plant_age = M_PLANT if obs[i + 2] else Y_PLANT
            matrix[obs[i]][obs[i + 1]][plant_age] = True

    elif game == "escalation":
        a, b, mark = (obs[0], obs[1]), (obs[2], obs[3]), (obs[4], obs[5])
        matrix[a[0]][a[1]][A_AGENT] = True
        matrix[b[0]][b[1]][B_AGENT] = True
        matrix[mark[0]][mark[1]][MARK] = True

    symbols = symbol_dict[game]

    stdout.write("╔═" + "═════"*grid_size[0] + "══╗\n")
    for row in matrix:
        stdout.write("║ ·")
        for col in row:
            cell = []
            cell.append("A") if col[0] == 1 else cell.append(" ")
            cell.append("B") if col[1] == 1 else cell.append(" ")
            cell.append(symbols[0]) if col[2] == 1 else cell.append(" ")
            if game != "escalation":
                cell.append(symbols[1]) if col[3] == 1 else cell.append(" ")
            else:
                cell.append(" ")
            stdout.write("".join(cell) + "·")
        stdout.write(" ║")
        stdout.write("\n")
    stdout.write("╚═" + "═════"*grid_size[0] + "══╝\n\r")
    stdout.flush()


def overlaps_entity(a, b):
    """
    :param a: (X, Y) tuple for entity 1
    :param b: (X, Y) tuple for entity 2
    :return: True if they are on the same cell, False otherwise
    """
    return (a == b).all()


def place_entity_in_unoccupied_cell(used_coordinates, grid_dims):
    """
    Returns a random unused coordinate.
    :param used_coordinates: a list of already used coordinates
    :param grid_dims: dimensions of the grid so we know what a valid coordinate is
    :return: the chosen x, y coordinate
    """
    all_coords = list(product(list(range(grid_dims[0])), list(range(grid_dims[1]))))

    for coord in used_coordinates:
        for test in all_coords:
            if all(test == coord):
                all_coords.remove(test)

    return choice(all_coords)


def spawn_plants(grid_dims, how_many, used_coordinates):
    new_plants = []
    for x in range(how_many):
        new_plant = zeros(2, dtype=uint8)
        new_pos = place_entity_in_unoccupied_cell(
            grid_dims=grid_dims, used_coordinates=new_plants + used_coordinates
        )
        new_plant[0], new_plant[1] = new_pos
        new_plants.append(new_plant)
    return new_plants


def respawn_plants(plants, tagged_plants, grid_dims, used_coordinates):
    for tagged_plant in tagged_plants:
        new_plant = zeros(2, dtype=uint8)
        new_pos = place_entity_in_unoccupied_cell(
            grid_dims=grid_dims, used_coordinates=plants + used_coordinates
        )
        new_plant[0], new_plant[1] = new_pos
        plants[tagged_plant] = new_plant
    return plants


def does_not_respawn_plants(plants, tagged_plants, grid_dims, used_coordinates):
    for tagged_plant in tagged_plants:
        new_plant = zeros(2, dtype=uint8)
        new_plant[0], new_plant[1] = 255, 255
        plants[tagged_plant] = new_plant
    return plants


def calculate_distance(location_1, location_2):
    """
    Calculate the distance between two locations
    """
    # return math.sqrt(math.pow((location_1[0] - location_2[0]), 2) + math.pow((location_1[1] - location_2[1]), 2))
    location_1 = location_1.astype('int32')
    location_2 = location_2.astype('int32')
    return abs((location_1[0]) - location_2[0]) + abs(location_1[1] - location_2[1])