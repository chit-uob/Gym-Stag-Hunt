from gw_proposed_agent import unpack_observation, negative_distance_delta, calculate_distance, normalize_deltas
import math

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAND = 4

ENV_FIRST_MOVES = {
    'both_far_from_plant_stag_in_mid': [UP, UP, LEFT, LEFT],
    'stag_on_the_side_plant_in_mid': [LEFT, UP, UP],
    'choose_from_stag_or_plant': [RIGHT, DOWN, LEFT, UP]
}



def generate_moves(a_row, a_col, b_row, b_col, max_steps):
    moves = []

    # Calculate the initial Euclidean distance between A and B
    distance = math.sqrt((b_row - a_row)**2 + (b_col - a_col)**2)

    # Iterate over the maximum number of steps
    for _ in range(max_steps):
        # If A is at the same position as B, we're done
        if distance == 0:
            break

        # Try all possible moves and choose the one that minimizes the Euclidean distance
        min_distance = float('inf')
        best_move = None

        # Try moving up
        new_distance = math.sqrt((b_row - (a_row - 1))**2 + (b_col - a_col)**2)
        if new_distance < min_distance:
            min_distance = new_distance
            best_move = "up"

        # Try moving down
        new_distance = math.sqrt((b_row - (a_row + 1))**2 + (b_col - a_col)**2)
        if new_distance < min_distance:
            min_distance = new_distance
            best_move = "down"

        # Try moving left
        new_distance = math.sqrt((b_row - a_row)**2 + (b_col - (a_col - 1))**2)
        if new_distance < min_distance:
            min_distance = new_distance
            best_move = "left"

        # Try moving right
        new_distance = math.sqrt((b_row - a_row)**2 + (b_col - (a_col + 1))**2)
        if new_distance < min_distance:
            min_distance = new_distance
            best_move = "right"

        # If no better move is found, we're stuck
        if best_move is None:
            break

        # Update A's position based on the best move
        if best_move == "up":
            a_row -= 1
        elif best_move == "down":
            a_row += 1
        elif best_move == "left":
            a_col -= 1
        else:
            a_col += 1

        moves.append(best_move)
        distance = min_distance

    return moves

class FindOutStrategyAgent:
    def __init__(self, env_name):
        self.strategy = None
        self.game_turn = 0
        self.first_moves = ENV_FIRST_MOVES[env_name]
        self.other_player_total_stag_distance = 0
        self.other_player_total_plant_distance = 0
        self.initial_other_player_location = None


    def play_move(self, observation):
        self_location, other_player_location, stag_location, plant_location_1, plant_location_2 = unpack_observation(
            observation)

        if self.game_turn == 0:
            self.initial_other_player_location = other_player_location

        if self.game_turn < len(self.first_moves):
            return self.first_moves[self.game_turn]

        print(calculate_distance(self.initial_other_player_location, stag_location))
        print(self.other_player_total_stag_distance)

        # need some progress
        # see if the agent is moving the max amount towards the stag or the plant
        print(generate_moves(self.initial_other_player_location[0], self.initial_other_player_location[1], stag_location[0], stag_location[1], 4))



        print(f"stag distant: {self.other_player_total_stag_distance}, plant distance: {self.other_player_total_plant_distance}")






    def receive_feedback(self, old_obs, new_obs):
        self_location, other_player_location, stag_location, plant_location_1, plant_location_2 = unpack_observation(
            old_obs)

        new_self_location, new_other_player_location, new_stag_location, new_plant_location_1, new_plant_location_2 = unpack_observation(
            new_obs)

        # Calculate how much the other agent moved towards the stag and the plant
        stag_distance_delta = negative_distance_delta(other_player_location, new_other_player_location, stag_location)
        closest_plant = plant_location_1 if calculate_distance(plant_location_1, new_other_player_location) < calculate_distance(
            plant_location_2, new_other_player_location) else plant_location_2
        plant_distance_delta = negative_distance_delta(other_player_location, new_other_player_location, closest_plant)

        self.other_player_total_plant_distance += plant_distance_delta
        self.other_player_total_stag_distance += stag_distance_delta

        self.game_turn += 1