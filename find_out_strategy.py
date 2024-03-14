from proposed_agent import unpack_observation, negative_distance_delta, calculate_distance, normalize_deltas

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

        print('calc')
        print(calculate_distance(self.initial_other_player_location, stag_location))
        print(self.other_player_total_stag_distance)
        if calculate_distance(self.initial_other_player_location, stag_location) == self.other_player_total_stag_distance:
            print("other player is going to the stag")

        # need some progress
        # see if the agent is moving the max amount towards the stag or the plant


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