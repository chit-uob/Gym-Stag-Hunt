

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAND = 4

def negative_distance_delta(old_location, new_location, target_position):
    """
    Calculate the change in distance between the old location and the target position and the new location and the target position
    :param old_location: list of x and y coordinates
    :param new_location: list of x and y coordinates
    :param target_position: list of x and y coordinates
    :return: the change in distance
    """
    old_distance = ((old_location[0] - target_position[0])**2 + (old_location[1] - target_position[1])**2)**0.5
    new_distance = ((new_location[0] - target_position[0])**2 + (new_location[1] - target_position[1])**2)**0.5
    return -(new_distance - old_distance)

def unpack_observation(observation):
    """
    Unpack the observation into the coordinates of the self, other player, stag, and plants
    :param observation: list of coordinates
    :return: the coordinates of the self, other player, stag, and plants
    """
    self_location = observation[0:2]
    other_player_location = observation[2:4]
    stag_location = observation[4:6]
    plant_location_1 = observation[6:8]
    plant_location_2 = observation[8:10]
    return self_location, other_player_location, stag_location, plant_location_1, plant_location_2


class ProposedAgent:
    """
    An agent to play the stag hunt game, which only considers its own location as the state
    have parameters of how much it want to approach the stag, plant and the other player
    the parameters are updated based on the other player's actions
    """
    stag_multiplier = 1
    plant_multiplier = 1
    player_multiplier = 1

    def __init__(self, location, stag_weight, plant_weight, player_weight):
        self.location = location
        self.stag_weight = stag_weight
        self.plant_weight = plant_weight
        self.player_weight = player_weight

    def update_parameters(self, old_obs, new_obs):
        # observation is a list of coordinates of self, other player, stag, and plants
        self_location, other_player_location, stag_location, plant_location_1, plant_location_2 = unpack_observation(old_obs)

        new_self_location, new_other_player_location, new_stag_location, new_plant_location_1, new_plant_location_2 = unpack_observation(new_obs)

        # update the parameters based on the change in distance to the stag, plant and the other player
        self.stag_weight += negative_distance_delta(self_location, new_self_location, stag_location) * self.stag_multiplier
        self.plant_weight += negative_distance_delta(self_location, new_self_location, plant_location_1) * self.plant_multiplier
        self.plant_weight += negative_distance_delta(self_location, new_self_location, plant_location_2) * self.plant_multiplier
        self.player_weight += negative_distance_delta(self_location, new_self_location, other_player_location) * self.player_multiplier

        # update the location
        self.location = new_self_location

    def calculate_action_reward(self, old_location, new_location, stag_location, plant_location_1, plant_location_2, other_player_location):
        # calculate the change in distance to the stag, plant and the other player
        stag_change = negative_distance_delta(old_location, new_location, stag_location)
        plant_change = negative_distance_delta(old_location, new_location, plant_location_1) + negative_distance_delta(old_location, new_location, plant_location_2)
        player_change = negative_distance_delta(old_location, new_location, other_player_location)
        # return the reward
        return self.stag_weight * stag_change + self.plant_weight * plant_change + self.player_weight * player_change

    def choose_action(self, observation):
        self_location, other_player_location, stag_location, plant_location_1, plant_location_2 = unpack_observation(observation)

        print(f"Self location: {self_location}, Other player location: {other_player_location}, Stag location: {stag_location}, Plant 1 location: {plant_location_1}, Plant 2 location: {plant_location_2}")
        print(f"weights: stag: {self.stag_weight}, plant: {self.plant_weight}, player: {self.player_weight}")

        # get the expected reward for each action
        up_reward = self.calculate_action_reward(self_location, [self_location[0], self_location[1] - 1], stag_location, plant_location_1, plant_location_2, other_player_location)
        down_reward = self.calculate_action_reward(self_location, [self_location[0], self_location[1] + 1], stag_location, plant_location_1, plant_location_2, other_player_location)
        left_reward = self.calculate_action_reward(self_location, [self_location[0] - 1, self_location[1]], stag_location, plant_location_1, plant_location_2, other_player_location)
        right_reward = self.calculate_action_reward(self_location, [self_location[0] + 1, self_location[1]], stag_location, plant_location_1, plant_location_2, other_player_location)

        print(f"Up reward: {up_reward}, Down reward: {down_reward}, Left reward: {left_reward}, Right reward: {right_reward}")

        # choose the action with the highest expected reward
        if up_reward > down_reward and up_reward > left_reward and up_reward > right_reward:
            return UP
        elif down_reward > left_reward and down_reward > right_reward:
            return DOWN
        elif left_reward > right_reward:
            return LEFT
        else:
            return RIGHT
