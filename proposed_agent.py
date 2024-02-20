import math

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAND = 4


def calculate_distance(location_1, location_2):
    """
    Calculate the distance between two locations
    """
    return math.sqrt((location_1[0] - location_2[0]) ** 2 + (location_1[1] - location_2[1]) ** 2)


def negative_distance_delta(old_location, new_location, target_position):
    """
    Calculate the change in distance between the old location and the target position and the new location and the target position
    :param old_location: list of x and y coordinates
    :param new_location: list of x and y coordinates
    :param target_position: list of x and y coordinates
    :return: the change in distance
    """
    old_distance = calculate_distance(old_location, target_position)
    new_distance = calculate_distance(new_location, target_position)
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

    def __init__(self, location, stag_weight=1, plant_weight=1, player_weight=1, stag_multiplier=1,
                 plant_multiplier=0.5,
                 player_multiplier=1):
        self.location = location
        self.stag_weight = stag_weight
        self.plant_weight = plant_weight
        self.player_weight = player_weight
        self.stag_multiplier = stag_multiplier
        self.plant_multiplier = plant_multiplier
        self.player_multiplier = player_multiplier

    def update_parameters(self, old_obs, new_obs):
        # observation is a list of coordinates of self, other player, stag, and plants
        self_location, other_player_location, stag_location, plant_location_1, plant_location_2 = unpack_observation(
            old_obs)

        new_self_location, new_other_player_location, new_stag_location, new_plant_location_1, new_plant_location_2 = unpack_observation(
            new_obs)

        # update the parameters based on the change in distance to the stag, plant and the other player
        self.stag_weight += negative_distance_delta(other_player_location, new_other_player_location,
                                                    stag_location) * self.stag_multiplier
        self.plant_weight += negative_distance_delta(other_player_location, new_other_player_location,
                                                     plant_location_1) * self.plant_multiplier
        self.plant_weight += negative_distance_delta(other_player_location, new_other_player_location,
                                                     plant_location_2) * self.plant_multiplier
        self.player_weight += negative_distance_delta(other_player_location, new_other_player_location,
                                                      self_location) * self.player_multiplier

        # update the location
        self.location = new_self_location

    def calculate_action_reward(self, new_location, stag_location, plant_location_1, plant_location_2,
                                other_player_location):
        # calculate the reward for moving to a new location based on the distance to the stag, plant and the other player
        return (self.stag_weight * -calculate_distance(new_location, stag_location)
                + self.plant_weight * (-calculate_distance(new_location, plant_location_1)
                                       + -calculate_distance(new_location, plant_location_2))
                + self.player_weight * -calculate_distance(new_location, other_player_location))

    def choose_action(self, observation):
        self_location, other_player_location, stag_location, plant_location_1, plant_location_2 = unpack_observation(
            observation)

        print(f"weights: stag: {self.stag_weight}, plant: {self.plant_weight}, player: {self.player_weight}")

        # get the expected reward for each action
        up_reward = self.calculate_action_reward([self_location[0], self_location[1] - 1], stag_location,
                                                 plant_location_1, plant_location_2, other_player_location)
        down_reward = self.calculate_action_reward([self_location[0], self_location[1] + 1],
                                                   stag_location, plant_location_1, plant_location_2,
                                                   other_player_location)
        left_reward = self.calculate_action_reward([self_location[0] - 1, self_location[1]],
                                                   stag_location, plant_location_1, plant_location_2,
                                                   other_player_location)
        right_reward = self.calculate_action_reward([self_location[0] + 1, self_location[1]],
                                                    stag_location, plant_location_1, plant_location_2,
                                                    other_player_location)
        still_reward = self.calculate_action_reward(self_location, stag_location, plant_location_1, plant_location_2,
                                                    other_player_location)

        print(
            f"Up reward: {up_reward}, Down reward: {down_reward}, Left reward: {left_reward}, Right reward: {right_reward}, Still reward: {still_reward}")

        # choose the action with the highest expected reward
        return [left_reward, down_reward, right_reward, up_reward, still_reward].index(
            max(left_reward, down_reward, right_reward, up_reward, still_reward))
