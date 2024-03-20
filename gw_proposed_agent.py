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
    return math.sqrt(math.pow((location_1[0] - location_2[0]), 2) + math.pow((location_1[1] - location_2[1]), 2))
    # return abs((location_1[0]) - location_2[0]) + abs(location_1[1] - location_2[1])


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
    # if user stays on the stag, there should be a positive reward
    if old_distance == 0 and new_distance == 0:
        return 1
    return -(new_distance - old_distance)


def unpack_observation(observation):
    """
    Unpack the observation into the coordinates of the self, other player, stag, and plants
    :param observation: list of coordinates
    :return: the coordinates of the self, other player, stag, and plants
    """
    observation_int32 = observation.astype('int32')  # sometimes the observation is uint8, which causes overflow
    self_location = observation_int32[0:2]
    other_player_location = observation_int32[2:4]
    stag_location = observation_int32[4:6]
    plant_location_1 = observation_int32[6:8]
    plant_location_2 = observation_int32[8:10]
    return self_location, other_player_location, stag_location, plant_location_1, plant_location_2


def normalize_deltas(stag_weight_delta, plant_weight_delta, player_weight_delta):
    # print(
    #     f"stag_weight_delta: {stag_weight_delta}, plant_weight_delta: {plant_weight_delta}, player_weight_delta: {player_weight_delta}")
    total_delta = abs(stag_weight_delta) + abs(plant_weight_delta) + abs(player_weight_delta)
    if total_delta != 0:
        return stag_weight_delta / total_delta, plant_weight_delta / total_delta, player_weight_delta / total_delta
    else:
        # Set the deltas to zero if the total delta is zero
        return 0, 0, 0


class ProposedAgent:
    """
    An agent to play the stag hunt game, which only considers its own location as the state
    have parameters of how much it want to approach the stag, plant and the other player
    the parameters are updated based on the other player's actions
    """

    def __init__(self, location, stag_weight=1, plant_weight=1, player_weight=1,
                 stag_leanring_rate=1, plant_learning_rate=1, player_learning_rate=1):
        self.location = location
        self.stag_weight = stag_weight
        self.plant_weight = plant_weight
        self.player_weight = player_weight
        self.stag_learning_rate = stag_leanring_rate
        self.plant_learning_rate = plant_learning_rate
        self.player_learning_rate = player_learning_rate

    def normalize_weights(self):
        total_weight = self.stag_weight + self.plant_weight + self.player_weight
        if total_weight != 0:
            self.stag_weight /= total_weight
            self.plant_weight /= total_weight
            self.player_weight /= total_weight
        else:
            # Set the weights to default values if the total weight is zero
            self.stag_weight = 1
            self.plant_weight = 1
            self.player_weight = 1

    def update_parameters(self, old_obs, new_obs):
        # observation is a list of coordinates of self, other player, stag, and plants
        self_location, other_player_location, stag_location, plant_location_1, plant_location_2 = unpack_observation(
            old_obs)

        new_self_location, new_other_player_location, new_stag_location, new_plant_location_1, new_plant_location_2 = unpack_observation(
            new_obs)

        # Calculate the change in each weight
        stag_weight_delta = negative_distance_delta(other_player_location, new_other_player_location,
                                                    stag_location) * self.stag_learning_rate
        # Focus on the plant that is closer to the other player
        closest_plant_location = plant_location_1 if calculate_distance(plant_location_1,
                                                                        new_other_player_location) < calculate_distance(
            plant_location_2, new_other_player_location) else plant_location_2
        plant_weight_delta = negative_distance_delta(other_player_location, new_other_player_location,
                                                     closest_plant_location) * self.plant_learning_rate
        player_weight_delta = negative_distance_delta(other_player_location, new_other_player_location,
                                                      self_location) * self.player_learning_rate

        # Normalize the deltas
        stag_weight_delta, plant_weight_delta, player_weight_delta = normalize_deltas(stag_weight_delta,
                                                                                      plant_weight_delta,
                                                                                      player_weight_delta)

        # Apply the changes to the weights
        self.stag_weight += stag_weight_delta
        self.plant_weight += plant_weight_delta
        self.player_weight += player_weight_delta

        # Normalize the weights after updating them
        self.normalize_weights()

        # update the location
        self.location = new_self_location

    def calculate_action_reward(self, new_location, stag_location, plant_location_1, plant_location_2,
                                other_player_location):
        # calculate the reward for moving to a new location based on the distance to the stag, plant and the other player
        return (self.stag_weight * -calculate_distance(new_location, stag_location)
                + self.plant_weight * max(-calculate_distance(new_location, plant_location_1),
                                          -calculate_distance(new_location, plant_location_2))
                + self.player_weight * -calculate_distance(new_location, other_player_location))

    def choose_action(self, observation):
        self_location, other_player_location, stag_location, plant_location_1, plant_location_2 = unpack_observation(
            observation)

        # print(f"weights: stag: {self.stag_weight}, plant: {self.plant_weight}, player: {self.player_weight}")

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

        # print(
        #     f"Up reward: {up_reward}, Down reward: {down_reward}, Left reward: {left_reward}, Right reward: {right_reward}, Still reward: {still_reward}")

        # choose the action with the highest expected reward
        return [left_reward, down_reward, right_reward, up_reward, still_reward].index(
            max(left_reward, down_reward, right_reward, up_reward, still_reward))
