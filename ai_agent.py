import numpy as np
import torch

import battleship
from battleship import CellState


# from mcts import MCTS

# noinspection PyTypeChecker
def convert_cell_states_to_channels(cell_state_tensor):
    game_field_size, game_field_size = cell_state_tensor.shape

    num_channels = len(CellState)

    output = torch.zeros([num_channels, game_field_size, game_field_size])
    for state_channel_index, state in enumerate(CellState):
        output[state_channel_index, cell_state_tensor == state.value] = 1

    return output


class AIAgent(battleship.Player):
    def __init__(self, network):
        super().__init__()
        self.network = network

        self.policy_history = []
        self.action_history = []
        self.reward_history = []
        self.state_history = []

    def choose_firing_target(self, opponent_field_state_matrix):
        """
        Choose and return a firing target.

        :param opponent_field_state_matrix: the current state of the game
        :return
        """

        possible_firing_targets = torch.tensor(np.argwhere(opponent_field_state_matrix == battleship.CellState.UNTOUCHED))

        opponent_field_state_tensor = torch.tensor(opponent_field_state_matrix)
        # opponent_field_state_tensor: shape = [field_size, field_size]
        field_size, _ = opponent_field_state_tensor.shape

        input_ = convert_cell_states_to_channels(opponent_field_state_tensor)
        self.state_history.append(input_)
        input_ = input_.unsqueeze(0)  # batch_size = 1
        # input: shape = [batch_size, channels, field_size, field_size]

        # temp = 1
        found_firing_target = False
        # try_count = 0
        while not found_firing_target:
            output_policy, _ = self.network(input_)
            action = torch.multinomial(output_policy, num_samples=1)
            firing_target_y = action % field_size
            firing_target_x = int(action / field_size)

            if opponent_field_state_tensor[firing_target_y, firing_target_x] == battleship.CellState.UNTOUCHED.value:
                found_firing_target = True

            # if try_count > 0:
            #     print(f"Wrong cell chosen by the AI: {target}")
            # try_count += 1

        # print(f"Chose action: {target}")

        self.action_history.append(action)
        self.policy_history.append(output_policy)

        return firing_target_y, firing_target_x  #.reshape((size_field, size_field))

    def inform_about_result(self, attack_succeeded):
        if attack_succeeded:
            reward = 1
        else:
            reward = -1
        self.reward_history.append(reward)