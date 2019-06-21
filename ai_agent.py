import numpy as np
import torch
from torch.nn import functional

import battleship
from battleship import CellState


# noinspection PyTypeChecker
def convert_cell_states_to_channels(cell_state_tensor):
    game_field_size, game_field_size = cell_state_tensor.shape

    num_channels = len(CellState)

    output = torch.zeros([num_channels, game_field_size, game_field_size])
    for state_channel_index, state in enumerate(CellState):
        output[state_channel_index, cell_state_tensor == state.value] = 1

    return output


def sample_policy_matrix(policy_matrix: torch.Tensor, num_samples=1):
    height, width = policy_matrix.shape
    policy_1d = policy_matrix.view(-1)
    sample_1d_index = torch.multinomial(policy_1d, num_samples=num_samples)
    sample_y_index = torch.floor(sample_1d_index / width)
    sample_x_index = sample_1d_index % width
    return sample_y_index, sample_x_index


def remove_impossible_moves(policy_1d: torch.Tensor, opponent_field_state_tensor_1d: torch.Tensor):
    possible_firing_targets = opponent_field_state_tensor_1d == battleship.CellState.UNTOUCHED.value

    filtered_policy_matrix = torch.zeros(policy_1d.shape)
    filtered_policy_matrix[possible_firing_targets] = policy_1d[possible_firing_targets]

    # for y, x in possible_firing_targets:
    #     filtered_policy_matrix[y, x] = policy_1d[y, x]

    normalized_policy_matrix = filtered_policy_matrix / filtered_policy_matrix.sum()

    assert 0.999 < normalized_policy_matrix.sum() < 1.0001
    return normalized_policy_matrix


class AIAgent(battleship.Player):
    def __init__(self, network: torch.nn.Module):
        super().__init__()
        self.network = network

        self.policy_history = []
        self.action_history = []
        self.reward_history = []
        self.state_history = []
        self.training = False

    def train(self, mode=True):
        self.network.train(mode)
        self.training = mode

    def eval(self):
        self.network.eval()
        self.training = False

    def choose_firing_target(self, opponent_field_state_matrix):
        """
        Choose and return a firing target.

        :param opponent_field_state_matrix: the current state of the game
        :return
        """

        opponent_field_state_tensor_2d = torch.tensor(opponent_field_state_matrix)  # shape = [field_size, field_size]
        field_height, field_width = opponent_field_state_tensor_2d.shape
        opponent_field_state_tensor_1d = opponent_field_state_tensor_2d.view(-1)  # shape = [field_height * field_width]

        network_input = convert_cell_states_to_channels(opponent_field_state_tensor_2d)
        self.state_history.append(network_input)
        network_input_batch = network_input.unsqueeze(0)  # shape = [batch_size, channels, field_size, field_size]

        output_policy_1d, _ = self.network(network_input_batch)  # shape = [batch_size, field_height * field_width]
        output_policy_1d = output_policy_1d.squeeze(dim=0)  # shape = [field_height * field_width]
        filtered_policy_1d = remove_impossible_moves(output_policy_1d, opponent_field_state_tensor_1d)

        if self.training:
            sample_1d_index = int(torch.multinomial(filtered_policy_1d, num_samples=1))
        else:
            sample_1d_index = int(filtered_policy_1d.argmax())
        firing_target_y = sample_1d_index // field_width
        firing_target_x = sample_1d_index % field_width
        assert opponent_field_state_tensor_2d[firing_target_y, firing_target_x] == battleship.CellState.UNTOUCHED.value

        self.action_history.append(sample_1d_index)
        self.policy_history.append(output_policy_1d)

        return firing_target_y, firing_target_x

    def inform_about_result(self, attack_succeeded):
        if attack_succeeded:
            reward = 1
        else:
            reward = -1
        self.reward_history.append(reward)