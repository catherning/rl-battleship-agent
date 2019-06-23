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

    normalized_policy_matrix = filtered_policy_matrix / filtered_policy_matrix.sum()

    assert 0.999 < normalized_policy_matrix.sum() < 1.0001
    return normalized_policy_matrix


class HistoryList(list):
    def __init__(self, max_size):
        super().__init__()
        self.max_size = max_size

    def append(self, object) -> None:
        super().append(object)
        if len(self) > self.max_size:
            self.pop(0)


class AIAgent(battleship.Player):
    def __init__(self, network: torch.nn.Module, max_history_size):
        super().__init__()
        self.network = network

        self.observed_state_history = HistoryList(max_history_size)
        self.actual_field_with_untouched_ship_cells_history = HistoryList(max_history_size) #TODO remove ?
        self.actual_ship_position_labels_history = HistoryList(max_history_size)

        self.training = False
        self.new_experience = None

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

        network_input_batch = network_input.unsqueeze(0)  # shape = [batch_size, channels, field_size, field_size]

        output_policy_1d = self.network(network_input_batch).squeeze(dim=0)  # shape = [field_height * field_width]
        filtered_policy_1d = remove_impossible_moves(output_policy_1d, opponent_field_state_tensor_1d)

        if self.training:
            sample_1d_index = int(torch.multinomial(filtered_policy_1d, num_samples=1))
        else:
            sample_1d_index = int(filtered_policy_1d.argmax())
        firing_target_y = sample_1d_index // field_width
        firing_target_x = sample_1d_index % field_width
        assert opponent_field_state_tensor_2d[firing_target_y, firing_target_x] == battleship.CellState.UNTOUCHED.value

        # save experiences for later training:
        self.observed_state_history.append(network_input)

        return firing_target_y, firing_target_x

    def inform_about_result(self, attack_succeeded, actual_opponent_field_state: battleship.FieldState):
        field_with_untouched_cells = actual_opponent_field_state.ship_field_matrix & (actual_opponent_field_state.state_matrix == battleship.CellState.UNTOUCHED.value)
        field_with_untouched_cells_2d = torch.tensor(
                field_with_untouched_cells.astype('float32')
        )
        field_with_untouched_cells_1d = field_with_untouched_cells_2d.view(-1)
        self.actual_field_with_untouched_ship_cells_history.append(field_with_untouched_cells_1d)
        ship_position_labels_1d = (field_with_untouched_cells_1d == battleship.CellOccupancy.HAS_SHIP.value).nonzero()
        self.actual_ship_position_labels_history.append(ship_position_labels_1d)
