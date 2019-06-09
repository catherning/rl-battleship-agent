from battleship import Player
from nnet import CNNet, ResidualNNet

class AIAgent(Player):
    def __init__(self, nnet):
        self.nnet = nnet

    def choose_firing_target(self, opponent_field_state_matrix):
        """
        Choose and return a firing target.

        :param opponent_field_state_matrix: the current state of the game
        :return
        """

        possible_firing_targets = list(zip(*np.where(opponent_field_state_matrix == CellState.UNTOUCHED)))
        # todo: change to argwhere
        target_y, target_x = random.choice(possible_firing_targets)
        return target_y, target_x