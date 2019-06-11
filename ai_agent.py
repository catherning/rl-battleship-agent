import numpy as np

import battleship


# from mcts import MCTS

def state_to_nnet_input(field):
    size_field = len(field)
    input_ = np.zeros((3, size_field, size_field)) # change position of 3 and of [i] depending on Keras or Pytorch
    for i, state in enumerate(battleship.CellState):
        cells_in_state = np.argwhere(field == state)
        for cell in cells_in_state:
            input_[i][cell[0]][cell[1]] = 1

    return input_

class AIAgent(battleship.Player):
    def __init__(self, nnet):
        self.nnet = nnet

        # MCTS parameters
        # self.sims = 15
        # self.cpuct = 1  # constant determining the level of exploration
        # self.mcts = MCTS(self.nnet, self.sims, self.cpuct)

    def choose_firing_target(self, opponent_field_state_matrix):
        """
        Choose and return a firing target.

        :param opponent_field_state_matrix: the current state of the game
        :return
        """

        size_field = len(opponent_field_state_matrix)
        possible_firing_targets = np.argwhere(opponent_field_state_matrix == battleship.CellState.UNTOUCHED).tolist()

        input_ = state_to_nnet_input(opponent_field_state_matrix)

        # temp = 1
        target = [size_field, size_field]
        # try_count = 0
        while list(target) not in possible_firing_targets:
            a_prob, _ = self.nnet.predict(input_)
            try:
                norm_prob = a_prob / a_prob.sum(axis=1, keepdims=1)
            except np.core._internal.AxisError:
                norm_prob = a_prob / a_prob.sum()

            action = np.random.choice(len(norm_prob), p=norm_prob)
            target = (action % size_field, int(action / size_field))

            # if try_count > 0:
            #     print(f"Wrong cell chosen by the AI: {target}")
            # try_count += 1

        # print(f"Chose action: {target}")

        return target[0], target[1], norm_prob #.reshape((size_field, size_field))
