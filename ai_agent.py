import numpy as np

import battleship


# from mcts import MCTS


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

        # 3 channels for now, 1 for not touched, 1 for missed, 1 for hit
        input_ = np.zeros((size_field, size_field,3))
        for i, state in enumerate(battleship.CellState):
            cells_in_state = np.argwhere(opponent_field_state_matrix == state)
            for cell in cells_in_state:
                input_[cell[0]][cell[1]][i] = 1

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


        print(f"Chose action: {target}")

        return target
