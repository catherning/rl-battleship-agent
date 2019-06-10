import math

import numpy as np

from battlefield import Field

eps = 1e-8


class MCTS:
    '''
    Cloned from:
    https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py
    '''

    # https://deepmind.com/documents/119/agz_unformatted_nature.pdf
    def __init__(self, field: Field, nnet, sims, cpuct):
        self.field = field
        self.nnet = nnet
        self.sims = sims
        self.cpuct = cpuct  # constant determining the level of exploration
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.times_s_a_visited = {}  # stores #times edge s,a was visited
        self.times_s_visited = {}  # stores #times board s was visited
        self.init_policy = {}  # stores initial policy (returned by neural net)
        self.game_ended_s = {}  # stores game.getGameEnded ended for board s
        self.validmoves_s = {}  # stores game.getValidMoves for board s

    def get_action_probabilities(self, ships, damages, t=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to times_s_a_visited[(s,a)]**(1./temp)
        """

        # Damages is own state_matrix: the progress we are at
        # ships is other player's ship_matrix

        # Create our supposition of where the ships are, according to the hit cells


        for i in range(self.sims):
            self.search(ships, damages)  # , 0, 500)

        visible_area = self.field.get_visible_area(ships, damages)
        counts = []
        s = np.array(visible_area).tostring()
        for a in range(self.field.get_num_actions()):
            visit_count = 0
            if (s, a) in self.times_s_a_visited:
                visit_count = self.times_s_a_visited[(s, a)]

            counts.append(visit_count)

        if t == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / t) for x in counts]
        probs = [x / float(sum(counts)) for x in counts]
        return probs

    def search(self, ships, damages):  # , depth=0, depth_limit=None):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of times_s_visited, times_s_a_visited, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        visible_area = self.field.get_visible_area(ships, damages)
        # print(visible_area)
        s = np.array(visible_area).tostring()

        if s not in self.game_ended_s:
            self.game_ended_s[s] = self.field.check_finish_game(ships, 1)
        if self.game_ended_s[s] != 0:
            # terminal node
            return -self.game_ended_s[s]

        # depth_exceeded = depth_limit is not None and depth > depth_limit
        if s not in self.init_policy:  # or depth_exceeded:
            # leaf node

            self.init_policy[s], v = self.nnet.predict(visible_area)
            valids = self.field.get_valid_moves(ships, damages)
            self.init_policy[s] = self.init_policy[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.init_policy[s])
            if sum_Ps_s > 0:
                self.init_policy[s] /= sum_Ps_s
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, do workaround.")
                self.init_policy[s] = self.init_policy[s] + valids
                self.init_policy[s] /= np.sum(self.init_policy[s])

            self.validmoves_s[s] = valids
            self.times_s_visited[s] = 0
            return -v

        valids = self.validmoves_s[s]
        cur_best = -float('inf')
        best_act = -1

        for a in range(self.field.get_num_actions()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.cpuct * self.init_policy[s][a] * math.sqrt(self.times_s_visited[s]) / (1 + self.times_s_a_visited[(s, a)])
                else:
                    u = self.cpuct * self.init_policy[s][a] * math.sqrt(self.times_s_visited[s] + eps)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_d, next_player = self.field.get_next_state(ships, damages, 1, a)

        v = self.search(ships, next_d)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.times_s_a_visited[(s, a)] * self.Qsa[(s, a)] + v) / (self.times_s_a_visited[(s, a)] + 1)
            self.times_s_a_visited[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.times_s_a_visited[(s, a)] = 1

        self.times_s_visited[s] += 1
        return -v
