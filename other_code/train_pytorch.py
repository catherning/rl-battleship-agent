import os
import sys
import time
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
import torch
import torch.nn as nn

from battlefield import print_all_battlefield, Field
from other_code.mcts_pytorch import MCTS
from other_code.nnet_pytorch import ResidualNNet, CNNet


class Battlefield:
    """
    Place where two agents are playing N fights to be evaluated
    """

    def __init__(self, player1, player2, game, display=None):
        self.player1 = player1
        self.player2 = player2
        self.current_player = 1
        self.game = game
        self.display = display

    def single_fight(self):
        """
        Having two players play one game until it finished print result
        Used to evaluate performance of trained model against previous version
        """

        players = [self.player2, None, self.player1]
        ships, damages = self.game.get_clean_field()
        it = 0
        while self.game.check_finish_game(ships, self.current_player) == 0:
            it += 1

            if self.display:
                print("Turn ", str(it), "Player ", str(self.current_player))
                print_all_battlefield(ships, damages)

            action = players[self.current_player + 1](ships, damages)  # lambda x,y

            valids = self.game.get_valid_moves(ships, damages)

            if valids[action] == 0:
                print('bad state', action)
                return -self.current_player

            ships, damages, self.current_player = self.game.get_next_state(ships, damages, self.current_player, action)

        if self.display:
            print("Game over: Turn ", str(it), "Result ", str(self.game.check_finish_game(ships, self.current_player)))
            # self.display(board)
        return self.game.check_finish_game(ships, -self.current_player)

    def new_vs_old(self, num):
        one_won = 0
        two_won = 0
        draws = 0
        for play_num in range(num):
            begin = time.time()
            game_result = self.single_fight()
            if game_result == 1:
                one_won += 1
            elif game_result == -1:
                two_won += 1
            else:
                draws += 1

            print('play', play_num, 'winner', game_result, 'time', time.time() - begin)

        return one_won, two_won, draws

    def fight(self, num):
        """
        Evaluate new model agains old, playing N games, half of which new is player 1, other half, new is player 2
        """
        num = int(num / 2)

        one_won, two_won, draws = self.new_vs_old(num)

        self.player1, self.player2 = self.player2, self.player1

        two_t, one_t, draws_t = self.new_vs_old(num)
        one_won += one_t
        two_won += two_t
        draws += draws_t

        return one_won, two_won, draws


class General:
    """
    Logic of the agent training, NN will carry out action probabilities and states values
    """

    def __init__(self, field, nnet, learning_rate=1e-3):
        self.field = field
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.field)

        # Training => from nnet class
        self.epochs = 20
        self.batch_size = 64

        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr=learning_rate)
        self.history = []
        self.skip_first_play = False

        self.num_iterations = 5 # 250
        self.nb_train_examples = 10
        self.thr = 15
        self.update_thr = 0.55
        self.len_queue = 200 # 000
        self.sims = 15
        self.compare = 20
        self.cpuct = 1  # constant determining the level of exploration
        self.mcts = MCTS(self.field, self.nnet, self.sims, self.cpuct)

        self.checkpoint = './temp/'
        self.load_model = False
        self.models_folder = './models/'
        self.best_model_file = 'best.pth'
        self.len_history = 40
        self.last_steps = []
        self.display = False

    def run_episode(self):
        """
        Training episode - a single game of two agents that are trying to get as much hits as possible
        having missing hits(water) and drown ship cells
        """

        batch = []
        ships, damages = self.field.get_clean_field()

        self.player = 1
        step = 0

        self.hits = [0, None, 0]

        while True:
            step += 1

            visible_area = self.field.get_visible_area(ships, damages)
            temp = int(step < self.thr)

            a_prob = self.mcts.get_action_probabilities(ships, damages, t=temp)
            sym = self.field.get_rotations(visible_area, a_prob)
            for a, b in sym:
                batch.append([a, self.player, b, None])

            # Random play ??
            action = np.random.choice(len(a_prob), p=a_prob)
            ships, damages, player = self.field.get_next_state(ships, damages, self.player, action)

            if player == self.player:
                self.hits[player + 1] += 1

            end = self.field.check_finish_game(ships, self.player)
            self.player = player

            if end != 0:
                # player having more hits
                if self.hits[0] < self.hits[2]:
                    winner = 1
                    print('FIRST')
                elif self.hits[0] > self.hits[2]:
                    winner = -1
                    print('SECOND')
                else:
                    winner = 0
                    print('NOBODY')

                self.last_steps.append(step)

                if self.display:
                    print_all_battlefield(ships, damages)

                res = [(x[0], x[2], winner * ((-1) ** (x[1] != self.player))) for x in batch]
                return res

    def fight(self):
        """
        Execute N of fight episodes, collect results, perform model training and evaluation with previous version
        """

        for i in range(1, self.num_iterations + 1):
            print('iteration: ', str(i))

            # TODO should be able to simplify the next 10 lines: why iteration_train_examples, history and train_examples ?
            if not self.skip_first_play or i > 1:
                iteration_train_examples = deque([], maxlen=self.len_queue)

                for eps in range(self.nb_train_examples):
                    self.mcts = MCTS(self.field, self.nnet, self.sims, self.cpuct)  # XXX necessary to initialize it each time ?
                    iteration_train_examples += self.run_episode()

                self.history.append(iteration_train_examples)

            if len(self.history) > self.len_history:
                print("len(history) =", len(self.history), " => remove the oldest train_examples")
                self.history.pop(0)

            self.save_train_examples(i - 1)

            train_examples = []
            for e in self.history:
                train_examples.extend(e)
            shuffle(train_examples)

            self.nnet.save_checkpoint(self.optimizer, folder=self.checkpoint, filename='temp.pth')
            _ = self.pnet.load_checkpoint(folder=self.checkpoint, filename='temp.pth')

            pmcts = MCTS(self.field, self.pnet, self.sims, self.cpuct)

            self.pytorch_train(train_examples)

            nmcts = MCTS(self.field, self.nnet, self.sims, self.cpuct)

            print('FIGHT AGAINST PREVIOUS VERSION')
            battlefield = Battlefield(lambda x, y: np.argmax(pmcts.get_action_probabilities(x, y, t=0)),
                                      lambda x, y: np.argmax(nmcts.get_action_probabilities(x, y, t=0)), self.field,
                                      display=False)
            pwins, nwins, draws = battlefield.fight(self.compare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins > 0 and float(nwins) / (pwins + nwins) < self.update_thr:
                print('REJECTING NEW MODEL')
                optimizer_dict = self.nnet.load_checkpoint(folder=self.checkpoint, filename='temp.pth')
                self.optimizer.load_state_dict(optimizer_dict)
                #TODO maybe just keep previous opt ?

            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(self.optimizer, folder=self.checkpoint, filename=self.get_checkpoint_file(i))
                self.nnet.save_checkpoint(self.optimizer, folder=self.checkpoint, filename='best.pth')

                # Support stuff

    def get_checkpoint_file(self, iteration):
        '''
        Load checkpoint
        '''
        return 'checkpoint_' + str(iteration) + '.pth'

    def save_train_examples(self, iteration):
        """
        Save played games and results to reuse them when training model
        """

        folder = self.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.get_checkpoint_file(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.history)

    def load_train_examples(self):
        """
        Load played games and results, reuse them for training model
        """
        model_file = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examples_file = model_file + ".examples"
        if not os.path.isfile(examples_file):
            print(examples_file)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examples_file, "rb") as file:
                self.history = Unpickler(file).load()

            # examples based on the model were already collected (loaded)
            self.skip_first_play = True

    def pytorch_train(self, input_):
        # TODO

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            input_fields, target_a_prob, target_values = list(zip(*input_))
            target_a_prob = torch.LongTensor(target_a_prob)
            target_a = torch.max(target_a_prob, 1)[1]

            predicted_a_prob, predicted_values = self.nnet(torch.Tensor(input_fields))
            log_predicted_a_prob = torch.log(predicted_a_prob)

            nllloss = nn.NLLLoss()
            mse_loss = nn.MSELoss()

            loss = mse_loss(predicted_values, torch.Tensor(target_values))
            loss += nllloss(log_predicted_a_prob, target_a)
            loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch}/{self.epochs} - Training loss: {loss}")


if __name__ == "__main__":

    network = "cnn"

    f = Field(6)  # init battlefield

    if network == "residual":
        nnet = ResidualNNet(f)  # init NN
    elif network == "cnn":
        nnet = CNNet(f)

    # define where to store\get checkpoint and model
    checkpoint = './temp/'
    load_model = False
    models_folder = './models/'
    best_model_file = 'best.pth'

    if load_model:
        nnet.load_checkpoint(models_folder, best_model_file)

    g = General(f, nnet)  # init play and players

    if load_model:
        print("Load trainExamples from file")
        g.load_train_examples()

    g.fight()