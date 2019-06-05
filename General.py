import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np

from Battlefield import Battlefield
from MCTS import MCTS


class General():
    '''
    Logic of the agent training, NN will carry out action probabilities and states values
    '''

    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)
        self.history = []
        self.skip_first_play = False

        self.num_iterations = 250
        self.epochs = 50
        self.thr = 15
        self.update_thr = 0.55
        self.len_queue = 200000
        self.sims = 15
        self.compare = 40
        self.cpuct = 1  # constant determining the level of exploration
        self.mcts = MCTS(self.game, self.nnet, self.sims, self.cpuct)
        self.checkpoint = './temp/'
        self.load_model = False
        self.models_folder = './models/'
        self.best_model_file = 'best.pth.tar'
        self.len_history = 40
        self.last_steps = []

    def run_episode(self):
        '''
        Training episode - a single game of two agents that are trying to get as much hits as possible
        having missing hits(water) and drown ship cells
        '''

        batch = []
        ships, damages = self.game.get_clean_field()

        self.player = 1
        step = 0

        self.hits = [0, None, 0]

        while True:
            step += 1

            visible_area = self.game.get_visible_area(ships, damages)
            temp = int(step < self.thr)

            a_prob = self.mcts.get_action_probabilities(ships, damages, t=temp)
            sym = self.game.get_rotations(visible_area, a_prob)
            for a, b in sym:
                batch.append([a, self.player, b, None])

            action = np.random.choice(len(a_prob), p=a_prob)
            ships, damages, player = self.game.get_next_state(ships, damages, self.player, action)

            if player == self.player:
                self.hits[player + 1] += 1

            # print('player, step', self.player, self.hits[self.player+1], step)
            # print('Ships | Hits ')
            # print_all_battlefield(ships, damages, visible_area)
            # visible_area = self.game.get_visible_area(ships, damages)
            # print('player, step', self.player, self.hits[0], self.hits[2], step)
            # print('ships')
            # print_battlefield(ships)
            # print('damages')
            # print_battlefield(damages)
            # print('visible_area')
            # print_battlefield(visible_area)

            end = self.game.check_finish_game(ships, self.player)
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

                # print('player, step', self.player, self.hits[self.player+1], step, sum(self.last_steps)/len(self.last_steps))
                # print('Ships')
                # print_battlefield(ships)
                # print('Hits')
                # print_battlefield(damages)
                # print('Visible_area')
                # visible_area = self.game.get_visible_area(ships, damages)
                # print_battlefield(visible_area)
                # print('Ships | Hits ')
                # print_all_battlefield(ships, damages, visible_area)

                res = [(x[0], x[2], winner * ((-1) ** (x[1] != self.player))) for x in batch]
                return res

    def fight(self):
        '''
        Execute N of fight episodes, collect results, perform model training and evaluation with previous version
        '''

        for i in range(1, self.num_iterations + 1):
            print('iteration: ', str(i))

            if not self.skip_first_play or i > 1:
                iterationTrainExamples = deque([], maxlen=self.len_queue)
                # end = time.time()

                for eps in range(self.epochs):
                    self.mcts = MCTS(self.game, self.nnet, self.sims, self.cpuct)
                    iterationTrainExamples += self.run_episode()

                    # end = time.time()

                self.history.append(iterationTrainExamples)

            if len(self.history) > self.len_history:
                print("len(history) =", len(self.history), " => remove the oldest trainExamples")
                self.history.pop(0)

            self.saveTrainExamples(i - 1)

            trainExamples = []
            for e in self.history:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self.nnet.save_checkpoint(folder=self.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.checkpoint, filename='temp.pth.tar')

            pmcts = MCTS(self.game, self.pnet, self.sims, self.cpuct)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.sims, self.cpuct)

            print('FIGHT AGAINST PREVIOUS VERSION')
            battlefield = Battlefield(lambda x, y: np.argmax(pmcts.get_action_probabilities(x, y, t=0)),
                                      lambda x, y: np.argmax(nmcts.get_action_probabilities(x, y, t=0)), self.game)
            pwins, nwins, draws = battlefield.fight(self.compare, verbose=True)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins > 0 and float(nwins) / (pwins + nwins) < self.update_thr:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.checkpoint, filename='best.pth.tar')

                # Support stuff

    def getCheckpointFile(self, iteration):
        '''
        Load checkpoint
        '''
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        '''
        Save played games and results to reuse them when training model
        '''

        folder = self.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.history)
        f.closed

    def loadTrainExamples(self):
        '''
        Load played games and results, reuse them for training model
        '''
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.history = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
