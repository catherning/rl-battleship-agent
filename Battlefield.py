import numpy as np

import time
from battleship.Field import print_battlefield, print_all_battlefield

class Battlefield():
    '''
    Place where two agents are playing N fights to be evaluated
    '''
    
    def __init__(self, player1, player2, game, display=None):
        self.player1 = player1
        self.player2 = player2
        self.current_player = 1
        self.game = game
        self.display = display

    def single_fight(self, verbose=False):
        '''
        Having two players play one game until it finished print result
        Used to evaluate performance of trained model against previous version
        '''
        
        players = [self.player2, None, self.player1]
        ships, damages = self.game.get_clean_field()
        it = 0
        while self.game.check_finish_game(ships, self.current_player)==0:
            it+=1

            visible_area = self.game.get_visible_area(ships, damages)
            
            if verbose:
                #assert(self.display)
                print("Turn ", str(it), "Player ", str(self.current_player))
                #print('ships')
                #print_battlefield(ships)
                #print('damages')
                #print_battlefield(damages)
                #print('visible_area')
                #print_battlefield(visible_area)
                print('Ships | Hits ')
                print_all_battlefield(ships, damages, visible_area)
                
            action = players[self.current_player + 1](ships, damages) # lambda x,y 

            valids = self.game.get_valid_moves(ships, damages)

            if valids[action]==0:
                print('bad state', action)
                return -self.current_player
                
            ships, damages, self.current_player = self.game.get_next_state(ships, damages, self.current_player, action)
            
        if verbose:
            #assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.check_finish_game(ships, self.current_player)))
            #self.display(board)
        return self.game.check_finish_game(ships, -self.current_player)

    def fight(self, num, verbose=False):
        '''
        Evaluate new model agains old, playing N games
        '''
        
        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for play_num in range(num):
            gameResult = self.single_fight(verbose=verbose)
            if gameResult==1:
                oneWon+=1
            elif gameResult==-1:
                twoWon+=1
            else:
                draws+=1

            eps += 1
            print('play', play_num, 'winner',gameResult,'time', time.time() - end)
            end = time.time()


        self.player1, self.player2 = self.player2, self.player1
        
        for play_num in range(num):
            gameResult = self.single_fight(verbose=verbose)
            if gameResult==-1:
                oneWon+=1                
            elif gameResult==1:
                twoWon+=1
            else:
                draws+=1

            eps += 1
            print('play', play_num, 'winner',gameResult,'time', time.time() - end)
            end = time.time()


        return oneWon, twoWon, draws
