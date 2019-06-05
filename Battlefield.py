import time

from battleship.Field import print_all_battlefield


class Battlefield():
    """
    Place where two agents are playing N fights to be evaluated
    """

    def __init__(self, player1, player2, game, display=None):
        self.player1 = player1
        self.player2 = player2
        self.current_player = 1
        self.game = game
        self.display = display  # fusion display and verbose ?

    def single_fight(self, verbose=False):
        """
        Having two players play one game until it finished print result
        Used to evaluate performance of trained model against previous version
        """

        players = [self.player2, None, self.player1]
        ships, damages = self.game.get_clean_field()
        it = 0
        while self.game.check_finish_game(ships, self.current_player) == 0:
            it += 1

            visible_area = self.game.get_visible_area(ships, damages)

            if verbose:
                # assert(self.display)
                print("Turn ", str(it), "Player ", str(self.current_player))
                # print('ships')
                # print_battlefield(ships)
                # print('damages')
                # print_battlefield(damages)
                # print('visible_area')
                # print_battlefield(visible_area)
                print('Ships | Hits ')
                print_all_battlefield(ships, damages, visible_area)

            action = players[self.current_player + 1](ships, damages)  # lambda x,y

            valids = self.game.get_valid_moves(ships, damages)

            if valids[action] == 0:
                print('bad state', action)
                return -self.current_player

            ships, damages, self.current_player = self.game.get_next_state(ships, damages, self.current_player, action)

        if verbose:
            # assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.check_finish_game(ships, self.current_player)))
            # self.display(board)
        return self.game.check_finish_game(ships, -self.current_player)

    def new_vs_old(self, num, verbose):
        one_won = 0
        two_won = 0
        draws = 0
        for play_num in range(num):
            begin = time.time()
            game_result = self.single_fight(verbose=verbose)
            if game_result == 1:
                one_won += 1
            elif game_result == -1:
                two_won += 1
            else:
                draws += 1

            print('play', play_num, 'winner', game_result, 'time', time.time() - begin)

        return one_won, two_won, draws

    def fight(self, num, verbose=False):
        """
        Evaluate new model agains old, playing N games
        """
        num = int(num / 2)

        one_won, two_won, draws = self.new_vs_old(num, verbose)

        self.player1, self.player2 = self.player2, self.player1

        two_t, one_t, draws_t = self.new_vs_old(num, verbose)
        one_won += one_t
        two_won += two_t
        draws += draws_t

        return one_won, two_won, draws
