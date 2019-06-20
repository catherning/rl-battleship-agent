import random
import re
from copy import deepcopy
from enum import Enum
from typing import List, Sized

import numpy as np


class Player(object):
    def __init__(self):
        pass

    def choose_firing_target(self, opponent_field_state_matrix):
        """
        Choose and return a firing target.

        :param opponent_field_state_matrix: the current state of the game
        :return: a tuple of y and x coordinates
        """
        raise NotImplementedError  # override this with RL agent, random agent, and user input player

    def inform_about_result(self, attack_succeeded):
        pass


class RandomAgent(Player):
    def choose_firing_target(self, opponent_field_state_matrix):
        possible_firing_targets = np.argwhere(opponent_field_state_matrix == CellState.UNTOUCHED)
        target_y, target_x = random.choice(possible_firing_targets)
        return target_y, target_x


class HumanPlayer(Player):
    def choose_firing_target(self, opponent_field_state_matrix):
        print(opponent_field_state_matrix)

        def input_a_target():
            y, x = re.match(r'(\d+)\s+(\d+)', string=input('Input a target as `y x`: ')).groups()
            y, x = int(y), int(x)

            if opponent_field_state_matrix[y, x] != CellState.UNTOUCHED:
                print('Wtf dude, forgot you already tried that one? Maybe try another one this time.')
                return input_a_target()
            return y, x

        return input_a_target()


class Orientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1


class CellState(Enum):
    """
    An enum that describes the state of a cell in a field:

    * MISSED: a shot has previously been fired at this cell, but the is no ship here.
    * UNTOUCHED: no shots have been fired at this cell yet.
    * HIT: a shot has previously been fired at this cell and there is a ship here.
    """

    UNTOUCHED = 0
    MISSED = 1
    HIT = 2

    def __eq__(self, other):
        if type(other) is CellState:
            return self.value == other.value
        elif type(other) is int:
            return self.value == other
        else:
            return False


class CellOccupancy(Enum):
    HAS_NO_SHIP = 0
    HAS_SHIP = 1

    def __eq__(self, other):
        if type(other) is CellState:
            return self.value == other.value
        elif type(other) is int:
            return self.value == other
        else:
            return False


class FieldState(object):
    def __init__(self, field_size: int):
        assert field_size > 0
        self.field_size = field_size
        self._ship_field_matrix = np.full(
            [self.field_size, self.field_size],
            fill_value=CellOccupancy.HAS_NO_SHIP.value
        )
        self._state_matrix = np.full(
            [self.field_size, self.field_size],
            fill_value=CellState.UNTOUCHED.value
        )

    def generate_ships(self, ship_sizes: List[int], ship_counts: List[int], num_attempts=50):
        assert len(ship_sizes) == len(ship_counts)

        for ship_size, ship_count in zip(ship_sizes, ship_counts):
            for _ in range(ship_count):
                succeeded = self.randomly_place_new_ship(ship_size=ship_size, num_attempts=num_attempts)
                if not succeeded:
                    return False
        return True

    def randomly_place_new_ship(self, ship_size: int, num_attempts):
        """
        Try to randomly place a ship of length `ship_size`. If no free place can be found for the new ship within
        `num_attempts` attempts of random placement, the function returns `False`.

        Usually the field is sparsely occupied, so randomly picking a place and then checking if it's free is quite
        efficient.

        :param ship_size:
        :param num_attempts:
        :return: a boolean indicating whether the operation succeeded or not
        """

        for _ in range(num_attempts):
            ship_orientation = random.choice([Orientation.HORIZONTAL, Orientation.VERTICAL])

            if ship_orientation == Orientation.HORIZONTAL:
                ship_width = ship_size
                ship_height = 1
            else:
                ship_width = 1
                ship_height = ship_size

            x = random.randint(0, self.field_size - ship_width)
            y = random.randint(0, self.field_size - ship_height)

            x_range = range(x, x + ship_width)
            y_range = range(y, y + ship_height)

            if 1 not in self._ship_field_matrix[y_range, x_range]:  # the ship location is still free
                self._ship_field_matrix[y_range, x_range] = 1
                return True  # succeeded to place the ship

        return False  # failed to randomly find a free location

    def get_num_actions(self):
        return self.field_size ** 2

    @property
    def state_matrix(self):
        """
        Return a matrix that represents the current state of the field from the opponent's perspective.
        0 represents previously MISSED shots
        1 represents UNTOUCHED locations
        2 represents previous successful shots

        :return:
        """
        return self._state_matrix

    def possible_moves(self):
        return np.argwhere(self.state_matrix == CellState.UNTOUCHED)

    @property
    def ship_field_matrix(self):
        return self._ship_field_matrix

    def __repr__(self):
        string = ''

        for y in range(self.field_size):
            for x in range(self.field_size):
                cell_occupancy = self.ship_field_matrix[y, x]
                cell_state = self.state_matrix[y, x]

                if cell_state == CellState.HIT:
                    string += 'x'  # hit ship cell
                elif cell_state == CellState.MISSED:
                    string += '@'  # a hit location with no ship
                elif cell_occupancy == CellOccupancy.HAS_SHIP:
                    string += 'o'  # a untouched ship cell
                else:
                    string += '~'  # an untouched location with no ship
            string += '\n'
        return string.strip('\n')

    def copy(self):
        return deepcopy(self)

    def fire_at_target(self, y, x):
        assert self.state_matrix[y, x] == CellState.UNTOUCHED

        if self.ship_field_matrix[y, x] == CellOccupancy.HAS_SHIP:
            self.state_matrix[y, x] = CellState.HIT.value
            return True
        else:
            self.state_matrix[y, x] = CellState.MISSED.value
            return False

    @property
    def is_alive(self):
        # todo: cache these variables (don't need to count every time, we can just keep track of the number)
        num_hit_cells = len(np.argwhere(self.state_matrix == CellState.HIT))
        num_ships = len(np.argwhere(self.ship_field_matrix == CellOccupancy.HAS_SHIP))
        return num_ships > num_hit_cells


class GameResult(object):
    def __init__(
            self,
            winner,
            num_turns,
            # todo: add more game info
    ):
        self.result_dict = {
            'winner': winner,
            'num_turns': num_turns
        }
        self.winner = winner
        self.num_turns = num_turns


class GameConfiguration(object):
    def __init__(
            self,
            field_size,
            ship_sizes: List[int],
            ship_counts: List[int],
            num_random_attempts=50,
    ):
        self.field_size = field_size
        self.ship_sizes = ship_sizes
        self.ship_counts = ship_counts
        self.num_random_attempts = num_random_attempts


class Game(object):
    def __init__(
            self,
            player_1: Player,
            player_2: Player,
            game_configuration: GameConfiguration,
    ):
        self._player_1 = player_1
        self._player_2 = player_2
        self._active_player = player_1

        # The game of battleship is kind of like two different games played at the same time. Both players have a
        # FieldState which comprises of their ships and the shots that have been fired at it. In each turn, a player
        # can fire at their opponent's FieldState.
        self.field_state_per_player = {
            player_1: FieldState(field_size=game_configuration.field_size),
            player_2: FieldState(field_size=game_configuration.field_size)
        }

        for field_state in self.field_state_per_player.values():
            generation_succeeded = field_state.generate_ships(
                ship_sizes=game_configuration.ship_sizes,
                ship_counts=game_configuration.ship_counts,
                num_attempts=game_configuration.num_random_attempts,
            )
            if not generation_succeeded:
                raise Exception("Can't generate ship on game field, try a large field or less ships")

    def get_opponent(self, player):
        if player == self._player_1:
            return self._player_2
        elif player == self._player_2:
            return self._player_1
        else:
            raise ValueError('The provided player is not part of this game.')

    def switch_active_player(self):
        self._active_player = self.non_active_player

    @property
    def non_active_player(self):
        return self.get_opponent(self.active_player)

    @property
    def active_player(self):
        return self._active_player

    def play_out(self):
        """
        Play out a full game and return the result.
        :return:
        """
        # players take turns firing at their opponent
        # if a full ship has been hit, the owner of the ship must announce this

        finished = False
        winner = None
        num_turns = 0

        while not finished:
            num_turns += 1

            if type(self.active_player) is HumanPlayer:
                print(self.field_state_per_player[self.active_player])

            opponent_field = self.field_state_per_player[self.non_active_player]

            firing_target_y, firing_target_x = self.active_player.choose_firing_target(
                opponent_field_state_matrix=opponent_field.state_matrix
            )
            succeeded = opponent_field.fire_at_target(y=firing_target_y, x=firing_target_x)
            self.active_player.inform_about_result(attack_succeeded=succeeded)

            if opponent_field.is_alive:
                self.switch_active_player()
            else:
                finished = True
                winner = self.active_player

        return GameResult(winner=winner, num_turns=num_turns)


class Tournament(object):
    def __init__(
            self,
            player_1: Player,
            player_2: Player,
            game_configuration: GameConfiguration,
            num_games: int,
    ):
        self.player_1 = player_1
        self.player_2 = player_2
        self.game_configuration = game_configuration
        self.num_games = num_games

    def play_out(self, verbose=False):
        tournament_results = []
        print('Starting tournament..')
        for i in range(self.num_games):
            game = Game(player_1=self.player_1, player_2=self.player_2, game_configuration=self.game_configuration)
            game_result = game.play_out()
            tournament_results.append(game_result)
            if verbose:
                print(f'Game {i} winner = {game_result.winner}')
        return tournament_results
