import random
import re
from copy import deepcopy
from enum import Enum
from math import ceil, floor
from typing import Iterable, List

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


class RandomAgent(Player):
    def choose_firing_target(self, opponent_field_state_matrix):
        possible_firing_targets = list(zip(*np.where(opponent_field_state_matrix == CellState.UNTOUCHED)))
        # todo: change to argwhere
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


class GameResult(object):
    def __init__(
            self,
            winner,
            # todo: add more game info
    ):
        self.winner = winner


class Game(object):
    def __init__(
            self,
            player_1: Player,
            player_2: Player,
            field_size,
            ship_sizes: List[int],
            ship_counts: List[int],
            num_random_attempts=50,
    ):
        self._player_1 = player_1
        self._player_2 = player_2
        self._active_player = player_1

        # The game of battleship is kind of like two different games played at the same time. Both players have a
        # FieldState which comprises of their ships and the shots that have been fired at it. In each turn, a player
        # can fire at their opponent's FieldState.
        self.field_state_per_player = {
            player_1: FieldState(field_size=field_size),
            player_2: FieldState(field_size=field_size)
        }

        for field_state in self.field_state_per_player.values():
            field_state.generate_ships(
                ship_sizes=ship_sizes,
                ship_counts=ship_counts,
                num_attempts=num_random_attempts,
            )

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
        return self.get_opponent(self._active_player)

    def play_out(self):
        """
        Play out a full game and return the result.
        :return:
        """
        # players take turns firing at their opponent
        # if a full ship has been hit, the owner of the ship must announce this

        finished = False
        winner = None

        while not finished:
            if type(self._active_player) is HumanPlayer:
                print(self.field_state_per_player[self._active_player])

            opponent_field = self.field_state_per_player[self.non_active_player]
            firing_target_y, firing_target_x = self._active_player.choose_firing_target(
                opponent_field_state_matrix=opponent_field.state_matrix
            )
            succeeded = opponent_field.fire_at_target(y=firing_target_y, x=firing_target_x)
            # todo: if it's an RL agent, return the reward
            # self._active_player.give_reward(succeeded)

            if opponent_field.is_alive:
                self.switch_active_player()
            else:
                finished = True
                winner = self._active_player

        return GameResult(winner=winner)


if __name__ == '__main__':
    g = Game(player_1=HumanPlayer(), player_2=RandomAgent(), field_size=10, ship_sizes=[4, 3, 2], ship_counts=[1, 2, 3])
    result = g.play_out()
    print(g.field_state_per_player[result.winner])
    print()
    print(g.field_state_per_player[g.get_opponent(result.winner)])

class Field:
    """
    Logic of the game, how ships should be placed and how winner determined

    Assumptions and rules:
    1. Ships align strategy is out of scope NN
    2. Agent is given with square area, where
        a) 0 means no data, fire at the cell is possible
        b) 1 means hit the ship, so agent should learn where to hit next to finish the ship
        c) -1 means shooting water with no success
    3. Ships cannot be placed near to the border => fixme WHY THE HELL NOT ?
    4. There should be distance of >1 between ships => fixme again, remove these rules ?
    5. Only ships with sizes more than 1 are available, because it is impossible to gain
       insights about places of single cell ships, random will perform the same as NN => xxx still to consider ? because original game has them
    6. To win, player must hit more ship cells
    """

    def __init__(self, n):

        self.n = n
        self.placing_limit = n * n

        # todo create sizes and counts depending of board size n
        self.ship_sizes = [2, 3]  # , 4]
        self.ship_count = [2, 1]  # , 1]
        # ship_sizes = [2, 3, 4, 5]
        # ship_count = [4, 3, 2, 1]
        # ship_sizes = [2, 3, 4]
        # ship_count = [4, 2, 1]

        self.field_of_ships = np.zeros([n, n])
        self.field_of_damages = np.zeros([n, n])
        self.generate_ships()
        self.num_alive_ship_cells = self.count_alive_ship_cells(self.field_of_ships)


    def write_placement(self, ships, x, y, z, new_ship_size):
        """
        Write ship signature onto the battlefield
        """

        x_range = range(x, x + (1 - z) + new_ship_size * z)
        y_range = range(y, y + z + new_ship_size * (1 - z))

        for x_i in x_range:
            for y_i in y_range:
                ships[x_i][y_i] = 1

    def check_placement(self, ships, x, y, z, new_ship_size):
        """
        Check whether it is possible to place the ship signature
        on battlefield or not
        """

        x_range = [x - 1, x + 2 + new_ship_size * z]
        y_range = [y - 1, y + 2 + new_ship_size * (1 - z)]

        for xi in range(x_range[0], x_range[1]):
            for yi in range(y_range[0], y_range[1]):
                if ships[xi][yi] != 0:
                    return False

        return True

    def place_ship(self, ships, new_ship_size):
        """
        Place one ship of the given size on the board
        """

        placed = False

        count_tries = 0
        while placed is False:
            count_tries += 1

            if count_tries > self.placing_limit:
                ships = np.zeros([self.n, self.n])
                count = 0

            z = floor(random.random() + 0.5)  # 0 == vertical or 1 == horizontal placement

            # avoid border and out of battlefield cells for x and y => todo change
            x = ceil(random.random() * self.n) - 2 - new_ship_size * z
            y = ceil(random.random() * self.n) - 2 - new_ship_size * (1 - z)

            # borders
            if x < 1 or y < 1 or x > self.n - 1 or y > self.n - 1:
                continue

            placed = self.check_placement(ships, x, y, z, new_ship_size)

            if placed is True:
                self.write_placement(ships, x, y, z, new_ship_size)
                break

    def generate_ships(self):
        """
        Place ships on the battlefield.
        Ship sizes and numbers of every ship are in static arrais of self
        """

        for s, c in zip(self.ship_sizes, self.ship_count):
            for i in range(0, c):
                self.place_ship(self.field_of_ships, s)

    def count_alive_ship_cells(self, ships):
        """
        Count ship cells that are still floating, cell[x][y] == 1
        """

        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if ships[x][y] == 1:
                    count += 1

        return count

    def fire(self, aim, player):
        """
        Execute action - fire at the specified position of the field
        """

        # print(self.ships)
        x = aim[0]
        y = aim[1]
        self.field_of_damages[x][y] = -1  # player + self.damages_name_offser

        # The chosen case has already been fired before => xxx shouldn't it know it with the model ?
        if self.field_of_ships[x][y] != 0:
            self.field_of_damages[x][y] = 1
            self.field_of_ships[x][y] = -1
            return player  # give next step

        return -player  # alternate player

    def get_allowed_moves(self):
        """
        Get list of locations where it is possible to execute action, undamaged cell[x][y] == 0
        """

        moves = []
        for y in range(self.n):
            for x in range(self.n):
                if self.field_of_damages[x][y] == 0:
                    moves.append([x, y])
                else:
                    pass

        return moves

    def get_clean_field(self):
        """
        Get initial state of the board - all ships are floating, noting is damaged
        """

        f = Field(self.n)
        return [np.array(f.field_of_ships), np.array(f.field_of_damages)]

    def get_field_size(self):
        """
        Get total number of cells in the environment
        """

        return (self.n, self.n)

    def get_num_actions(self):
        """
        Get number of possible actions, including 'pass' action
        """

        return self.n * self.n + 1

    def get_next_state(self, ships, damages, player, action):
        """
        Copy the battlefield and perform action, return resulted field and player
        """

        if action == self.n * self.n:
            return (ships, damages, -player)

        f = Field(self.n)
        f.field_of_ships = np.copy(np.array(ships))
        f.field_of_damages = np.copy(np.array(damages))
        aim = (int(action / self.n), action % self.n)
        player = f.fire(aim, player)

        return (f.field_of_ships, f.field_of_damages, player)

    def get_valid_moves(self, ships, damages):
        """
        Wrap possible actions into valid moves, perform on copied data
        """

        v = [0] * self.get_num_actions()
        f = Field(self.n)
        f.field_of_ships = np.copy(np.array(ships))
        f.field_of_damages = np.copy(np.array(damages))
        moves = f.get_allowed_moves()
        if len(moves) == 0:
            v[-1] = 1
            return np.array(v)

        for pair in moves:
            x = pair[0]
            y = pair[1]
            v[self.n * x + y] = 1

        return np.array(v)

    def check_finish_game(self, ships, player):
        """
        See if there is something left floating..
        """

        f = Field(self.n)
        if f.count_alive_ship_cells(ships) > 0:
            return 0
        else:
            return player

    def get_visible_area(self, ships, damages):
        """
        Compute the visible area - missing hits and drown ship cells,
        floating cells are not shown because of the strong foggy cloud :]
        """

        visible = [[0 for i in range(0, self.n)] for j in range(0, self.n)]
        for i in range(0, self.n):
            for j in range(0, self.n):
                if damages[i][j] != 0:
                    # success hits 1, elsewhere -1
                    visible[i][j] = damages[i][j]

                    # ? having 'damages' as output, is it enough to learn what to do ?

                    # if ships[i][j] != 0:
                    #    visible[i][j] = ships[i][j]
                    # else:
                    #    visible[i][j] = 1
        # print(visible)
        return np.array(visible)

    def get_rotations(self, arr, a_prob):
        a_prob_arr = np.reshape(a_prob[:-1], (self.n, self.n))
        r = []
        for i in range(1, 5):
            for j in [True, False]:
                new_arr = np.rot90(np.array(arr), i)
                new_a_prob = np.rot90(a_prob_arr, i)
                if j:
                    new_arr = np.fliplr(new_arr)
                    new_a_prob = np.fliplr(new_a_prob)
                r += [(new_arr, list(new_a_prob.ravel()) + [a_prob[-1]])]
        return r


def print_all_battlefield(ships, damages):
    """
    Print all of the state stuff
    """

    print('Ships | Hits ')

    n = len(ships[0])
    for y in range(n):
        print("| ", end="")

        for x in range(n):
            pos = ships[y][x]
            if pos == -1:
                print("x ", end="")  # dead ship cell
            elif pos == 1:
                print("o ", end="")  # alive ship cell
            else:
                print("- ", end="")

        print("|", end="")

        for x in range(n):
            pos = damages[y][x]
            if pos == -1:
                print("* ", end="")  # non-profit hit
            elif pos == 1:
                print("@ ", end="")  # good hit

            else:
                print("- ", end="")

        print("|")

    print("")
