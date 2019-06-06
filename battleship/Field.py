import random
from math import ceil, floor

import numpy as np


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

        self.ships = self.init_2d_arr(n)
        self.damages = self.init_2d_arr(n)
        # self.opponent_ships = self.init_ships(n)
        self.generate_ships()
        # self.generate_ships(self.opponent_ships)
        self.ship_cells = self.count_alive_ship_cells(self.ships)

        # todo create sizes and counts depending of board size n
        self.ship_sizes = [2, 3]  # , 4]
        self.ship_count = [2, 1]  # , 1]
        # ship_sizes = [2, 3, 4, 5]
        # ship_count = [4, 3, 2, 1]

        # ship_sizes = [2, 3, 4]
        # ship_count = [4, 2, 1]
        # damages_name_offser = 4

    def init_2d_arr(self, n):
        """
        Initialize 2D array with zeros
        """

        # todo use np.zeros here directly so that we don't convert to np.arrays each time later ? if we still use np
        # todo and if there's no place where it must not be np.array
        arr = [None] * n
        for i in range(n):
            arr[i] = [0] * n

        return arr

    def write_placement(self, ships, x, y, z, new_ship_size):
        """
        Write ship signature onto the battlefield
        """

        x_range = [x, x + (1 - z) + new_ship_size * z]
        y_range = [y, y + z + new_ship_size * (1 - z)]

        for xi in range(x_range[0], x_range[1]):
            for yi in range(y_range[0], y_range[1]):
                ships[xi][yi] = 1

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
                ships = self.init_2d_arr(self.n)
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
                self.place_ship(self.ships, s)

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
        self.damages[x][y] = -1  # player + self.damages_name_offser

        # The chosen case has already been fired before => xxx shouldn't it know it with the model ?
        if self.ships[x][y] != 0:
            self.damages[x][y] = 1
            self.ships[x][y] = -1
            return player  # give next step

        return -player  # alternate player

    def get_allowed_moves(self):
        """
        Get list of locations where it is possible to execute action, undamaged cell[x][y] == 0
        """

        moves = []
        for y in range(self.n):
            for x in range(self.n):
                if self.damages[x][y] == 0:
                    moves.append([x, y])
                else:
                    pass

        return moves

    def get_clean_field(self):
        """
        Get initial state of the board - all ships are floating, noting is damaged
        """

        f = Field(self.n)
        return [np.array(f.ships), np.array(f.damages)]

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
        f.ships = np.copy(np.array(ships))
        f.damages = np.copy(np.array(damages))
        aim = (int(action / self.n), action % self.n)
        player = f.fire(aim, player)

        return (f.ships, f.damages, player)

    def get_valid_moves(self, ships, damages):
        """
        Wrap possible actions into valid moves, perform on copied data
        """

        v = [0] * self.get_num_actions()
        f = Field(self.n)
        f.ships = np.copy(np.array(ships))
        f.damages = np.copy(np.array(damages))
        moves = f.get_allowed_moves()
        if len(moves) == 0:
            v[-1] = 1
            return np.array(v)

        for pair in moves:
            x = pair[0]
            y = pair[1]
            v[self.n * x + y] = 1

        return np.array(v)

    # TODO change to game_finished => True or False ?
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
