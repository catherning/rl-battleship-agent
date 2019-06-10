from battleship import Game, RandomAgent #Random for now
from ai_agent import AIAgent


if __name__ == '__main__':
    g = Game(player_1=AIAgent(), player_2=RandomAgent(), field_size=10, ship_sizes=[4, 3, 2], ship_counts=[1, 2, 3])

    result = g.play_out()
    print(g.field_state_per_player[result.winner])
    print()
    print(g.field_state_per_player[g.get_opponent(result.winner)])