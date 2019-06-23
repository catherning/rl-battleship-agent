import argparse
import os

from torch.utils.data import TensorDataset, DataLoader

from ai_agent import AIAgent
from battleship import *
from network import ResidualNNet

import torch
import torch.nn.functional as functional
import torch.nn as nn

import pandas as pd
from datetime import datetime


CURRENT_DIRECTORY = os.path.dirname(__file__)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=20000, type=int, help="number of episodes to train for.")
    parser.add_argument("--game_field_size", default=6, type=int, help="size of the playing field.")
    parser.add_argument("--ship_sizes", default=[4], type=int, nargs=3, help="ship sizes")
    parser.add_argument("--ship_counts", default=[2], type=int, nargs=3, help="ship count per size")
    
    parser.add_argument("--epochs", default=1, type=int, help="number of epochs to train after each episode.")
    parser.add_argument("--history_size", default=200, type=int,
                        help="number of recent episodes to remember for training.")
    parser.add_argument("--batch_size", default=500, type=int, help="Batch size to use during training.")
    parser.add_argument("--display_freq", default=1, type=int, help="Display frequency")
    parser.add_argument("--save_freq", default=5, type=int, help="Save every x episodes.")
    parser.add_argument("--num_residual_blocks", default=3, type=int, help="Number of residual blocks.")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate for optimizer")

    parser.add_argument("--results_csv_path",
                        default=os.path.join(CURRENT_DIRECTORY, f'results/training_results_{datetime.now()}.csv'),
                        type=str, help="the path where the training results CSV file will be stored.")
    parser.add_argument("--model_save_dir",
                        default=os.path.join(CURRENT_DIRECTORY, f'models/'),
                        type=str, help="the path where the model's learned parameters will be stored.")
    parser.add_argument("--load_model",
                        default=None,
                        type=str, help="the path where the model's learned parameters will be stored.")
    parser.add_argument("--eval", action='store_true', help="Evaluate instead of training")
    parser.add_argument("--eval_num_games", action='store', type=int, default=150, help="Evaluate instead of training")
    return parser


def make_sure_dir_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        print(f'Creating {dir} directory..')
        os.makedirs(dir)


def save_results(results_per_episode, path):
    make_sure_dir_exists(path)
    print(f'Saving results to {path}')
    results_data_frame = pd.DataFrame([results.result_dict for results in results_per_episode])
    results_data_frame.to_csv(path)
    print('Saved')


def save_network_parameters(network: nn.Module, optimizer, model_save_dir):
    make_sure_dir_exists(model_save_dir)
    file_name = f'model_{datetime.now().strftime("%H-%M-%S")}.torch'
    torch.save(
        {"network_state_dict": network.state_dict(),
         "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(model_save_dir, file_name)
    )
    torch.save(
        {"network_state_dict": network.state_dict(),
         "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(model_save_dir, 'model_latest.torch')
    )


def load_network_parameters(path, model, optimizer=None):
    if not os.path.exists(path):
        raise Exception(f"{path} doesn't exist.")
    saved_model = torch.load(path)
    model.load_state_dict(saved_model["network_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(saved_model['optimizer_state_dict'])

# todo:
#     - improve experience history keeping
#     - add actual ship positions to history
#     - define loss as sum of cross entropy between policy and each actual ship position
#     - add MC search for actual probabilities of ship positions based on discovered ships
#     - move evaluation to different file


def build_dataset(state_history, actual_untouched_ship_indices_history):
    """

    :param state_history: a list of state tensors
    :param actual_untouched_ship_indices_history: a list of index lists (for each state there are multiple
    attackable ships)
    :return:
    """

    trainable_states = []
    trainable_labels = []

    for state, actual_untouched_ship_indices in zip(state_history, actual_untouched_ship_indices_history):
        for index in actual_untouched_ship_indices:
            trainable_states.append(state)
            trainable_labels.append(index)

    return TensorDataset(torch.stack(trainable_states), torch.tensor(trainable_labels))


def train(network: ResidualNNet, optimizer,
          state_history, actual_ship_position_labels_history,
          epochs, batch_size):
    dataset = build_dataset(state_history, actual_ship_position_labels_history)
    data_loader = DataLoader(dataset, batch_size, shuffle=True)

    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        for batch_i, (state_batch, ship_position_label_batch) in enumerate(data_loader):
            predicted_policy = network(state_batch)
            log_predicted_policy = torch.log(predicted_policy)

            loss = functional.nll_loss(log_predicted_policy, ship_position_label_batch)
            loss.backward()
            losses.append(float(loss))
            optimizer.step()

    mean_loss = np.mean(np.array(losses))
    return mean_loss


def evaluate(agent: AIAgent, game_configuration, num_games, path):
    agent.eval()
    random_agent = RandomAgent()
    tournament = Tournament(player_1=agent, player_2=random_agent, game_configuration=game_configuration,
                            num_games=num_games)
    results = tournament.play_out(verbose=True)
    agent_win_count = sum(result.winner == agent for result in results)
    random_win_count = sum(result.winner == random_agent for result in results)
    draw_count = len(results) - agent_win_count - random_win_count
    
    save_results(results_per_episode=results, path=path)

    print(f'agent wins = {agent_win_count}; random wins = {random_win_count}; draws = {draw_count}')


def main():
    args = build_arg_parser().parse_args()

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    agent_network = ResidualNNet(args.game_field_size,num_residual_blocks=args.num_residual_blocks)

    game_configuration = GameConfiguration(
        field_size=args.game_field_size,
        ship_sizes=args.ship_sizes, ship_counts=args.ship_counts
    )

    agent = AIAgent(agent_network, args.history_size)
    optimizer = torch.optim.Adam(agent_network.parameters(), lr=args.learning_rate)

    if args.load_model is not None:
        if args.eval:
            optimizer = None
        load_network_parameters(args.load_model, agent_network, optimizer)

    if args.eval:
        evaluate(agent, game_configuration, args.eval_num_games, args.results_csv_path)
        return

    results_per_episode = []

    try:
        for episode_i in range(args.episodes):
            game = Game(player_1=agent, player_2=RandomAgent(), game_configuration=game_configuration)
            episode_results = game.play_out()

            agent.train()
            loss = train(
                network=agent_network,
                optimizer=optimizer,
                state_history=agent.observed_state_history,
                actual_ship_position_labels_history=agent.actual_ship_position_labels_history,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            episode_results.result_dict['loss'] = loss
            results_per_episode.append(episode_results)

            if episode_i % args.display_freq == 0:
                print(f'{datetime.now()} - '
                      f'Episode {episode_i} - '
                      f'{episode_results.num_turns} turns - '
                      f'loss = {loss:.4f} - '
                      f'winner = {episode_results.winner}')

            if episode_i % args.save_freq == 0 and episode_i > 0:
                save_results(results_per_episode=results_per_episode, path=args.results_csv_path)
                save_network_parameters(agent_network, optimizer, args.model_save_dir)
    except KeyboardInterrupt:
        print('Keyboard interrupt: stopping training..')
        save_results(results_per_episode=results_per_episode, path=args.results_csv_path)
        save_network_parameters(agent_network, optimizer, args.model_save_dir)


if __name__ == '__main__':
    main()
