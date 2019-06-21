import argparse
import os

from torch.utils.data import TensorDataset, DataLoader

from ai_agent import AIAgent
from battleship import *
from nnet_pytorch import CNNet, ResidualNNet

import torch
import torch.nn.functional as functional
import torch.nn as nn

import pandas as pd
from datetime import datetime


CURRENT_DIRECTORY = os.path.dirname(__file__)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=20000, type=int, help="number of episodes to train for.")
    parser.add_argument("--game_field_size", default=5, type=int, help="size of the playing field.")
    parser.add_argument("--ship_sizes", default=[4, 3, 2], type=int, nargs=3, help="ship sizes")
    parser.add_argument("--ship_counts", default=[1, 2, 3], type=int, nargs=3, help="ship count per size")
    parser.add_argument("--epochs", default=1, type=int, help="number of epochs to train after each episode.")
    parser.add_argument("--history_size", default=400, type=int,
                        help="number of recent episodes to remember for training.")
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size to use during training.")
    parser.add_argument("--display_freq", default=1, type=int, help="Display frequency")
    parser.add_argument("--save_freq", default=5, type=int, help="Save every x episodes.")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate for optimizer")
    parser.add_argument("--network_type", default='ResNet', type=str,
                        choices=['ResNet', 'CNN'], help="the type of network the agent uses")
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
    parser.add_argument("--gpu", action='store_true', help="Use GPU")
    return parser


def make_sure_dir_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        print(f'Creating {dir} directory..')
        os.makedirs(dir)


def save_training_results(results_per_episode, path):
    make_sure_dir_exists(path)
    print(f'Saving training results to {path}')
    results_data_frame = pd.DataFrame([results.result_dict for results in results_per_episode])
    results_data_frame.to_csv(path)
    print('Saved')


def save_network_parameters(network: nn.Module, optimizer, model_save_dir):
    make_sure_dir_exists(model_save_dir)
    file_name = f'model_{datetime.now()}.torch'
    torch.save(
        {"network_state_dict": network.state_dict(),
         "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(model_save_dir, file_name)
    )


def load_network_parameters(path, model, optimizer=None):
    if not os.path.exists(path):
        raise Exception(f"{path} doesn't exist.")
    saved_model = torch.load(path)
    model.load_state_dict(saved_model["network_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(saved_model['optimizer_state_dict'])


def train(network: ResidualNNet, optimizer, states, actions, rewards, epochs, batch_size, use_gpu):
    dataset = TensorDataset(states, actions, rewards)
    data_loader = DataLoader(dataset, batch_size, shuffle=True)

    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        for batch_i, (state_batch, action_batch, reward_batch) in enumerate(data_loader):
            if use_gpu:
                state_batch = state_batch.cuda()
                action_batch = action_batch.cuda()
                reward_batch = reward_batch.cuda()

            predicted_policy, predicted_values = network(state_batch)
            log_predicted_policy = torch.log(predicted_policy)

            loss = functional.mse_loss(predicted_values.squeeze(dim=1), reward_batch)
            loss += functional.nll_loss(log_predicted_policy, action_batch.squeeze(dim=1).squeeze(1))
            loss.backward()
            losses.append(float(loss))
            optimizer.step()

    mean_loss = np.mean(np.array(losses))
    return mean_loss


def evaluate(agent: AIAgent, game_configuration):
    agent.eval()
    random_agent = RandomAgent()
    tournament = Tournament(player_1=agent, player_2=random_agent, game_configuration=game_configuration, num_games=50)
    results = tournament.play_out(verbose=True)
    agent_win_count = sum(result.winner == agent for result in results)
    random_win_count = sum(result.winner == random_agent for result in results)
    draw_count = len(results) - agent_win_count - random_win_count

    print(f'agent wins = {agent_win_count}; random wins = {random_win_count}; draws = {draw_count}')


def main():
    args = build_arg_parser().parse_args()

    if args.network_type == 'ResNet':
        agent_network = ResidualNNet(args.game_field_size)
    else:  # args.network_type == "cnn":
        agent_network = CNNet(args.game_field_size)

    game_configuration = GameConfiguration(
        field_size=args.game_field_size,
        ship_sizes=args.ship_sizes, ship_counts=args.ship_counts
    )

    agent = AIAgent(agent_network)
    optimizer = torch.optim.Adam(agent_network.parameters())

    if args.load_model is not None:
        load_network_parameters(args.load_model, agent_network, optimizer)

    if args.eval:
        evaluate(agent, game_configuration)
        return

    results_per_episode = []

    try:
        for episode_i in range(args.episodes):
            game = Game(player_1=agent, player_2=agent, game_configuration=game_configuration)
            agent_network.eval()
            episode_results = game.play_out()

            if len(agent.state_history) > args.history_size:
                agent.state_history.pop(0)
                agent.reward_history.pop(0)
                agent.action_history.pop(0)

            agent_network.train()
            loss = train(
                network=agent_network,
                optimizer=optimizer,
                states=torch.stack(agent.state_history),
                rewards=torch.tensor(agent.reward_history, dtype=torch.float),
                actions=torch.stack(agent.action_history),
                epochs=args.epochs,
                batch_size=args.batch_size,
                use_gpu=args.gpu,
            )
            episode_results.result_dict['loss'] = loss
            results_per_episode.append(episode_results)

            if episode_i % args.display_freq == 0:
                print(f'{datetime.now()} - Episode {episode_i} - {episode_results.num_turns} turns - loss = {loss:.4f}')

            if episode_i % args.save_freq == 0 and episode_i > 0:
                save_training_results(results_per_episode=results_per_episode, path=args.results_csv_path)
                save_network_parameters(agent_network, optimizer, args.model_save_dir)
    except KeyboardInterrupt:
        print('Keyboard interrupt: stopping training..')
        save_training_results(results_per_episode=results_per_episode, path=args.results_csv_path)
        save_network_parameters(agent_network, optimizer, args.model_save_dir)


if __name__ == '__main__':
    main()
