from ai_agent import AIAgent, state_to_nnet_input
from battleship import *
from nnet_pytorch import CNNet, ResidualNNet

import torch
import torch.nn as nn

def pytorch_train(self, input_, epochs):
    # TODO

    for epoch in range(epochs):
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


if __name__ == '__main__':

    FIELD_SIZE = 5
    ITERATIONS = 10

    network = "cnn"

    if network == "residual":
        nnet = ResidualNNet(FIELD_SIZE)  # init NN
    elif network == "cnn":
        nnet = CNNet(FIELD_SIZE)

    # optimizer =

    # TODO save all data iteration after iteration => convert to array in nnet.train, not here
    for i in range(ITERATIONS):

        g = Game(player_1=AIAgent(nnet), player_2=RandomAgent(), field_size=FIELD_SIZE, ship_sizes=[4, 3, 2],
                 ship_counts=[1, 2, 3])

        result, history_states, history_a_prob, history_success = g.play_out()

        history_a_prob_array = np.array(history_a_prob)

        input_ = np.zeros((len(history_states), FIELD_SIZE, FIELD_SIZE, 3))
        for j in range(len(history_states)):
            input_[j] = state_to_nnet_input(history_states[j])

        print(f"Game {i}: winner {result.winner}")

        nnet.train(input_, history_a_prob_array, history_success)
