import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

sys.path.append('..')


def calculate_padding_to_maintain_size(kernel_size):
    """
    Calculate padding assuming that stride == 1
    """
    return int((kernel_size - 1) / 2)


def create_cnn_layer(in_filters, out_filters, filter_size, padding):
    """
    Create a Sequential module that contains a Conv2d, BatchNorm2d, and ReLU layer

    :param in_filters:
    :param out_filters:
    :param filter_size:
    :param padding:
    :return:
    """
    # padding = calculate_padding_to_maintain_size(in_size)
    return nn.Sequential(nn.Conv2d(in_filters, out_filters, filter_size, padding=padding),
                         nn.BatchNorm2d(out_filters),
                         nn.ReLU())


def create_dense_layer(in_dim, out_dim, dropout):
    """
    Create a Sequential module that contains a Linear, BatchNorma1d, ReLU, and Dropout layer

    :param in_dim:
    :param out_dim:
    :param dropout:
    :return:
    """
    return nn.Sequential(nn.Linear(in_dim, out_dim),
                         nn.BatchNorm1d(out_dim),
                         nn.ReLU(),
                         nn.Dropout(p=dropout))


class ResidualBlock(nn.Module):
    def __init__(self, num_filters, filter_size):
        super().__init__()

        self.num_filters = num_filters
        self.filter_size = filter_size

        padding = calculate_padding_to_maintain_size(filter_size)

        self.res_block_l1 = create_cnn_layer(num_filters, num_filters, filter_size, padding)
        self.res_block_l2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, filter_size, padding=padding),
                                          nn.BatchNorm2d(num_filters))

    def forward(self, x):
        """
        Each residual block applies the following modules sequentially to its input:
        1. A convolution of 256 filters of kernel size 3 × 3 with stride 1
        2. Batch normalisation
        3. A rectifier non-linearity
        4. A convolution of 256 filters of kernel size 3 × 3 with stride 1
        5. Batch normalisation
        6. A skip connection that adds the input to the block
        7. A rectifier non-linearity

        :param x:
        :return:
        """

        output_layer1 = self.res_block_l1(x)
        output_layer2 = self.res_block_l2(output_layer1)
        output_layer2 += x
        activated_output = functional.relu(output_layer2)

        return activated_output


class ResidualNNet(nn.Module):
    def __init__(self, game_field_size, num_filters=256, filter_size=3, num_residual_blocks=3, value_out_dims=128):
        super().__init__()

        self.game_field_size = game_field_size
        self.num_filters = num_filters
        self.filter_size = filter_size

        self.game_field_size = game_field_size

        # Model
        self.conv_block = create_cnn_layer(
            in_filters=3, out_filters=num_filters,
            filter_size=filter_size, padding=calculate_padding_to_maintain_size(kernel_size=filter_size))

        self.residual_blocks = [
            ResidualBlock(self.num_filters, self.filter_size)
            for _ in range(num_residual_blocks)
        ]

        # XXX Not used anymore, but to keep if we want to load older models
        self.value_conv_1 = create_cnn_layer(
            in_filters=num_filters, out_filters=1,
            filter_size=1, padding=calculate_padding_to_maintain_size(kernel_size=1)
        )
        self.value_out_layer = nn.Sequential(nn.Linear(self.game_field_size ** 2, value_out_dims),
                                             nn.ReLU(),
                                             nn.Linear(value_out_dims, 1),
                                             nn.Tanh())

        self.policy_conv_1 = create_cnn_layer(
            in_filters=num_filters, out_filters=1,
            filter_size=1, padding=calculate_padding_to_maintain_size(kernel_size=1)
        )

        num_cells_in_game_field = game_field_size * game_field_size
        action_dims = num_cells_in_game_field
        self.policy_output_layer = nn.Sequential(
            nn.Linear(in_features=num_cells_in_game_field, out_features=action_dims),
            nn.Softmax(),
        )


    def convolutional_block(self, input_):
        """
        1. A convolution of 256 filters of kernel size 3 × 3 with stride 1
        2. Batch normalisation 18
        3. A rectifier non-linearity
        """

        batch_size = input_.shape[0]
        x = input_  # .reshape((batch_size, 1, self.game_field_size, self.game_field_size))
        output = self.conv_block(x)

        return output

    def policy_head(self, x):
        """
        1. A convolution of 2 filters of kernel size 1 × 1 with stride 1
        2. Batch normalisation
        3. A rectifier non-linearity
        4. A fully connected linear layer that outputs a vector of size 19**2 + 1 = 362 corresponding to
           logit probabilities for all intersections and the pass move

        :param x:
        :return:
        """
        activ_1 = self.policy_conv_1(x)

        flatt_2 = activ_1.view(activ_1.shape[0], -1)
        a_prob = self.policy_output_layer(flatt_2)

        return a_prob

    def forward(self, input_):
        """
        Network described in DeepMind paper (pp 27-29)

        https://deepmind.com/documents/119/agz_unformatted_nature.pdf

        dual-res:
        The network contains a 20-block residual tower, as described above, followed by
        a policy head. This is the architecture used in AlphaGo Zero

        """

        residuals = self.convolutional_block(input_)

        for residual_block in self.residual_blocks:
            residuals = residual_block(residuals)

        a_prob = self.policy_head(residuals)

        return a_prob
