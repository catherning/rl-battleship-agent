import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

sys.path.append('..')


def pad_2d(input_, padding, k_h, k_w, s_h=1, s_w=1):
    """
    Adds padding like in Tensorflow/Keras for Conv2D
    https://gist.github.com/Ujjwal-9/e13f163d8d59baa40e741fc387da67df

    :param input_:
    :param padding: either valid or same
    :param k_h: kernel_height
    :param k_w: kernel_width
    :param s_h: stride_height
    :param s_w: stride_width
    :return:
    """
    if padding == 'valid':
        return input_
    elif padding == 'same':
        in_height, in_width = input_.size(2), input_.size(3)
        out_height = int(np.ceil(float(in_height) / float(s_h)))
        out_width = int(np.ceil(float(in_width) / float(s_w)))

        pad_along_height = max((out_height - 1) * s_h + k_h - in_height, 0)
        pad_along_width = max((out_width - 1) * s_w + k_w - in_width, 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        output = functional.pad(input_, (pad_left, pad_right, pad_top, pad_bottom))
        return output


def create_cnn_layer_seq(in_filters, out_filters, filter_size):
    return nn.Sequential(nn.Conv2d(in_filters, out_filters, filter_size),
                         nn.BatchNorm2d(out_filters),
                         nn.ReLU())


def create_dense_layer(in_dim, out_dim, dropout):
    return nn.Sequential(nn.Linear(in_dim, out_dim),
                         nn.BatchNorm1d(out_dim),
                         nn.ReLU(),
                         nn.Dropout(p=dropout))


class NNet(nn.Module):

    def __init__(self, game):
        super(NNet, self).__init__()

        self.w, self.h = game.get_field_size()
        self.action_size = game.get_num_actions()  # all

        # Model hyper-parameters
        self.num_filters = 256
        self.filter_size = 3

    def save_checkpoint(self, optimizer, folder='checkpoint', filename='checkpoint.pth'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({"model_state_dict": self.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        saved_model = torch.load(filepath)
        self.load_state_dict(saved_model["model_state_dict"])

        return saved_model["optimizer_state_dict"]


class CNNet(NNet):

    def __init__(self, game):
        super().__init__(game)

        self.dropout = 0.3
        self.cnn_out_dim = 1024
        self.dense_in_dim = 512 #1024
        self.dense_out_dim = 256 #512

        # Model
        self.conv_1 = create_cnn_layer_seq(1, self.num_filters, self.filter_size)
        self.conv_2 = create_cnn_layer_seq(self.num_filters, self.num_filters, self.filter_size)
        self.conv_3 = create_cnn_layer_seq(self.num_filters, self.num_filters, self.filter_size)
        self.conv_4 = create_cnn_layer_seq(self.num_filters, self.num_filters, self.filter_size)

        # 1024 is original in_dim
        self.dense_l1 = create_dense_layer(self.cnn_out_dim, self.dense_in_dim, self.dropout)
        self.dense_l2 = create_dense_layer(self.dense_in_dim, self.dense_out_dim, self.dropout)
        self.probabilities = nn.Sequential(nn.Linear(self.dense_out_dim, self.action_size), nn.Softmax())
        self.values = nn.Sequential(nn.Linear(self.dense_out_dim, 1), nn.Tanh())

        print(self)

    def forward(self, input_):
        """
        Regular CNN network

        :return: [a_prob, values]
        """

        batch_size = input_.shape[0]
        x = input_.reshape((batch_size, 1, self.w, self.h))
        if batch_size == 1:
            x = torch.cat([x, x])

        # Same padding
        x = pad_2d(x, "same", self.filter_size, self.filter_size)
        a_conv = self.conv_1(x)
        a_conv = pad_2d(a_conv, "same", self.filter_size, self.filter_size)
        a_conv = self.conv_2(a_conv)

        # Valid padding
        a_conv = self.conv_3(a_conv)
        a_conv = self.conv_4(a_conv)

        a_conv = a_conv.view(a_conv.shape[0], -1)

        dense = self.dense_l1(a_conv)
        dense = self.dense_l2(dense)

        a_prob = self.probabilities(dense)
        values = self.values(dense)

        return a_prob, values


class ResidualBlock(nn.Module):
    def __init__(self, num_filters, filter_size):
        super().__init__()

        self.num_filters = num_filters
        self.filter_size = filter_size

        self.res_block_l1 = create_cnn_layer_seq(self.num_filters, self.num_filters, self.filter_size)
        self.res_block_l2 = nn.Sequential(nn.Conv2d(self.num_filters, self.num_filters, self.filter_size),
                                          nn.BatchNorm2d(self.num_filters))

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

        padded_x = pad_2d(x, "same", self.filter_size, self.filter_size)
        output_layer1 = self.res_block_l1(padded_x)

        padded_l1 = pad_2d(output_layer1, "same", self.filter_size, self.filter_size)
        output_layer2 = self.res_block_l2(padded_l1)

        output_layer2 += x
        activ_2 = nn.ReLU()(output_layer2)

        return activ_2


class ResidualNNet(NNet):

    def __init__(self, game):
        super().__init__(game)

        self.num_residual = 3  # 19 #39 # only residual
        self.value_head_dense = 128 # 256  # only residual
        self.game_size = game.get_field_size()[0]

        # Model
        self.conv_block = create_cnn_layer_seq(1, self.num_filters, self.filter_size)

        self.list_res_blocks = [ResidualBlock(self.num_filters, self.filter_size) for _ in range(self.num_residual)]

        self.value_conv_1 = create_cnn_layer_seq(self.num_filters, 1, 1)
        self.value_out_layer = nn.Sequential(nn.Linear(self.game_size ** 2, self.value_head_dense),
                                             nn.ReLU(),
                                             nn.Linear(self.value_head_dense, 1),
                                             nn.Tanh())

        self.policy_conv_1 = create_cnn_layer_seq(self.num_filters, 1, 1)

        self.policy_out_layer = nn.Sequential(nn.Linear(self.game_size ** 2, self.action_size),
                                              nn.Softmax())

        print(self)

    def convolutional_block(self, input_):
        """
        1. A convolution of 256 filters of kernel size 3 × 3 with stride 1
        2. Batch normalisation 18
        3. A rectifier non-linearity
        """

        batch_size = input_.shape[0]
        x = input_.reshape((batch_size, 1, self.w, self.h))
        padded_x = pad_2d(x, "same", self.filter_size, self.filter_size)
        output = self.conv_block(padded_x)

        return output

    def value_head(self, x):
        """
        1. A convolution of 1 filter of kernel size 1 × 1 with stride 1
        2. Batch normalisation
        3. A rectifier non-linearity
        4. A fully connected linear layer to a hidden layer of size 256
        5. A rectifier non-linearity
        6. A fully connected linear layer to a scalar
        7. A tanh non-linearity outputting a scalar in the range [−1, 1]

        :param x:
        :return:
        """
        padded_x = pad_2d(x, "same", 1, 1)
        activ_1 = self.value_conv_1(padded_x)
        flatt_2 = activ_1.view(activ_1.shape[0], -1)

        values = self.value_out_layer(flatt_2)

        return values

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
        padded_x = pad_2d(x, "same", 1, 1)
        activ_1 = self.policy_conv_1(padded_x)

        flatt_2 = activ_1.view(activ_1.shape[0], -1)
        a_prob = self.policy_out_layer(flatt_2)

        return a_prob

    def forward(self, input_):
        """
        Network described in DeepMind paper (pp 27-29)

        https://deepmind.com/documents/119/agz_unformatted_nature.pdf

        dual-res:
        The network contains a 20-block residual tower, as described above, followed by
        both a policy head and a value head. This is the architecture used in AlphaGo Zero

        """

        residuals = self.convolutional_block(input_)

        for i in range(0, self.num_residual):
            residuals = self.list_res_blocks[i](residuals)

        values = self.value_head(residuals)
        a_prob = self.policy_head(residuals)

        return a_prob, values