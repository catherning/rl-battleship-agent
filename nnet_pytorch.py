import os
import sys

import numpy as np
import torch
import torch.nn as nn
from keras.models import *
from keras.optimizers import Adam

sys.path.append('..')


def pad_2d(input_, padding, k_h, k_w, s_h=1, s_w=1):
    """
    Adds padding like in Tensorflow/Keras for Conv2D
    https://gist.github.com/Ujjwal-9/e13f163d8d59baa40e741fc387da67df

    :param input_:
    :param padding:
    :param k_h:
    :param k_w:
    :param s_h:
    :param s_w:
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
        output = F.pad(input_, (pad_left, pad_right, pad_top, pad_bottom))
        return output


class NNet:

    def __init__(self, game):
        self.w, self.h = game.get_field_size()
        self.action_size = game.get_num_actions()  # all

        # Model hyper-parameters
        self.num_filters = 256
        self.filter_size = 3

        # Training
        self.epochs = 10
        self.batch_size = 64
        self.adam_lr = 1e-3

        self.input_layer = Input(shape=(self.w, self.h))
        self.model = None

    def cnn_layer(self, x, out_filters=None, filter_size=None, activation='linear', padding='same'):
        """

        :param x: input of size (batch_size,in_channels=out_channels,self.h,self.w)
        :param out_filters:
        :param filter_size:
        :param activation:
        :param padding:
        :return: a_conv1 of size (batch_size, out_channels, self.h, self.w)
        """

        if out_filters is None:
            out_filters = self.num_filters
            in_channels = out_filters
        else:
            in_channels = self.num_filters

        if filter_size is None:
            filter_size = self.filter_size

        padded_x = pad_2d(x, padding, filter_size, filter_size)
        conv2d_1 = nn.Conv2d(in_channels, out_filters, filter_size)(padded_x)
        # activation is always linear for now, so do nothing after conv2D

        bnorm_1 = nn.BatchNorm2d(out_filters)(conv2d_1)
        a_conv1 = nn.ReLU()(bnorm_1)

        return a_conv1

    def train(self, examples):
        """
        Train the network with provided data
        """
        # todo: convert to pytorch
        input_fields, target_a_prob, target_values = list(zip(*examples))
        input_fields = np.asarray(input_fields)
        target_a_prob = np.asarray(target_a_prob)
        target_values = np.asarray(target_values)
        self.model.fit(x=input_fields, y=[
            target_a_prob, target_values], batch_size=self.batch_size, epochs=self.epochs)

    def predict(self, field):
        """
        Given cell, calculate where to step next from the given cell,
        output most probable action and value for the next state
        """
        # todo: convert to pytorch
        field = field[np.newaxis, :, :]
        a_prob, value = self.model.predict(field)
        return a_prob[0], value[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        self.model.load_weights(filepath)


class CNNet(NNet):

    def __init__(self, game):
        super().__init__(game)

        self.dropout = 0.3  # only CNN

        self.a_prob, self.values = self.cnn_net()

        # todo: change to pytorch
        self.model = Model(inputs=self.input_layer, outputs=[
            self.a_prob, self.values])
        self.model.compile(loss=['categorical_crossentropy',
                                 'mean_squared_error'], optimizer=Adam(self.adam_lr))

    def cnn_net(self):
        """
        Regular CNN network

        :return: [a_prob, values]
        """

        batch_size = self.input_layer[0]
        x = self.input_layer.reshape((batch_size, 1, self.w, self.h))

        a_conv1 = self.cnn_layer(x)
        a_conv2 = self.cnn_layer(a_conv1)
        a_conv3 = self.cnn_layer(a_conv2, padding='valid')
        a_conv4 = self.cnn_layer(a_conv3, padding='valid')

        a_conv4_flat = a_conv4.view(a_conv4.shape[0], -1)

        fc_1 = nn.Linear(a_conv4_flat.shape[1], 1024)
        batch_norm_1 = nn.BatchNorm1d(fc_1.shape[1])(fc_1)
        act_1 = nn.ReLU()(batch_norm_1)
        dense_1 = nn.Dropout(p=self.dropout)(act_1)

        fc_2 = nn.Linear(dense_1.shape[1], 512)
        batch_norm_2 = nn.BatchNorm1d(fc_2.shape[1])(fc_2)
        act_2 = nn.ReLU()(batch_norm_2)
        dense_2 = nn.Dropout(p=self.dropout)(act_2)

        fc_prob = nn.Linear(dense_2.shape[1], self.action_size)(dense_2)
        a_prob = nn.Softmax()(fc_prob)

        fc_values = nn.Linear(dense_2.shape[1], 1)(dense_2)
        values = nn.Tanh()(fc_values)

        return [a_prob, values]


class ResidualNNet(NNet):

    def __init__(self, game):
        super().__init__(game)

        self.num_residual = 3  # 19 #39 # only residual
        self.value_head_dense = 256  # only residual

        self.a_prob, self.values = self.residual_net()

        # todo: change to pytorch
        self.model = Model(inputs=self.input_layer, outputs=[
            self.a_prob, self.values])
        self.model.compile(loss=['categorical_crossentropy',
                                 'mean_squared_error'], optimizer=Adam(self.adam_lr))

    def convolutional_block(self):
        """
        1. A convolution of 256 filters of kernel size 3 × 3 with stride 1
        2. Batch normalisation 18
        3. A rectifier non-linearity
        """

        batch_size = self.input_layer[0]
        x = self.input_layer.reshape((batch_size, 1, self.w, self.h))
        output = self.cnn_layer(x)

        return output

    def residual_block(self, x):
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
        output_layer1 = self.cnn_layer(x)

        padded_x = pad_2d(x, "same", self.filter_size, self.filter_size)
        convl_4 = nn.Conv2d(output_layer1.shape[1], self.num_filters, self.filter_size)(padded_x)
        bnorm_5 = nn.BatchNorm2d(self.num_filters)(convl_4)
        merge_6 = torch.cat((bnorm_5, x))
        activ_7 = nn.ReLU()(merge_6)

        return activ_7

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
        activ_3 = self.cnn_layer(x, out_filters=1, filter_size=1)
        flatt_4 = activ_3.view(activ_3.shape[0], -1)

        fc_5 = nn.Linear(flatt_4.shape[1], self.value_head_dense)
        dense_5 = nn.ReLU()(fc_5)
        fc_6 = nn.Linear(dense_5.shape[1], 1)
        values = nn.Tanh()(fc_6)

        return values

    def policy_head(self, x):
        """
        1. A convolution of 2 filters of kernel size 1 × 1 with stride 1
        2. Batch normalisation
        3. A rectifier non-linearity
        4. A fully connected linear layer that outputs a vector of size 192 + 1 = 362 corresponding to
           logit probabilities for all intersections and the pass move

        :param x:
        :return:
        """
        activ_3 = self.cnn_layer(x, out_filters=1, filter_size=1)

        flatt_4 = activ_3.view(activ_3.shape[0], -1)

        fc_4 = nn.Linear(flatt_4.shape[1], self.action_size)
        a_prob = nn.Softmax()(fc_4)

        return a_prob

    def residual_net(self):
        """
        Network described in DeepMind paper (pp 27-29)

        https://deepmind.com/documents/119/agz_unformatted_nature.pdf

        dual-res:
        The network contains a 20-block residual tower, as described above, followed by
        both a policy head and a value head. This is the architecture used in AlphaGo Zero

        """

        conv_block = self.convolutional_block()

        residuals = self.residual_block(conv_block)

        for i in range(0, self.num_residual):
            residuals = self.residual_block(residuals)

        values = self.value_head(residuals)
        a_prob = self.policy_head(residuals)

        return [a_prob, values]
