import os
import sys

import numpy as np
from keras.layers import Conv2D, BatchNormalization, Activation, Reshape, Dropout, Flatten, Dense, Add
from keras.models import *
from keras.optimizers import Adam

sys.path.append('..')


class NNet:
    def __init__(self, game):
        self.w, self.h = game.get_field_size()
        self.action_size = game.get_num_actions()  # all

        # Model hyper-parameters
        self.num_channels = 256
        self.filter_size = 3

        # Training
        self.epochs = 10
        self.batch_size = 64
        self.adam_lr = 1e-3

        self.input_layer = Input(shape=(self.w, self.h))
        self.model = None

    def cnn_layer(self, x, num_channels=None, filter_size=None, activation='linear', padding='same'):

        if num_channels is None:
            num_channels = self.num_channels
        if filter_size is None:
            filter_size = self.filter_size

        conv2d_1 = Conv2D(num_channels, filter_size,
                          activation=activation, padding=padding)(x)
        bnorm_1 = BatchNormalization(axis=3)(conv2d_1)
        a_conv1 = Activation('relu')(bnorm_1)
        return a_conv1

    def train(self, examples):
        """
        Train the network with provided data
        """
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

        self.model = Model(inputs=self.input_layer, outputs=[
            self.a_prob, self.values])
        self.model.compile(loss=['categorical_crossentropy',
                                 'mean_squared_error'], optimizer=Adam(self.adam_lr))

    def cnn_net(self):
        """
        Regular CNN network

        :return: [a_prob, values]
        """


        x = Reshape((self.w, self.h, 1))(self.input_layer)

        a_conv1 = self.cnn_layer(x)
        a_conv2 = self.cnn_layer(a_conv1)
        a_conv3 = self.cnn_layer(a_conv2, padding='valid')
        a_conv4 = self.cnn_layer(a_conv3, padding='valid')

        a_conv4_flat = Flatten()(a_conv4)

        temp1 = Dense(1024)(a_conv4_flat)
        temp2 = BatchNormalization(axis=1)(temp1)
        dense_1 = Dropout(self.dropout)(Activation('relu')( temp2 ))
        dense_2 = Dropout(self.dropout)(Activation('relu')(
            BatchNormalization(axis=1)(Dense(512)(dense_1))))

        a_prob = Dense(self.action_size, activation='softmax',
                       name='a_prob')(dense_2)
        values = Dense(1, activation='tanh', name='values')(dense_2)

        return [a_prob, values]


class ResidualNNet(NNet):

    def __init__(self, game):
        super().__init__(game)

        self.num_residual = 3  # 19 #39 # only residual
        self.value_head_dense = 256  # only residual

        self.a_prob, self.values = self.residual_net()

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

        x = Reshape((self.w, self.h, 1))(self.input_layer)
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

        convl_4 = Conv2D(self.num_channels, self.filter_size,
                         activation='linear', padding='same')(output_layer1)
        bnorm_5 = BatchNormalization(axis=3)(convl_4)
        merge_6 = Add()([bnorm_5, x])
        activ_7 = Activation('relu')(merge_6)

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
        """

        activ_3 = self.cnn_layer(x, num_channels=1, filter_size=1)

        flatt_4 = Flatten()(activ_3)
        dense_5 = Dense(self.value_head_dense, activation='relu')(flatt_4)
        dense_6 = Dense(1, activation='tanh', name='value_head')(dense_5)

        return dense_6

    def policy_head(self, x):
        """
        1. A convolution of 2 filters of kernel size 1 × 1 with stride 1
        2. Batch normalisation
        3. A rectifier non-linearity
        4. A fully connected linear layer that outputs a vector of size 19**2 + 1 = 362 corresponding to
           logit probabilities for all intersections and the pass move
        """

        activ_3 = self.cnn_layer(x, num_channels=1, filter_size=1)

        flatt_4 = Flatten()(activ_3)
        dense_4 = Dense(self.action_size, activation='softmax',
                        name='policy_head')(flatt_4)

        return dense_4

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
