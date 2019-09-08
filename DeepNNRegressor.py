import os
import random

import numpy as np
import sklearn
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense
from keras.models import Sequential

# GPU Selection
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Prevent tensorflow from allocating the entire GPU memory at once
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)


CHECKPOINT_DIR = 'dnn_checkpoints'


class TrainedModel(object):
    def __init__(self, checkpoint_path, scaler_x, scaler_y, trained_model):
        self.checkpoint_path = checkpoint_path
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.trained_model = trained_model

    def predict(self, X_test):
        return self.scaler_y.inverse_transform(self.trained_model.predict(self.scaler_x.transform(X_test)).reshape(-1, 1))


class DeepNNRegressor(object):
    def __init__(self, num_hidden_layers, num_sigmoids_in_each_hidden_layer, loss, optimizer, mini_batch_size, num_epochs, validation_split):
        self.num_sigmoids_in_each_hidden_layer = num_sigmoids_in_each_hidden_layer
        self.num_hidden_layers = num_hidden_layers
        self.loss = loss
        self.optimizer = optimizer
        self.mini_batch_size = mini_batch_size
        self.num_epochs = num_epochs
        self.validation_split = validation_split

    def __build_model(self, X):
        model = Sequential()
        # Input and hidden layers
        for i in range(self.num_hidden_layers):
            model.add(Dense(units=self.num_sigmoids_in_each_hidden_layer, activation='sigmoid',
                            **({'input_dim': X.shape[1]} if i == 0 else {})))
        # Output layer
        model.add(Dense(units=1, activation='linear'))
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def fit(self, X, y):
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        unique_id = np.base_repr(random.randint(2 ** 63, 2 ** 64), 36)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model_%s.hdf5' % unique_id)
        scaler_x = sklearn.preprocessing.StandardScaler().fit(X)
        scaler_y = sklearn.preprocessing.StandardScaler().fit(y.reshape(-1, 1))
        model = self.__build_model(X)
        model.fit(
            scaler_x.transform(X),
            scaler_y.transform(y.reshape(-1, 1)),
            batch_size=self.mini_batch_size,
            epochs=self.num_epochs,
            validation_split=self.validation_split,
            verbose=2,
            callbacks=[ModelCheckpoint(
                checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')]
        )
        model.load_weights(checkpoint_path)
        model.summary()
        return TrainedModel(checkpoint_path, scaler_x, scaler_y, model)
