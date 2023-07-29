import tensorflow as tf
import yaml
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten, AveragePooling1D


import warnings

warnings.filterwarnings("ignore")


class LinearNN:
    def __init__(self, input_size, output_size, config_dirctory: str = './config'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        with open(os.path.join(config_dirctory + "/dnn_config.yml"), 'rb') as config_file:
            self.params = yaml.safe_load(config_file)
        self.batch_size = self.params['batch_size']
        self.epochs = self.params['epochs']
        self.hidden_size = self.params['hidden_size']

    def create_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_size, input_shape=(self.input_size,), activation='relu'))
        model.add(Dropout(0.1, input_shape=(self.hidden_size,)))
        model.add(Dense(self.output_size, activation='softmax'))
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model

    def trainer(self, model, samples, labels):
        model.fit(samples, labels, epochs=self.epochs, batch_size=self.batch_size)
        return model


class CNNNetwork:
    def __init__(self, input_size, output_size, config_dirctory: str = './config'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        with open(os.path.join(config_dirctory + "/dnn_config.yml"), 'rb') as config_file:
            self.params = yaml.safe_load(config_file)
        self.batch_size = self.params['batch_size']
        self.epochs = self.params['epochs']

    def create_model(self):
        model = Sequential()
        model.add(Conv1D(filters=128, kernel_size=5, strides=5, activation='relu', input_shape=(self.input_size, 1)))
        model.add(AveragePooling1D(pool_size=5, strides=5))
        model.add(Conv1D(filters=64, kernel_size=5, strides=5, activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.output_size, activation='softmax'))

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model

    def trainer(self, model, samples, labels):
        model.fit(samples, labels, epochs=self.epochs, batch_size=self.batch_size)
        return model
