import os
from unicodedata import name
import config
import random
import collections
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, Dense, Input
from tensorflow.keras.optimizers import Adam

class MODEL:
    def __init__(self, input_shape, output_shape, model_file_path=None) -> None:
        if model_file_path == None:
            # Create new model to train the agent
            self.build(input_shape, output_shape)
        else:
            # Load the old_model and start from where left
            self.load_model(model_file_path)

    def build(self, input_shape, output_shape):
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,), name='Input'),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(output_shape, activation='softmax', name='output')
        ])
        self.model.compile(
                optimizer = Adam(learning_rate=config.LEARNING_RATE),
                loss='mean_squared_error'
        )

    def fit(self, x_train, y_train, batch_size):
        self.model.fit(x_train, y_train, batch_size, verbose=0)

    def predict(self, pred_x):
        # print('pred_x', pred_x)
        return self.model.predict(pred_x)
    
    def save_model(self, model_file_path='model/dqn_snake.h5'):
        self.model.save(model_file_path)

    def load_model(self, model_file_path='model/dqn_snake.h5'):
        if not os.path.exists(model_file_path):
            print(model_file_path, 'not exist')
        self.model = load_model(model_file_path)

class ReplayMemory:
    def __init__(self) -> None:
        self.memory = collections.deque(maxlen=config.MAX_MEMORY)
    
    def append(self, sample):
        self.memory.append(sample)

    def random_sample(self, batch_size):
        if len(self.memory) < batch_size:
            sample_batch = self.memory
        else:
            sample_batch = random.sample(self.memory, batch_size)
        batch_size = len(sample_batch)
        return sample_batch, batch_size

