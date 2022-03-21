import os
import config
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam

class NEURAL_NETWORK:
    def __init__(self, input_shape, output_shape, model_file_path=None) -> None:
        if model_file_path == None:
            # Create new model to train the agent
            self.build(input_shape, output_shape)
        else:
            # Load the old_model and start from where left
            self.load_model(model_file_path)

    def build(self, input_shape, output_shape):
        self.model = Sequential([
            InputLayer(input_shape=input_shape),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(output_shape, activation=None)
        ])
        self.model.compile(
                optimizer = Adam(learning_rate=config.LEARNING_RATE),
                loss='mean_squared_error'
              )

    def fit(self, x_train, y_train, batch_size):
        # Prepare a tensorflow dataset
        # dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        self.model.fit(x_train, y_train, batch_size)

    def predict(self, pred_x):
        # print(pred_x)
        return self.model.predict(pred_x)
    
    def save_model(self, model_file_path='model/dqn_snake.h5'):
        self.model.save(model_file_path)

    def load_model(self, model_file_path='model/dqn_snake.h5'):
        if not os.path.exists(model_file_path):
            print(model_file_path, 'not exist')
        self.model = load_model(model_file_path)