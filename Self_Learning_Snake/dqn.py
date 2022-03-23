import copy
import config
import numpy as np
import tensorflow as tf
from model import MODEL

class DEEP_Q_LEARNING_NETWORK:
    def __init__(self) -> None:
        """ DQN Algorithm
        args:
            model: keras neural network
                model_input: game_state(11)
                model_output: pred_Q_value(3)
            action_dim: int, num_of_actions(3)
            gamma: discount_rate
            lr: learning_rate
        """
        # Define the DQN model and training-target model
        self.dqn_model = MODEL(11, 3)
        self.target_model = copy.deepcopy(self.dqn_model)
        self.target_model.build(11, 3)

        self.dqn_model.model.summary()
        self.target_model.model.summary()

        # Assign inputs as class variables
        self.action_dim = int(4)
        self.gamma = float(config.DISCOUNT_RATE)
        self.lr = float(config.LEARNING_RATE)

    def predict(self, state_value):
        """
        For the given state_value,
        return a list of predicted q-values with number of actions 
        """
        state_value = tf.convert_to_tensor(state_value, dtype=tf.float32)
        state_value = tf.expand_dims(state_value, 0)
        pred_q_list = self.dqn_model.predict(state_value)
        pred_q_list = tf.squeeze(pred_q_list)
        return pred_q_list

    def load_model(self, model_file_path='model/dqn_snake.h5'):
        """
        load the dqn_model from the specified file_path
        """
        self.dqn_model.load_model(model_file_path=model_file_path)

    def save_model(self, model_file_path='model/dqn_snake.h5'):
        """
        save the dqn_model to the specified file_path
        """
        self.dqn_model.save_model(model_file_path=model_file_path)

    def train(self, training_sample, batch_size, update):
        """
        By using the Deep Q_learning Network, update the value network of the model
        Input:
            state_value: bool_list of 11
            [   is_snake_dir_up, is_snake_dir_down, is_snake_dir_left, is_snake_dir_right, 
                is_danger_ahead, is_danger_left, is_danger_right, 
                is_food_left, is_food_right, is_food_up, is_food_down   ]
            onehot_action: bool_list of 3
                (1, 0, 0) -> go_straight
                (0, 1, 0) -> turn_left
                (0, 0, 1) -> turn_right
            reward: INT
                reward given rules: eat -> +10; die -> -10;
            next_state_value: bool_list of 11
            [   is_snake_dir_up, is_snake_dir_down, is_snake_dir_left, is_snake_dir_right, 
                is_danger_ahead, is_danger_left, is_danger_right, 
                is_food_left, is_food_right, is_food_up, is_food_down   ]
            is_done: bool
                dead or not
        """
        assert len(training_sample) == batch_size
        
        # disassemble data from training_sample bundle
        state_value, onehot_action, reward, next_state_value, is_done = zip(*training_sample)

        # Convert input values to float tensor type
        state_value = tf.convert_to_tensor(state_value, dtype=tf.float32)
        onehot_action = tf.convert_to_tensor(onehot_action, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_state_value = tf.convert_to_tensor(next_state_value, dtype=tf.float32)

        # for one sample training, expand dimansions by 1, (1, x=5)
        # if batch_size == 1:
        #     state_value = tf.expand_dims(state_value, 0)
        #     onehot_action = tf.expand_dims(onehot_action, 0)
        #     reward = tf.expand_dims(reward, 0)
        #     next_state_value = tf.expand_dims(next_state_value, 0)
        #     is_done = (is_done, )

        q_pred_list = self.dqn_model.predict(state_value)
        print('q_pred_list', q_pred_list[:2])
        
        # next_q_pred_list = self.target_model.predict(next_state_value)
        # print('next_q_pred_list', next_q_pred_list[:2])
        
        # max_next_q_pred = tf.reduce_max(next_q_pred_list, axis=1) 
        # print('max_next_q_pred', max_next_q_pred[:2])
        
        # q_target = reward + self.gamma * max_next_q_pred * (1 - bool(is_done))
        # print('reward', reward)
        # print('q_target', q_target)

        # 1
        # pred_action_value = tf.reduce_sum(tf.math.multiply(onehot_action, q_pred_list), 1)
        # print('pred_action_value', pred_action_value)

        # 2
        # q_pred_target_list = q_pred_list.copy()
        # for idx in range(batch_size):
        #     Q_new = reward[idx]
        #     if not is_done[idx]:
        #         Q_new = reward[idx] + self.gamma * tf.reduce_max(self.target_model.predict(next_state_value[idx]), axis=1)
        #     q_pred_target_list[idx][tf.argmax(onehot_action).numpy()] = Q_new

        # 3
        # update q_target_list
        # q_pred_target_list = q_pred_list.copy()
        # idx = np.arange(0, batch_size)
        # print('q_pred_target_list', q_pred_target_list)
        # print('onehot_action', tf.argmax(onehot_action).numpy())
        # q_pred_target_list[idx][tf.argmax(onehot_action).numpy()] = q_target

        # Train the model
        self.dqn_model.fit(x_train=state_value, y_train=q_pred_target_list, batch_size=batch_size)
        
        # sync weight from dqn_model to target_model
        if update:
            print('model update')
            self.dqn_model.save_model()
            self.target_model.load_model()
            # self.target_model.set_weights(self.dqn_model.get_weights())

        # Calculate loss, mean_squared_error
        # loss = tf.keras.losses.mean_squared_error(y_true=q_pred_target_list, y_pred=q_pred_list)
