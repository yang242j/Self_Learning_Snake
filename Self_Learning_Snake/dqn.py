import config
import numpy as np
import tensorflow as tf
from model import NEURAL_NETWORK

class DEEP_Q_LEARNING_NETWORK:
    def __init__(self) -> None:
        """ DQN Algorithm
        args:
            model: keras neural network
                model_input: game_state
                model_output: pred_Q_value
            action_dim: int, num_of_actions (4)
            gamma: discount_rate
            lr: learning_rate
        """
        # Define the DQN model and training-target model
        self.dqn_model = NEURAL_NETWORK(11, 4)
        # self.target_model = copy.deepcopy(self.dqn_model)

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
        deep copy the dqn_model as the target_model
        """
        self.dqn_model.load_model(model_file_path=model_file_path)
        # self.target_model = copy.deepcopy(self.dqn_model)

    def save_model(self, model_file_path='model/dqn_snake.h5'):
        """
        save the dqn_model to the specified file_path
        """
        self.dqn_model.save_model(model_file_path=model_file_path)

    def __to_indices(self, tensor_actions):
        """
        Convert the tensor_action to action index list
        """
        action_idx_list = []
        for action in tensor_actions:
            if tf.reduce_all(tf.equal(action, config.DIR_LEFT)):
                action_idx_list.append(0)
            elif tf.reduce_all(tf.equal(action, config.DIR_RIGHT)):
                action_idx_list.append(1)
            elif tf.reduce_all(tf.equal(action, config.DIR_UP)):
                action_idx_list.append(2)
            elif tf.reduce_all(tf.equal(action, config.DIR_DOWN)):
                action_idx_list.append(3)
            else:
                print(action, 'not converted')
        return action_idx_list

    def train(self, training_sample, batch_size):
        """
        By using the Deep Q_learning Network, update the value network of the model
        Input:
            state_value: [
                snake_dir_x, snake_dir_y, 
                is_danger_ahead, is_danger_left, is_danger_right, 
                is_food_left, is_food_right, is_food_up, is_food_down ]
            action: 
                left  -> Vector2 (-1,  0) -> idx: 0;
                right -> Vector2 ( 1,  0) -> idx: 1;
                up    -> Vector2 ( 0, -1) -> idx: 2;
                down  -> Vector2 ( 0,  1) -> idx: 3
            reward:
                integer number,
                eat -> +10,
                die -> -10,
                opposite_dir (ignored) -> -5
            next_state_value:
                agent.get_state(action)
            is_done:
                bool, good_over or not
        """
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        
        # disassemble data from training_sample bundle
        # state_value, action, reward, next_state_value, is_done = tf.map_fn(tf.tensor(), training_sample, dtype=tf.float32)
        state_value, action, reward, next_state_value, is_done = training_sample

        # Change dtype of each
        state_value = tf.convert_to_tensor(state_value, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_state_value = tf.convert_to_tensor(next_state_value, dtype=tf.float32)

        # for one sample training, (1, x=5)
        if batch_size == 1:
            state_value = tf.expand_dims(state_value, 0)
            action = tf.expand_dims(action, 0)
            reward = tf.expand_dims(reward, 0)
            next_state_value = tf.expand_dims(next_state_value, 0)
            is_done = (is_done, )

        # Predict Q-value, for the given state_value, for all action
        q_pred_list = self.dqn_model.predict(state_value)  # [Q(s, a1), Q(s, a2), Q(s, a3), ...]

        # # One-hot each action e.g. left -> idx: 0, 4 actions -> [1, 0, 0, 0]
        action_indices = self.__to_indices(action)
        # action_onehot = tf.one_hot(action_indices, depth=4)

        # # Get the predicted Q_value for the action of each sample, 1*Q(s, a1) + 0*Q(s, a2) + ... = Q(s, a1)
        # q_pred = np.sum(q_pred_list * action_onehot)

        
        next_q_pred_list = self.dqn_model.predict(next_state_value)
        max_next_q_pred = tf.reduce_max(next_q_pred_list, axis=1) # the maximum next_q_value 
        max_next_q_pred.stop_gradient = True # Stop gradient decent

        if is_done:
            q_target = reward
        else:
            q_target = reward + self.gamma * max_next_q_pred

        idx = np.arange(0, len(state_value))
        q_pred_list[idx, action_indices] = q_target

        self.dqn_model.fit(x_train=state_value, y_train=q_target, batch_size=len(state_value))
