import os
import copy
import config
import random
import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.losses import mean_squared_error
from pygame.math import Vector2
import matplotlib.pyplot as plt
from IPython import display

"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3' 

DIR_UP = config.DIR_UP # @param {type:"Vector2"}
DIR_DOWN = config.DIR_DOWN # @param {type:"Vector2"}
DIR_LEFT = config.DIR_LEFT # @param {type:"Vector2"}
DIR_RIGHT = config.DIR_RIGHT # @param {type:"Vector2"}

LR = config.LEARNING_RATE # @param {type:"number"}
GAMMA = config.DISCOUNT_RATE # @param {type:"number"}
TAU = config.AVERAGE_RATE # @param {type:"number"}
EPSILON = config.EPSILON # @param {type:"number"}
EPSILON_DECAY_RATE = config.EPSILON_DECAY_RATE # @param {type:"number"}
EPSILON_MIN = config.EPSILON_MIN # @param {type:"number"}
BATCH_SIZE = config.BATCH_SIZE # @param {type:"integer"}
MAX_MEMORY = config.MAX_MEMORY # @param {type:"integer"}

class ReplayMemory(object):
    def __init__(self) -> None:
        self.memory = collections.deque(maxlen=MAX_MEMORY)
    
    def append(self, sample):
        self.memory.append(sample)

    def random_sample(self, batch_size):
        if len(self.memory) < batch_size:
            # sample_batch = self.memory
            return [], 0
        else:
            sample_batch = random.sample(self.memory, batch_size)
            new_batch_size = len(sample_batch)
            return sample_batch, new_batch_size

class NeuralNetwork(object):
    def __init__(self, input_shape, output_shape, fc_dim, learning_rate, model_file_path=None) -> None:
        if model_file_path == None:
            # Create new model to train the agent
            self.build(input_shape, output_shape, fc_dim, learning_rate)
        else:
            # Load the old_model and start from where left
            self.load_model(model_file_path)

    def build(self, input_shape, output_shape, fc_dim, learning_rate):
        self.model = Sequential([
            InputLayer(input_shape=(input_shape,)),
            Dense(fc_dim, activation='relu'),
            Dense(fc_dim, activation='relu'),
            # Dense(fc_dim, activation='relu'),
            # Dense(fc_dim, activation='relu'),
            # Dense(fc_dim, activation='relu'),
            Dense(output_shape)
        ])
        self.model.compile(
            optimizer = Adam(learning_rate=learning_rate),
            loss='mse'
        )

    def fit(self, x_train, y_train, batch_size, verbose_level):
        self.model.fit(x_train, y_train, batch_size, verbose=verbose_level)

    def predict(self, pred_x):
        return self.model.predict(pred_x)
    
    def save_model(self, model_file_path):
        self.model.save(model_file_path)

    def load_model(self, model_file_path):
        if not os.path.exists(model_file_path):
            print(model_file_path, 'not exist')
        self.model = load_model(model_file_path)

class DDQN_Agent(object):
    def __init__(self, game_env, model_file_path=None) -> None:
        self.env = game_env
        self.memory = ReplayMemory()
        self.dqn_eval_model = NeuralNetwork(input_shape=19, output_shape=3, fc_dim=256, learning_rate=LR)
        self.dqn_targ_model = NeuralNetwork(input_shape=19, output_shape=3, fc_dim=256, learning_rate=LR)
        self.game_record = 0
        self.round_count = 0
        self.epsilon = EPSILON
        self.stop_training = False
        self.default_file_path = 'model/ddqn_snake.h5'
        self.file_path = None
        if model_file_path:
            self.file_path = model_file_path
            self.load_model()

        self.plot_scores = []
        self.plot_mean_scores = []
        self.total_score = 0
    
    def remember(self, sample_list):
        # Append the list of game status to the memory
        self.memory.append(sample_list)
    
    def get_action(self, state):
        # Get the perdicted action based on given state
        if random.random() < self.epsilon:
            action_idx = random.randint(0, 2)
            # print('<', self.epsilon, action_idx)
        else:
            state = np.array([state], dtype=np.float32)
            q_pred = self.dqn_eval_model.predict(state)
            action_idx = np.squeeze(np.argmax(q_pred, axis=1).astype(np.int))
            # print('>', self.epsilon, action_idx, q_pred)
        onehot_action = [0, 0, 0]
        onehot_action[action_idx] = 1
        return onehot_action
    
    def get_state(self):
        # Get the game state evaluation status
        snake_sight = 1
        food_pos = self.env.food.position
        snake_head = self.env.snake.body[0]
        snake_dir = self.env.snake.direction

        if snake_dir.x == 0:
            snake_dir_L = Vector2(snake_dir.y, 0)
            snake_dir_LU = Vector2(snake_dir.y, snake_dir.y)
            snake_dir_LD = Vector2(snake_dir.y, -snake_dir.y)
            snake_dir_R = Vector2(-snake_dir.y, 0)
            snake_dir_RU = Vector2(-snake_dir.y, snake_dir.y)
            snake_dir_RD = Vector2(-snake_dir.y, -snake_dir.y)
        elif snake_dir.y == 0:
            snake_dir_L = Vector2(0, -snake_dir.x)
            snake_dir_LU = Vector2(snake_dir.x, -snake_dir.x)
            snake_dir_LD = Vector2(-snake_dir.x, -snake_dir.x)
            snake_dir_R = Vector2(0, snake_dir.x)
            snake_dir_RU = Vector2(snake_dir.x, snake_dir.x)
            snake_dir_RD = Vector2(-snake_dir.x, snake_dir.x)
        
        state_status = [ # 19 
            # Snake heading direction 0, 1, 0, 0
            snake_dir == DIR_UP, 
            snake_dir == DIR_DOWN,
            snake_dir == DIR_LEFT, 
            snake_dir == DIR_RIGHT,

            # Danger detection, in seven direction, no back
            self.env.is_danger(snake_head + snake_sight * snake_dir),      # Danger Ahead
            self.env.is_danger(snake_head + snake_sight * snake_dir_L),    # Danger Left
            self.env.is_danger(snake_head + snake_sight * snake_dir_LU),    # Danger Left-Up
            self.env.is_danger(snake_head + snake_sight * snake_dir_LD),    # Danger Left-Down
            self.env.is_danger(snake_head + snake_sight * snake_dir_R),    # Danger Right
            self.env.is_danger(snake_head + snake_sight * snake_dir_RU),    # Danger Right-Up
            self.env.is_danger(snake_head + snake_sight * snake_dir_RD),    # Danger Right-Down

            # Food censoring, in eight direction
            food_pos.x < snake_head.x and food_pos.y < snake_head.y,
            food_pos.x == snake_head.x and food_pos.y < snake_head.y,
            food_pos.x > snake_head.x and food_pos.y < snake_head.y,
            food_pos.x < snake_head.x and food_pos.y == snake_head.y,
            food_pos.x > snake_head.x and food_pos.y == snake_head.y,
            food_pos.x < snake_head.x and food_pos.y > snake_head.y,
            food_pos.x == snake_head.x and food_pos.y > snake_head.y,
            food_pos.x > snake_head.x and food_pos.y > snake_head.y
        ]
        # print(state_status, food_pos, snake_head, snake_dir, snake_dir_L, snake_dir_R)
        return np.array(state_status, dtype=np.int32)
    
    def play_action(self, onehot_action):
        # Perform the chosen action and get action-results
        # Get current snake direction
        snake_dir = self.env.snake.direction

        # Get the left and right direction of the current direction
        if snake_dir.x == 0:
            snake_dir_L = Vector2(snake_dir.y, 0)
            snake_dir_R = Vector2(-snake_dir.y, 0)
        elif snake_dir.y == 0:
            snake_dir_L = Vector2(0, -snake_dir.x)
            snake_dir_R = Vector2(0, snake_dir.x)

        # Convert based on input onehot action
        if onehot_action == [1, 0, 0]: # Go straight
            new_dir = snake_dir
        elif onehot_action == [0, 1, 0]: # Turn Left
            new_dir = snake_dir_L
        elif onehot_action == [0, 0, 1]: # Turn Right
            new_dir = snake_dir_R
        else:
            print('Get wrong action', onehot_action)
        
        # Perform the new direction, return the returned values
        return self.env.agent_play(new_dir.x, new_dir.y)
    
    def sync_model(self):
        # Sync the weights of dqn_eval_model to the target_model
        print('Model Sync...')
        eval_weights = self.dqn_eval_model.model.get_weights()
        # targ_weights = self.dqn_targ_model.model.get_weights()
        # new_weights = TAU * np.array(eval_weights, dtype=float) + (1-TAU) * np.array(targ_weights, dtype=float)
        self.dqn_targ_model.model.set_weights(eval_weights)
    
    def save_model(self, file_path):
        self.dqn_eval_model.save_model(file_path)
        print('model saved ->', file_path)
    
    def load_model(self):
        print('loading from ', str(self.file_path))
        self.dqn_eval_model.load_model(self.file_path)
        self.dqn_targ_model = copy.deepcopy(self.dqn_eval_model).model
        print('loading success')
    
    def plot_graph(self, scores, mean_scores):
        # TODO
        plt.ion()
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores)
        plt.plot(mean_scores)
        plt.ylim(ymin=0)
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    
    def demo(self):
        self.epsilon = 0
        state = self.get_state()
        onehot_action = self.get_action(state)
        stop_demo, _, is_done, round_score = self.play_action(onehot_action)
        # state = self.get_state()
        if is_done:
            self.env.reset_game_state()
            if round_score > self.game_record:
                print('New Record:', round_score)
                self.game_record = round_score
            self.round_count += 1
            print('Round', self.round_count, 'Score', round_score, 'Record', self.game_record)
        return stop_demo, self.game_record
    
    def trainer(self):
        # Collect and remember [curr_state, onehot_action, move_reward, new_state, is_done] status
        curr_state = self.get_state()
        onehot_action = self.get_action(curr_state)
        self.stop_training, move_reward, is_done, round_score = self.play_action(onehot_action)
        new_state = self.get_state()
        
        state_action_sample = [curr_state, onehot_action, move_reward, new_state, is_done]
        self.remember(state_action_sample)
        # _ = self.learn([state_action_sample], batch_size=1)
        
        if is_done:
            self.env.reset_game_state()
            if round_score > self.game_record:
                self.game_record = round_score
                print('New record:', self.game_record)
                self.save_model(self.default_file_path)
            sample_batch, new_batch_size = self.memory.random_sample(batch_size=BATCH_SIZE)
            mse_loss = 0
            if new_batch_size != 0:
                mse_loss = self.learn(sample_batch, batch_size=new_batch_size)            
            self.round_count += 1
            print('Round', self.round_count, 'Score', round_score, 'Record', self.game_record, 'MSE_loss', mse_loss, 'Epsilon', self.epsilon)
                       
            # TODO: Print traning messages and plots
            self.plot_scores.append(round_score)
            self.total_score += round_score
            mean_score = self.total_score / self.round_count
            self.plot_mean_scores.append(mean_score)
            self.plot_graph(self.plot_scores, self.plot_mean_scores)
            
        return self.stop_training, self.game_record
    
    def learn(self, training_sample, batch_size):
        # Data processing
        assert len(training_sample) == batch_size
        curr_states, onehot_actions, move_rewards, new_states, is_dones = zip(*training_sample)
        curr_states = np.array(curr_states, dtype=np.float32)
        onehot_actions = np.array(onehot_actions, dtype=np.int32)
        move_rewards = np.array(move_rewards, dtype=np.float32)
        new_states = np.array(new_states, dtype=np.float32)
        action_idx_range = np.array([0, 1, 2], dtype=np.int32)
        action_idx = np.dot(onehot_actions, action_idx_range) #(n, 3) dot (3,) => n

        """"""""
        # predict current state Q-values by using both model
        q_eval = self.dqn_eval_model.predict(curr_states)
        q_targ = self.dqn_targ_model.predict(curr_states) # Not-in-use
        
        # predict next state Q-values by using both model
        q_eval_new = self.dqn_eval_model.predict(new_states)
        q_targ_new = self.dqn_targ_model.predict(new_states)
        
        # Update new q_targets
        updated_q_targets = q_eval.copy()
        max_act_idx = np.argmax(q_targ_new, axis=1).astype(np.int32)
        sample_idx = np.arange(batch_size, dtype=np.int32)
        updated_q_targets[sample_idx, action_idx] = move_rewards + GAMMA * q_eval_new[sample_idx, max_act_idx] * (1-bool(is_dones))
        
        # Fit dqn_eval_model with curr_states and updated_q_targets
        self.dqn_eval_model.fit(curr_states, updated_q_targets, batch_size, verbose_level=0)
        """"""""
        
        # Calculate Mean_Squared_Error loss
        mse_loss = mean_squared_error(q_eval, updated_q_targets).numpy()[0]

        # Decay epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon = self.epsilon * EPSILON_DECAY_RATE 
        else:
            # print('epsilon reaches minimum', EPSILON_MIN)
            self.epsilon = EPSILON_MIN
        
        # Sync. dqn_targ_model with weights of dqn_eval_model
        if self.round_count != 0 and self.round_count % 100 == 0:
            self.sync_model()

        return mse_loss

