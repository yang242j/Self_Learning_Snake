import os
import copy
import config
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
from pygame.math import Vector2

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.models import Sequential, load_model

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
EPSILON = config.EPSILON # @param {type:"number"}
EPSILON_DECAY_RATE = config.EPSILON_DECAY_RATE # @param {type:"number"}
EPSILON_MIN = config.EPSILON_MIN # @param {type:"number"}
BATCH_SIZE = config.BATCH_SIZE # @param {type:"integer"}
MAX_MEMORY = config.MAX_MEMORY # @param {type:"integer"}

STATE_LEN = 19
ACTION_RANGE = 3
FC_DIM = config.FC_DIM # @param {type:"integer"}

class ReplayMemory(object):
    """ Replay Memory Pool
    Store the traing data in a memory pool for future training
    Random sample training data to break the corelation time bond
    """
    def __init__(self) -> None:
        """INIT
        define the memory as deque with fixed max length 
        """
        self.memory = collections.deque(maxlen=MAX_MEMORY)
    
    def append(self, sample):
        """ReplayMemory.append()
        Append the memory pool with given sample, popleft if the max memory been reached
        
        INPUT: sample {list}, one training sample
        """
        self.memory.append(sample)
        # print ("{:.2%}".format(len(self.memory)/MAX_MEMORY)) # Uncomment to see memory loading

    def random_sample(self, batch_size):
        """ReplayMemory.random_sample()
        Collect random samples as a minibatch, return empty if not enough

        INPUT: batch_size {int}, how many data need
        """
        if len(self.memory) < batch_size:
            return [], 0
        else:
            sample_batch = random.sample(self.memory, batch_size)
            new_batch_size = len(sample_batch)
            return sample_batch, new_batch_size

class NeuralNetwork(object):
    """NeuralNetwork
    Define a simple neural network to learn and predict Q-values with given state_values
    
    NN_INPUT: state_value_list
    NN_OUTPUT: Q-values for each action
    """
    def __init__(self, input_shape, output_shape, fc_dim, learning_rate, model_file_path=None) -> None:
        """ INIT
        Load the model from given file_path if given, otherwise, build new NN

        INPUT:
            input_shape: {int} how many neurons in input layer
            output_shape: {int} how many neurons in output layer
            fc_dim: {int} how many neurons in each hidden layer
            learning_rate: {float} how much does the model learn through each training cycle
            model_file_path: {string} where to find the pre-trained model
        """
        if model_file_path == None:
            # Create new model to train the agent
            self.build(input_shape, output_shape, fc_dim, learning_rate)
        else:
            # Load the old_model and start from where left
            self.load_model(model_file_path)

    def build(self, input_shape, output_shape, fc_dim, learning_rate):
        """ NeuralNetwork.build()
        1) Build a sequential neural network model with 
            one input_layer, 
            two fully connected hidden layers with ReLu activation function, and 
            one outpur_layer

        2) Compile the model with
            Adam optimizer, and
            mean_squared_error loss

        INPUT:
            input_shape: {int} how many neurons in input layer
            output_shape: {int} how many neurons in output layer
            fc_dim: {int} how many neurons in each hidden layer
            learning_rate: {float} how much does the model learn through each training cycle
        """
        self.model = Sequential([
            InputLayer(input_shape=(input_shape,)),
            Dense(fc_dim, activation='relu'),
            Dense(fc_dim, activation='relu'),
            Dense(output_shape)
        ])
        self.model.compile(
            optimizer = Adam(learning_rate=learning_rate),
            loss='mse'
        )

    def fit(self, x_train, y_train, batch_size, verbose_level) -> tf.keras.callbacks.History:
        """ NeuralNetwork.fit()
        Fit the model with given training input & output samples

        INPUT:
            x_train: {list} input training sample data
            y_train: {list} output training sample data
            batch_size: {int} how many batch of data to be trained
            verbose_level: {int} print model training message or not
                0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.
        """
        history = self.model.fit(x_train, y_train, batch_size, verbose=verbose_level)
        return history

    def predict(self, pred_x):
        """ NeuralNetwork.predict()
        Predict the Q-value-list for each action with given state-value

        INPUT:
            pred_x: {list} state_value list
        """
        return self.model.predict(pred_x)
    
    def save_model(self, model_file_path):
        # Save the model to the given file_path
        self.model.save(model_file_path)

    def load_model(self, model_file_path):
        # Load the model with the given file_path
        if not os.path.exists(model_file_path):
            print(model_file_path, 'not exist')
        self.model = load_model(model_file_path)

class DDQN_Agent(object):
    """DDQN_Agent
    Defines the algorithm to train the agent as well as make the Double-Deep-Q-Network learn

    Methods:
        - self.remember(sample_list) -> Append the sample list into the memory pool
        - self.get_action(state) -> Return the onehoted action with the max predicted Q-values
        - self.get_state() -> Return state-value list for the current game environment
        - self.play_action(onehot_action) -> Return the stop_training status,
                                          -> reward after perform the action, 
                                          -> game_over status, 
                                          -> game_score after perform the action
        - self.sync_model() -> Synchronize the weights of evaluation_model to the target_model
        - self.save_model(file_path) -> Save the model to the given file_path
        - self.load_model() -> Load the model from the pre_defined file_path in the __init__()
        - self.plot_loss() -> Plot and save the loss graph
        - self.plot_graph(score_list, mean_score_list) -> Plot and save the score graph
        - self.demo() -> Play the game with 0 epsilon and no learning
        - self.trainer() -> Agent training algorithm/steps
        - self.learn(training_sample, batch_size) -> Double-Deep-Q-Network learning algorithm
    """
    def __init__(self, game_env, model_file_path=None) -> None:
        # Init Instances
        self.env = game_env
        self.memory = ReplayMemory()
        self.dqn_eval_model = NeuralNetwork(input_shape=STATE_LEN, output_shape=ACTION_RANGE, fc_dim=FC_DIM, learning_rate=LR)
        self.dqn_targ_model = NeuralNetwork(input_shape=STATE_LEN, output_shape=ACTION_RANGE, fc_dim=FC_DIM, learning_rate=LR)
        
        # Init training parameters
        self.game_record = 0
        self.round_count = 0
        self.epsilon = EPSILON
        self.stop_training = False

        # Init plot parameters
        self.loss_list = []
        self.score_list = []
        self.mean_score_list = []

        # Init file loading
        self.default_file_path = 'model/ddqn_snake.h5'
        self.file_path = None
        if model_file_path:
            self.file_path = model_file_path
            self.load_model()
            self.epsilon=EPSILON_MIN
    
    def remember(self, sample_list) -> None:
        # Append the list of game status to the memory
        self.memory.append(sample_list)
    
    def get_action(self, state) -> list:
        """
        Algorithm: Epsilon Greedy
            Pick a random float between 0 and 1
            if the random float is less than epsilon, 
                then do random action
            else:
                then do predicted action
        Return: onehot_action
        """
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
    
    def get_state(self) -> list:
        """
        Collect values to discribe the current game state
        19 bool values to discribe the game state
        [
            snake goes up,
            snake goes down,
            snake goes left,
            snake goes right,
            danger ahead,
            danger left,
            danger left-up,
            danger left-down,
            danger right,
            danger right-up,
            danger right-down,
            food left-up,
            food up,
            food right-up,
            food left,
            food right,
            food left-down,
            food down,
            food right-down
        ]
        """
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
        """ Perform the chosen action and get action-results """
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
    
    def sync_model(self) -> None:
        """
        Set the weigh of target model by weights of evaluation model

        Alternitive Method:
            Synchronize partial weights to minimize the wrong effectiveness 
            TAU = 0.1
            new_weight = TAU * target_weight + (1-TAU) * eval_weight
        """
        # Sync the weights of dqn_eval_model to the target_model
        print('Model Sync...')
        eval_weights = self.dqn_eval_model.model.get_weights()
        self.dqn_targ_model.model.set_weights(eval_weights)
    
    def save_model(self, file_path):
        # Save the model to the file_path
        self.dqn_eval_model.save_model(file_path)
        print('model saved ->', file_path)
    
    def load_model(self):
        # Load the evaluation model from the file_path and deep copy it as the target model
        print('loading from ', str(self.file_path))
        self.dqn_eval_model.load_model(self.file_path)
        # print(self.dqn_eval_model.model.summary())
        self.dqn_targ_model = copy.deepcopy(self.dqn_eval_model)
        self.dqn_targ_model.build(STATE_LEN, ACTION_RANGE, FC_DIM, LR)
        # print(self.dqn_targ_model.model.summary())
        print('loading success')
    
    def plot_loss(self):
        # Plot loss
        plt.plot(self.loss_list, label='Mean_Squared_Error Loss')
        plt.text(
                x = len(self.loss_list)-1, # Last index of the score list
                y = self.loss_list[-1], # Last score in the list
                s = str(self.loss_list[-1]) # Last score
            )

        # Plot init
        plt.style.use('ggplot')
        plt.title('Training Loss')
        plt.xlabel('Number of Rounds')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        # save plot
        plt.savefig('plots/loss_plt.png')

        # Clear the figure and axes
        plt.clf()
        plt.cla()

    def plot_graph(self):
        # Plot Scores
        plt.plot(self.score_list, label='round_score')
        plt.text(
                x = len(self.score_list)-1, # Last index of the score list
                y = self.score_list[-1], # Last score in the list
                s = str(self.score_list[-1]) # Last score
            )

        # Plot Mean Scores
        plt.plot(self.mean_score_list, label='total_score_mean')
        plt.text(
                x = len(self.mean_score_list)-1, # Last index of the mean list
                y = self.mean_score_list[-1], # Last mean
                s = str(self.mean_score_list[-1]) # Last mean
            )

        # Plot init
        plt.style.use('ggplot')
        plt.title('Training Score')
        plt.xlabel('Number of Rounds')
        plt.ylabel('Score')
        plt.legend(loc='upper left')

        # save plot
        plt.savefig('plots/score_mean_plt.png')

        # Clear the figure and axes
        plt.clf()
        plt.cla()
    
    def demo(self):
        """
        Let the agent play the game without learning and randomness
        """
        self.epsilon = 0
        state = self.get_state()
        onehot_action = self.get_action(state)
        stop_demo, _, is_done, round_score = self.play_action(onehot_action)
        if is_done:
            self.env.reset_game_state()
            if round_score > self.game_record:
                print('New Record:', round_score)
                self.game_record = round_score
            self.round_count += 1
            print('Round', self.round_count, 'Score', round_score, 'Record', self.game_record)
        return stop_demo, self.game_record
    
    def trainer(self):
        """
        Double Deep Q-learning Network Training Algorithm
            1) Get current game state
            2) Let the agent decide what action to take, based on the current game state
            3) Perform the action and get action status from the game
            4) Get next game state after action
            5) Store the data into the Replay Memory Pool
            6) If round is over,
                - Reset game
                - Store game record if record break
                - Random collect training data as minibatch
                - Agent learn through the minibatch
                - Plot graphs and print messages
        """
        # Collect data
        curr_state = self.get_state()
        onehot_action = self.get_action(curr_state)
        self.stop_training, move_reward, is_done, round_score = self.play_action(onehot_action)
        new_state = self.get_state()
        
        # Remember data
        state_action_sample = [curr_state, onehot_action, move_reward, new_state, is_done]
        self.remember(state_action_sample)
        
        # Round is over
        if is_done:
            self.env.reset_game_state()
            if round_score > self.game_record:
                self.game_record = round_score
                print('New record:', self.game_record)
                self.save_model(self.default_file_path)
            sample_batch, new_batch_size = self.memory.random_sample(batch_size=BATCH_SIZE)
            mse_loss = 0
            if new_batch_size != 0:
                mse_loss = self.learn(sample_batch, batch_size=new_batch_size)[0]         
            self.round_count += 1
            print('Round', self.round_count, 'Score', round_score, 'Record', self.game_record, 'Epsilon %.4f' % self.epsilon)
                       
            # Plot score graph
            self.score_list.append(round_score)
            self.mean_score_list.append(np.mean(self.score_list))
            self.plot_graph()

            # Plot loss graph
            self.loss_list.append(mse_loss)
            self.plot_loss()
            
        return self.stop_training, self.game_record
    
    def learn(self, training_sample, batch_size):
        """
        Double Deep Q-learning Network Learning Algorithm
            For each sample in the minibatch,
                - if it is the last action of round, new_q_value = the round reward
                - else, new_q_value = reward + gamma * eval_q_value(future_sate, action_max_targ_q_future_value)
                - Fit the model with current state value and the new_q_value
                - minimizing the loss
                - sync the model every 100 steps
        """
        # Data processing
        assert len(training_sample) == batch_size
        curr_states, onehot_actions, move_rewards, new_states, is_dones = zip(*training_sample)
        curr_states = np.array(curr_states, dtype=np.float32)
        onehot_actions = np.array(onehot_actions, dtype=np.int32)
        move_rewards = np.array(move_rewards, dtype=np.float32)
        new_states = np.array(new_states, dtype=np.float32)
        is_dones = np.array(is_dones, dtype=np.int32)

        """"""""
        # predict current state Q-values by using both model
        q_eval = self.dqn_eval_model.predict(curr_states)
        q_targ = self.dqn_targ_model.predict(curr_states) # Not-in-use
        
        # predict next state Q-values by using both model
        q_eval_new = self.dqn_eval_model.predict(new_states)
        q_targ_new = self.dqn_targ_model.predict(new_states)
        
        # Calculate new target Q-values
        max_act_idx = np.argmax(q_targ_new, axis=1).astype(int)
        sample_idx_range = np.arange(batch_size, dtype=np.int32)
        new_q_target = move_rewards + GAMMA * q_eval_new[sample_idx_range, max_act_idx] * (1-is_dones)

        # Update new q_targets
        updated_q_targets = q_eval.copy()
        action_idx_range = np.arange(3, dtype=np.int8)
        action_idx = np.dot(onehot_actions, action_idx_range) #(n, 3) dot (3,) => n
        updated_q_targets[sample_idx_range, action_idx] = new_q_target

        # Fit dqn_eval_model with curr_states and updated_q_targets
        history = self.dqn_eval_model.fit(curr_states, updated_q_targets, batch_size, verbose_level=0)
        mse_loss = history.history['loss']
        """"""""

        # Decay epsilon
        self.epsilon = max(self.epsilon*EPSILON_DECAY_RATE, EPSILON_MIN)
        
        # Sync. model every 100 steps
        if self.round_count != 0 and self.round_count % 100 == 0:
            self.sync_model()

        return mse_loss

