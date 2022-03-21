import config
import random
import numpy as np
import tensorflow as tf
from collections import deque
from dqn import DEEP_Q_LEARNING_NETWORK

class AGENT:
    def __init__(self, env, model_file_path=None) -> None:
        self.env = env
        self.dqn = DEEP_Q_LEARNING_NETWORK()
        if model_file_path != None:
            self.dqn = self.dqn.load_model(model_file_path)
        self.game_record = 0
        self.round_count = 0
        self.memory = deque(maxlen=config.MAX_MEMORY)
        self.stop_training = False

    def demo(self) -> None:
        # TODO demo play step
        pass

    def training(self) -> None:
        """ AGENT.training()
        training steps:
        1) Get the current game_state
        2) Get decided move_dir based on the game_state
        3) Get the move_reward, is_done status, round_score, by perform the move_dir
        4) Get the next_game_state, by perform the move_dir
        5) Integrate the state_move_sample = [game_state, move_dir, move_reward, next_game_state, is_done]
        6) Train the dqn_model with one sample of game status, state_move_sample
        7) Store/Remember the state_move_sample into pre-defined memory
        8) If the game is_done:
            - Reset game_env
            - Increment the round_count
            - If the round_score > game_record:
                - renew the record, save the model
            - Train the dqn_model with a batch of state_move_samples
            - Print round_info, training_status
        """
        game_state = self.get_game_state() #1
        move_dir = self.get_action(game_state) #2
        move_reward, is_done, round_score = self.play(move_dir) #3
        next_game_state = self.get_game_state() #4
        state_move_sample = [game_state, move_dir, move_reward, next_game_state, is_done] #5
        self.dqn.train(training_sample=state_move_sample, batch_size=1) #6
        self.memory.append(state_move_sample) #7
        if is_done: #8
            self.env.reset_game_state()
            self.round_count += 1
            if round_score > self.game_record:
                print('New record', self.game_record)
                self.game_record = round_score
                self.dqn.save_model()
            self.dqn_batch_train(batch_size=config.BATCH_SIZE)
            print('Round', self.round_count)
        if self.game_record == 2:
            self.stop_training = True
        return self.stop_training, self.game_record

    def get_game_state(self):
        food_pos = self.env.food.position
        snake_sight = config.SNAKE_SIGHT_DISTANCE
        snake_head = self.env.snake.body[0]
        snake_dir = self.env.snake.direction

        if snake_dir.x == 0:
            snake_dir_L = config.Vector2(snake_dir.y, 0)
            snake_dir_R = config.Vector2(-snake_dir.y, 0)
        elif snake_dir.y == 0:
            snake_dir_L = config.Vector2(0, -snake_dir.x)
            snake_dir_R = config.Vector2(0, snake_dir.x)

        state_status = [
            # Snake 4 heading direction
            snake_dir == config.DIR_UP, 
            snake_dir == config.DIR_DOWN,
            snake_dir == config.DIR_LEFT, 
            snake_dir == config.DIR_RIGHT,

            # Danger detection
            self.env.is_danger(snake_head + snake_sight * snake_dir),   # Danger Ahead
            self.env.is_danger(snake_head + snake_sight * snake_dir_L),    # Danger Left
            self.env.is_danger(snake_head + snake_sight * snake_dir_R),    # Danger Right

            # Food censoring
            food_pos.x < snake_head.x,
            food_pos.x > snake_head.x,
            food_pos.y < snake_head.y,
            food_pos.y > snake_head.y
        ]
        # print(state_status)
        return np.array(state_status, dtype=int)

    def get_action(self, state):

        epsilon = 100 - self.round_count
        onehot_action = [0, 0, 0, 0]

        if random.randint(0, 300) < epsilon:
            move = random.randint(0, 3)
            onehot_action[move] = 1
        else:
            q_pred_list = self.dqn.predict(state)
            move = tf.math.argmax(q_pred_list).numpy()
            onehot_action[move] = 1

        if onehot_action == [1, 0, 0, 0]:
            agent_action = config.DIR_LEFT
        elif onehot_action == [0, 1, 0, 0]:
            agent_action = config.DIR_RIGHT
        elif onehot_action == [0, 0, 1, 0]:
            agent_action = config.DIR_UP
        elif onehot_action == [0, 0, 0, 1]:
            agent_action = config.DIR_DOWN
        else:
            print('Get wrong action', onehot_action)

        return agent_action


        # if self.round_count <= config.BATCH_SIZE:
        #     action_list = [
        #         config.DIR_UP,
        #         config.DIR_DOWN,
        #         config.DIR_LEFT,
        #         config.DIR_RIGHT
        #     ]
        #     rdm_act_idx = np.random.randint(0, 3)
        #     return action_list[rdm_act_idx]
        # else:
        #     agent_action = self.dqn.predict(state)
        #     return np.rint(agent_action[0])

    def play(self, action):
        return self.env.agent_play(action[0], action[1])

    def dqn_batch_train(self, batch_size):
        if len(self.memory) < batch_size:
            # memory is less short, collect all
            sample_batch = self.memory
            print(self.memory)
        else:
            # collect a random batch of samples from memory
            sample_batch = random.sample(self.memory, batch_size)

        # Send to deep q-learning network to lean/train/fit
        states, actions, rewards, next_states, is_dones = zip(*sample_batch)
        self.dqn.train(training_sample=[states, actions, rewards, next_states, is_dones], batch_size=batch_size)