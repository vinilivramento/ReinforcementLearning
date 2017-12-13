#Author: Vinicius Livramento
#File: q_learning.py

#Implementation of a Q-Learning agent and a Deep Q-Learning agent to play FrozenLake
#For simple games with 1D states, the Q-Learning agent performs better than the Deep Q-Learning agent

import gym
import numpy as np
import random

from abc import ABC, abstractmethod
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Abstract agent class
class Agent(ABC):
    def __init__(self, num_actions, learning_rate):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        #learning parameters
        self.epsilon = 1.0 #exploration rate
        self.min_epsilon = 0.001 #min exploration rate
        self.epsilon_decay = 0.995
        self.gamma = 0.95    # future state discount rate #reward app 69 steps in the future counting as much as half of the immediate reward: 0.99^69 ~ 0.5
        np.random.seed(3) #ensure the same result for consecutive runs
    
    @abstractmethod
    def act(self, state):
        raise NotImplementedError()

    @abstractmethod
    def update(self, cur_state, action, reward, next_state, done):
        raise NotImplementedError()

    # This method is only implemented by Deep Q-Learning agents, which use experience replay
    def replay(self):
        pass

    def train(self, env, num_episodes=2000, max_actions_per_episode=100):
        total_reward = 0
        for episode in range(num_episodes):
            cur_state = env.reset()
            for act_count in range(max_actions_per_episode):
                action = self.act(cur_state)
                next_state, reward, done, _ = env.step(action)
                self.update(cur_state, action, reward, next_state, done)
                total_reward += reward
                cur_state = next_state
                if done: break
            if episode % 100 == 0:
                print("Episode {}/{}: r: {} e: {:.2}" .format(episode, num_episodes, total_reward/100, self.epsilon))
                total_reward = 0
            self.replay()
            self.epsilon = max(self.min_epsilon, self.epsilon*self.epsilon_decay)

    def test(self, env, num_episodes=100, max_actions_per_episode=100):
        total_reward = 0
        self.epsilon = 0 #ensure to always get the highest cost
        for episode in range(num_episodes):
            cur_state = env.reset()
            for act_count in range(max_actions_per_episode):
                # env.render()
                action = self.act(cur_state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                cur_state = next_state
                if done: break
        print("Test Score: ", total_reward/num_episodes)

    @classmethod
    def random_actions(cls, env, num_actions=1000):
        env.reset()
        for _ in range(num_actions):
            env.render()
            _, _, done, _ = env.step(env.action_space.sample()) # take a random action
            if done: return

#Simple Qlearning agent to play games with 1D states
class QLearningAgent(Agent):
    def __init__(self, num_actions, num_states, learning_rate):
        super().__init__(num_actions, learning_rate)
        self.Q = np.zeros((num_states,self.num_actions))

    #Epsilon-greedy policy
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            #if all actions have the same Q-cost, then sample one at random
            max_action_index = np.where(self.Q[state,] == np.max(self.Q[state,]))[0]
            if max_action_index.shape[0] > 1:
                return np.random.choice(max_action_index)
            else:
                return int(max_action_index)

    # This function updates the QTable
    def update(self, cur_state, action, reward, next_state, done):
        #Bellman equation
        q_val = self.Q[cur_state, action] 
        self.Q[cur_state, action] = q_val + self.learning_rate*(reward + self.gamma * self.Q[next_state, self.act(next_state)] - q_val)

class DQNAgent(Agent):
    def __init__(self, num_actions, state_size, batch_size, learning_rate):
        super().__init__(num_actions, learning_rate)
        self.batch_size = batch_size
        self.state_size = state_size
        self.memory = deque(maxlen=2000)
        self.model = self.__buil_nn_model()

    #Build Deep Neural Network for Q-Learning
    def __buil_nn_model(self):
        model = Sequential()
        model.add(Dense(25, input_dim=self.state_size, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def __update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def __predict(self, x):
        return self.model.predict(x)

    #Preprocess to fit Keras shape requirements (batch_number, features)
    def __preprocess(self, state):
        return np.reshape(state, (1, self.state_size))

    #Epsilon-greedy policy
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.__predict(self.__preprocess(state)))

    def update(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size: return

        minibatch = random.sample(self.memory, self.batch_size)
        cur_states  = np.array([x[0] for x in minibatch])
        actions     = np.array([x[1] for x in minibatch])
        rewards     = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones       = np.array([x[4] for x in minibatch])

        targets = self.__predict(cur_states)
        fut_action = self.__predict(next_states)
        for i in range(self.batch_size):
            targets[i, actions[i]] = rewards[i] if dones[i] else rewards[i] + self.gamma * np.amax(fut_action[i]) 
        history = self.model.fit(cur_states, targets, batch_size=self.batch_size, epochs=1, verbose=0) #single gradient update over one batch of samples

if __name__ == "__main__":
    # GAME = 'CartPole-v1'
    GAME = 'FrozenLake-v0'
    env = gym.make(GAME)
    num_actions = env.action_space.n

    if len(env.observation_space.shape) == 0:
        num_states = env.observation_space.n
        state_size = 1
    else:
        state_size = env.observation_space.shape[0]

    print("---Running Q-Learning Agent---")
    agent = QLearningAgent(num_actions, num_states, learning_rate=0.5)
    agent.train(env)
    agent.test(env)

    print("---Running Deep Q-Learning Agent---")
    agent = DQNAgent(num_actions, state_size, batch_size=32, learning_rate=0.001)
    agent.train(env)
    agent.test(env)


