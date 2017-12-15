## Author: Vinicius Livramento
## File: deep_qlearning.py

## Implementation of a Q-Learning agent and a Deep Q-Learning agent to play FrozenLake
## For simple games with 1D states, the Q-Learning agent performs better than the Deep Q-Learning agent

import gym
import numpy as np
import random

from abc import ABC, abstractmethod
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


##------------------ Q-Learning Agents -----------------------

## Abstract agent class
class Agent(ABC):
    def __init__(self, num_actions, learning_rate, epsilon, min_epsilon, epsilon_decay, gamma):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        ##  learning parameters
        self.epsilon = epsilon ## exploration rate
        self.min_epsilon = min_epsilon ## min exploration rate
        self.epsilon_decay = epsilon_decay 
        self.gamma = gamma ## future state discount rate, e.g. gamma=0.00, means 69 steps in the future counting as much as half of the immediate reward: 0.99^69 ~ 0.5
        np.random.seed(100) ## ensure the same result for consecutive runs
    
    @abstractmethod
    def act(self, state):
        raise NotImplementedError()

    @abstractmethod
    def update(self, cur_state, action, reward, next_state, done):
        raise NotImplementedError()

    ##  This method is only implemented by Deep Q-Learning agents, which use experience replay
    def replay(self):
        pass

   ## This method is only implemented by the Double DQN agent 
    def update_target_model(self):
        pass

    def train(self, env, num_episodes, game):
        total_reward = 0
        for episode in range(num_episodes):
            cur_state = env.reset()
            for act_count in range(game.max_actions_per_episode()):
                action = self.act(cur_state)
                next_state, reward, done, _ = env.step(action)
                self.update(cur_state, action, reward, next_state, done)
                total_reward += reward
                cur_state = next_state
                if done: break
            if episode % 100 == 0:
                score = total_reward/100
                print("Episode {}/{}: score: {} epsilon: {:.2}" .format(episode, num_episodes, score, self.epsilon))
                if score >= game.score_to_be_solved():
                    print("Solved! Achieved score: {} >= {}" .format(score, game.score_to_be_solved()))
                    return
                total_reward = 0
            self.replay()
            self.epsilon = max(self.min_epsilon, self.epsilon*self.epsilon_decay)
        print("Failure! Not achieved a score >= {}" .format(game.score_to_be_solved()))

    def test(self, env, game):
        num_episodes=100
        total_reward = 0
        self.epsilon = -1 ## ensure to always get the highest cost
        for episode in range(num_episodes):
            cur_state = env.reset()
            for act_count in range(game.max_actions_per_episode()):
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

## Simple Qlearning agent to play games with 1D states
class QLearningAgent(Agent):
    def __init__(self, num_actions, num_states, learning_rate, epsilon, min_epsilon, epsilon_decay, gamma):
        super().__init__(num_actions, learning_rate, epsilon, min_epsilon, epsilon_decay, gamma)
        self.Q = np.zeros((num_states,self.num_actions))

    ## Epsilon-greedy policy
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            ## if all actions have the same Q-cost, then sample one at random
            max_action_index = np.where(self.Q[state,] == np.max(self.Q[state,]))[0]
            if max_action_index.shape[0] > 1:
                return np.random.choice(max_action_index)
            else:
                return int(max_action_index)

    ## Update the QTable
    def update(self, cur_state, action, reward, next_state, done):
        #Bellman equation
        q_val = self.Q[cur_state, action] 
        self.Q[cur_state, action] = q_val + self.learning_rate*(reward + self.gamma * self.Q[next_state, self.act(next_state)] - q_val)

## Deep Q-Learning Agent
class DQNAgent(Agent):
    def __init__(self, num_actions, state_size, batch_size, learning_rate, epsilon, min_epsilon, epsilon_decay, gamma):
        super().__init__(num_actions, learning_rate, epsilon, min_epsilon, epsilon_decay, gamma)
        self.batch_size = batch_size
        self.state_size = state_size
        self.memory = deque(maxlen=2000)
        self.model = self.buil_nn_model()

    ## Build Deep Neural Network for Q-Learning
    def buil_nn_model(self):
        model = Sequential()
        model.add(Dense(50, input_dim=self.state_size, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    ## Preprocess to fit Keras shape requirements (batch_number, features)
    def preprocess(self, state):
        return np.reshape(state, (1, self.state_size))

    ## Epsilon-greedy policy
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.model.predict(self.preprocess(state)))

    def update(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def break_minibatch_into_arrays(self):
        minibatch = random.sample(self.memory, self.batch_size)
        cur_states  = np.array([x[0] for x in minibatch])
        actions     = np.array([x[1] for x in minibatch])
        rewards     = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones       = np.array([x[4] for x in minibatch])
        return cur_states, actions, rewards, next_states, dones

    ## Y = (r + gamma*argmaxQ(s',a'; weights) - Q(s,a; weights))
    def replay(self):
        if len(self.memory) < self.batch_size: return
        cur_states, actions, rewards, next_states, dones = self.break_minibatch_into_arrays()
        
        Y = self.model.predict(cur_states)
        fut_action = self.model.predict(next_states)
        for i in range(self.batch_size):
            Y[i, actions[i]] = rewards[i] if dones[i] else rewards[i] + self.gamma * np.amax(fut_action[i]) 
        history = self.model.fit(cur_states, Y, batch_size=self.batch_size, epochs=1, verbose=0) #single gradient update over one batch of samples

## Implement double deep q-learning algorithm to overcome the drawback of DQN (i.e. overestimate the value of actions, 
## because the same NN is used to compute action and target): a target network (to compute next state actions) and 
## an online network (to compute actions)
class DualDQNAgent(DQNAgent):
    def __init__(self, num_actions, state_size, batch_size, learning_rate, epsilon, min_epsilon, epsilon_decay, gamma):
        super().__init__(num_actions, state_size, batch_size, learning_rate, epsilon, min_epsilon, epsilon_decay, gamma)
        self.target_model = self.buil_nn_model()

    ## copy the weights from the online network (model) to the target network (target_model)
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    ## Loss = (r + gamma*argmaxQ(s',a'; weights) - Q(s,a; weights))
    def replay(self):
        if len(self.memory) < self.batch_size: return
        cur_states, actions, rewards, next_states, dones = self.break_minibatch_into_arrays()
        
        Y = self.model.predict(cur_states)
        fut_action_model = self.model.predict(next_states)
        fut_action_target_model = self.target_model.predict(next_states)
        for i in range(self.batch_size):
            Y[i, actions[i]] = rewards[i] if dones[i] else rewards[i] + self.gamma * fut_action_target_model[i][np.amax(fut_action_model[i])]
        history = self.model.fit(cur_states, Y, batch_size=self.batch_size, epochs=1, verbose=0) #single gradient update over one batch of samples

