## Author: Vinicius Livramento
## File: games.py

## Classes to run OpenAI Gym games

import argparse

from deep_qlearning import *

##------------------ Utilitary Classes For Handling Different Games -----------------------

## Abstract class for playing different OpenAI Gym games
## https://gym.openai.com/envs/
class Game(ABC):
    @abstractmethod
    def name(self):
        raise NotImplementedError()
   
    ## Score defined by OpenAI Gym for a game to be solved over 100 epochs
    @abstractmethod
    def score_to_be_solved(self):
        raise NotImplementedError()

    @abstractmethod
    def train_and_play(self):
        raise NotImplementedError()

    @abstractmethod
    def play(self):
        raise NotImplementedError()

    @abstractmethod
    def max_actions_per_episode(self):
        raise NotImplementedError()

## https://gym.openai.com/envs/CartPole-v0/
class CardPole(Game):
    def name(self):
        return 'CartPole-v1'

    def score_to_be_solved(self):
        return 195.0

    def train_and_play(self):
        env = gym.make(self.name())
        print("---Running Deep Q-Learning Agent---")
        agent = DQNAgent(num_actions=env.action_space.n, 
                         state_size=env.observation_space.shape[0], 
                         batch_size=32, 
                         learning_rate=0.001,
                         epsilon=1.0,
                         min_epsilon=0.01, 
                         epsilon_decay=0.995, 
                         gamma=0.95)

        agent.train(env,
                    num_episodes=3000,
                    game=self)
        agent.test(env,
                   game=self)

    def play(self):
        pass

    def max_actions_per_episode(self):
        return 200

## https://gym.openai.com/envs/FrozenLake-v0/
class FrozenLake(Game):
    def name(self):
        return 'FrozenLake-v0'

    def score_to_be_solved(self):
        return 0.78

    # Use the classic qlearning agent
    def train_and_play_using_qlearning_agent(self):
        env = gym.make(self.name())
        print("---Running classic Q-learning Agent---")
        agent = QLearningAgent(num_actions=env.action_space.n, 
                               num_states=env.observation_space.n,
                               learning_rate=0.2,
                               epsilon=1.0,
                               min_epsilon=0.01, 
                               epsilon_decay=0.995, 
                               gamma=0.95)

        agent.train(env,
                    num_episodes=5000,
                    game=self)
        agent.test(env,
                   game=self)

    def train_and_play(self):
        env = gym.make(self.name())
        print("---Running Deep Q-Learning Agent---")
        agent = DQNAgent(num_actions=env.action_space.n, 
                         state_size=1, 
                         batch_size=32, 
                         learning_rate=0.001,
                         epsilon=0.5,
                         min_epsilon=0.01, 
                         epsilon_decay=0.995, 
                         gamma=0.95)

        agent.train(env,
                    num_episodes=3000,
                    game=self)
        agent.test(env,
                   game=self)

    def play(self):
        pass

    def max_actions_per_episode(self):
        return 200

## https://gym.openai.com/envs/MountainCar-v0/
## https://github.com/openai/gym/wiki/MountainCar-v0
class MountainCar(Game):
    def name(self):
        return 'MountainCar-v0'

    def score_to_be_solved(self):
        return -110.0

    def train_and_play(self):
        env = gym.make(self.name())
        print("---Running Deep Q-Learning Agent---")
        agent = DQNAgent(num_actions=env.action_space.n, 
                         state_size=env.observation_space.shape[0], 
                         batch_size=32, 
                         learning_rate=0.001,
                         epsilon=1.0,
                         min_epsilon=0.01, 
                         epsilon_decay=0.995, 
                         gamma=0.95)

        agent.train(env,
                    num_episodes=3000,
                    game=self)
        agent.test(env,
                   game=self)

    def play(self):
        pass

    def max_actions_per_episode(self):
        return 200

##------------------ Utility Functions for Different Games -----------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parsing the Game to be Played.')
    parser.add_argument(dest='gameName',
                        default='CardPole',
                        help='Specify the name of the game to be played. Default: CardPole')

    args = parser.parse_args()
    print("Game: ", args.gameName)

    runDict = {"FrozenLake" : FrozenLake().train_and_play_using_qlearning_agent,
               "CardPole"   : CardPole().train_and_play,
               "MountainCar": MountainCar().train_and_play}

    runDict[args.gameName]()
    
