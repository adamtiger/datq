import os
import gym
from gym import wrappers
import numpy as np
from environment import Environment as E


class Q:

    def __init__(self, clustering, epsilon, gamma, max_iter, env_name, record_video, folder):
        '''
        This class trains a Q-table. The input state is derived from the clustered
        encoded state.

        clustering - a function to determine which cluster the next processed input corresponds to
        epsilon - the exploration ratio
        gamma - discounting factor
        max_iter - the number of iteration in Q-learning
        env_name - name of the Atari environment
        record_video - record a video about the agent during evaluation
        folder - 
        '''
        self.clustering = clustering
        self.epsilon = epsilon
        self.gamma = gamma
        self.max_iter = max_iter
        self.env_name = env_name
        self.record_video = record_video
        self.folder = folder
    
    def __env_init(self):
        self.env = gym.make(self.env_name)
        if self.record_video:
            self.env = wrappers.Monitor(self.env, os.path.join(self.folder, 'video'))
        self.env.reset()
    
    # ---------------------------------
    # Handling the q-table

    def __q_table_init(self):
        num_states = self.clustering.num_clusters
        num_actions = self.env.action_space.n
        self.q_table = np.random.rand(num_states, num_actions)

    def __q_table_get(self, state, action):
        '''
        state - index of state (cluster)
        action - index of action
        '''
        return self.q_table[state, action]

    def __q_table_set(self, state, action, value):
        '''
        state - index of state (cluster)
        action - index of action
        value - new value to put into the table
        '''
        self.q_table[state, action] = value
    
    def __q_table_max(self, state):
        return self.q_table[state].max()

    def __q_table_argmax(self, state):
        return self.q_table[state].argmax()

    # ---------------------------------
    # Target and behaviour policy
    
    def __policy(self, state):
        return self.__q_table_argmax(state)

    def __epsilon_greedy(self, state):
        # creating the list of actions to choose from
        n = self.env.action_space.n
        actions = [a for a in range(n)]
        # the probabilities associated with each entry in actions
        p = np.ones(n) * self.epsilon/(n-1)
        p[self.__policy(state)] = 1 - self.epsilon
        return np.random.choice(actions, p=p)
    
    # ---------------------------------
    # Train and evaluate
    
    def train(self):
        self.__env_init()
        self.__q_table_init()
        ev = E(self.env)
        alpha = 1.0
        prev_state = ev.observation

        def policy(state):
            x = self.clustering(state)
            return self.__epsilon_greedy(x)

        for cntr in range(self.max_iter):

            # take a step in the environment
            state, action, reward, done = ev.environment_step(policy)

            # update the q function
            value = (1-alpha) * self.__q_table_get(prev_state, action) + alpha * (reward + (1-done) * self.gamma * self.__q_table_max(state))
            self.__q_table_set(prev_state, action, value)

            # update the alpha and epsilon
            alpha = 1.0/cntr
            self.epsilon = max(0.05, self.epsilon - 0.05)

    def evaluate(self):
        
        env = E(gym.make(self.env_name))

        def policy(state):
            x = self.clustering(state)
            return self.__policy(x)

        return_per_episode = []
        utility = 0.0
        for _ in range(20):

            # take a step in the environment
            _, _, reward, done = env.environment_step(policy)
            utility += reward
            if done:
                return_per_episode.append(utility)
                utility = 0.0
        return return_per_episode
        