import os
import gym
from gym import wrappers
import numpy as np
from autoencoder import numpy2torch
from environment import Environment as E


class Q:

    def __init__(self, ae_model, clustering, env_name, record_video=False, folder=''):
        '''
        This class trains a Q-table. The input state is derived from the clustered
        encoded state.
        
        ae_model - the autoencoder the compress the state
        clustering - a function to determine which cluster the next processed input corresponds to
        env_name - name of the Atari environment
        record_video - record a video about the agent during evaluation
        folder - folder for saving videos, logs and resulting q-table
        '''
        self.ae_model = ae_model
        self.clustering = clustering
        self.clustering.kmeans
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
    
    def __q_table_save(self, file_name):
        np.save(file_name, self.q_table, allow_pickle=False)

    def __q_table_load(self, file_name):
        self.q_table = np.load(file_name, allow_pickle=False)

    # ---------------------------------
    # Target and behaviour policy
    
    def __policy(self, state):
        return self.__q_table_argmax(state)

    def __epsilon_greedy(self, state, epsilon):
        # creating the list of actions to choose from
        n = self.env.action_space.n
        actions = [a for a in range(n)]
        # the probabilities associated with each entry in actions
        p = np.ones(n) * epsilon/(n-1)
        p[self.__policy(state)] = 1 - epsilon
        return np.random.choice(actions, p=p)
    
    # ---------------------------------
    # Train and evaluate

    def __state2cluster(self, state):
        tensor = numpy2torch(state)
        latent = self.ae_model.calculate_feature(tensor).detach().numpy() # calculate the latent vector and convert tensor to numpy
        x = self.clustering.predict(latent)
        return x
    

    def train(self, params, callback=None):
        self.__env_init()
        self.__q_table_init()
        ev = E(self.env)
        epsilon = params['epsilon_0']
        gamma = params['gamma']
        alpha = 1.0
        prev_state = self.__state2cluster(ev.observation)

        print("Start training.")

        def policy(state):
            x = self.__state2cluster(state)
            return self.__epsilon_greedy(x, epsilon)

        for cntr in range(1, params['max_iter']):

            # take a step in the environment
            observation, action, reward, done = ev.environment_step(policy)
            state = self.__state2cluster(observation)

            # update the q function
            value = (1-alpha) * self.__q_table_get(prev_state, action) + alpha * (reward + (1-done) * gamma * self.__q_table_max(state))
            self.__q_table_set(prev_state, action, value)

            # update the alpha and epsilon
            alpha = 1.0/cntr
            epsilon = max(params['epsilon_min'], epsilon - params['epsilon_delta'])

            # save the current state
            prev_state = state
            
            # do evaluation if necessary
            if cntr % params['eval_freq'] == 0:
                print("Evaluation: %d%%"%int(cntr * 100 / params['max_iter']))
                return_per_episode = self.evaluate()
                self.__q_table_save(os.path.join(self.folder, 'q_table' + str(cntr)))
                if callback is not None:
                    callback(return_per_episode)


    def evaluate(self):
        
        env = E(gym.make(self.env_name))

        def policy(state):
            x = self.__state2cluster(state)
            return self.__policy(x)

        return_per_episode = []
        utility = 0.0
        for _ in range(20):
            done = False
            while not done:
                # take a step in the environment
                _, _, reward, done = env.environment_step(policy)
                utility += reward
                if done:
                    return_per_episode.append(utility)
                    utility = 0.0
        return return_per_episode
        