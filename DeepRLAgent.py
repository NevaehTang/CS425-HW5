# import neurolab as nl
import numpy as np
import operator
from random import shuffle
import pickle
import tensorflow as tf

class DictQLearningAgent(object):
    def __init__(self, action_space, learning_rate=0.1, discount=0.95, exploration_rate=0.5, exploration_decay_rate=0.99):
        self._q_table = dict()
        self._n_actions = action_space.n
        self._learning_rate = learning_rate
        self._discount = discount
        self._exploration_rate = exploration_rate
        self._exploration_decay = exploration_decay_rate
        self._updatesSum=0.0
        self._maxChange=0.0

    def reset(self):
        self._exploration_rate *= self._exploration_decay
        self._updatesSum=0.0
        self._maxChange=0.0

    def save(self, filename):
        # raise NotImplementedError('***Error: save to file  not implemented')
        # YOUR CODE HERE: save trained model to file
        pickle_out = open(filename, "wb")
        pickle.dump(self._q_table, pickle_out)
        pickle_out.close()

    def load(self, filename):
        # raise NotImplementedError('***Error: load from file not implemented')
        # YOUR CODE HERE: load trained model from file
        pickle_in = open(filename, "rb")
        self._q_table = pickle.load(pickle_in)


    def act(self, observation):
        if np.random.random_sample() < self._exploration_rate:
            return np.random.randint(0, self._n_actions)
        else:
            action_vals = [(self._q_table.get((observation, action), 0), action) for action in range(self._n_actions)]
            shuffle(action_vals)
            return max(action_vals, key=operator.itemgetter(0))[1]

    def update(self, observation, action, new_observation, reward):
        observation_action = (observation, action)
        val = max(self._q_table.get((new_observation, a), 0) for a in range(self._n_actions))
        old=self._q_table.get(observation_action, 0)
        self._q_table[observation_action] =\
            self._q_table.get(observation_action, 0) + self._learning_rate * \
                                                       (reward + self._discount * val -
                                                        self._q_table.get(observation_action, 0))
        delta=abs(self._q_table[observation_action]-old)
        self._updatesSum += delta
        if(delta>self._maxChange):
            self._maxChange=delta


class DeepRLAgent(object):
    def __init__(self, observation_space_dim, action_space,
                 learning_rate=0.1,
                 discount=0.99,
                 exploration_rate=0.5,
                 exploration_decay_rate=0.99,
                 batch_size=10):
        # Create train samples
        self.input_size=observation_space_dim
        self.output_size=action_space.n
        self._batch_size=batch_size

        #define and initialize your network here
        #UNCOMMENT THESE LINES TO TEST TENSORFLOW
        #self._sess = tf.Session()
        #self._discount = tf.constant(discount)
        #self._sess.run([tf.initialize_all_variables()])

    def save(self, filename):
        raise NotImplementedError('***Error: save to file  not implemented')
        #YOUR CODE HERE: save trained model to file


    def load(self, filename):
        raise NotImplementedError('***Error: load from file not implemented')
        #YOUR CODE HERE: load trained model from file

    def reset(self):
        raise NotImplementedError('***Error: load from file not implemented')
        # YOUR CODE HERE: load trained model from file

    def act(self, observation):
        raise NotImplementedError('***Error: load from file not implemented')
        # YOUR CODE HERE: pick actual best action

    def update(self, observation, action, new_observation, reward):
        raise NotImplementedError('***Error: load from file not implemented')
        # YOUR CODE HERE: pick actual best action
        # Note: you may need to change the function signature as needed by your training algorithm
