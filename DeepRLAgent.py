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
        self.actions = action_space
        self.output_size=action_space.n
        self._batch_size=batch_size
        self.learning_rate = learning_rate

        #define and initialize your network here
        #UNCOMMENT THESE LINES TO TEST TENSORFLOW
        #self._sess = tf.Session()
        #self._discount = tf.constant(discount)
        #self._sess.run([tf.initialize_all_variables()])
        with tf.variable_scope(name):
            self.input = tf.placeholder(tf.float32,[None, *self.input_size], name = "input")
            self.action = tf.placeholder(tf.float32,[None, self.output_size], name = "action")
            self.Qtarget = tf.placeholder(tf.float32,[None], name = "target")

    #         first convnet
            self.conv1 = tf.layers.conv2d(inputs = self.input, filters = 32, kernel_size = [8,8], strides = [4,4],
                                          padding = "VALID", kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = "conv1")
            self.conv1batch = tf.layers.batch_normalization(self.conv1, training = True, epsilon = 0.00001, name = "conv1batch")
            self.conv1out = tf.nn.elu(self.conv1batch, name = "conv1out")

    #         second convnet
            self.conv2 = tf.layers.conv2d(inputs = self.conv1out, filters = 64, kernel_size = [4,4], strides = [2,2],
                                          padding = "VALID", kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = "conv2")
            self.conv2batch = tf.layers.batch_normalization(self.conv2, training = True, epsilon = 0.00001, name = "conv2batch")
            self.conv2out = tf.nn.elu(self.conv2batch, name = "conv2out")
    #         thrid convnet
            self.conv3 = tf.layers.conv2d(inputs = self.conv2out, filters = 128, kernel_size = [4,4], strides = [2,2],
                                          padding = "VALID", kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(), name = "conv3")
            self.conv3batch = tf.layers.batch_normalization(self.conv3, training = True, epsilon = 0.00001, name = "conv3batch")
            self.conv3out = tf.nn.elu(self.conv3batch, name = "conv3out")

            self.flatten = tf.layers.flatten(self.conv3out)

            self.fc = tf.layers.dense(input = self.flatten, units = 512, activation = tf.nn.elu, kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                      name = "fcl")
            self.output = tf.layers.dense(inputs = self.fc, kernel_initializer = tf.contrib.layers.xavier_initializer(), units = self.output_size, activation = None)

            self.Qpredict = tf.reduce_sum(tf.multiply(self.output,self.action), axis = 1)

            self.loss = tf.reduce_mean(tf.square(self.Qtarget-self.Qpredict))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)




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
        # raise NotImplementedError('***Error: load from file not implemented')
        # YOUR CODE HERE: pick actual best action

        if np.random.random_sample() < self._exploration_rate:
            return np.random.randint(0, self._n_actions)
        else:
            Q = sess.run(self.output, feed_dict = {self.output: observation.reshape((1, *observation.shape))} )
            actionIndex = np.argmax(Q)
            action = self.ations[int(actionIndex)]
            return action

    def update(self, observation, action, new_observation, reward):
        raise NotImplementedError('***Error: load from file not implemented')
        # YOUR CODE HERE: pick actual best action
        # Note: you may need to change the function signature as needed by your training algorithm
