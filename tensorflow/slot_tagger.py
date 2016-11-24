import argparse

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument('--train', metavar='CONFIG', help='input a config file for train')
group.add_argument('--test', metavar='CONFIG', help='input a config file for test')
args = parser.parse_args()

if args.train is None and args.test is None:
    parser.print_help()
    exit()

import tensorflow as tf
import numpy as np
import json
import random
import functools


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


class Dataset:
    def __init__(self, data=None, target=None):
        self._data = data
        self._target = target

    @property
    def data(self):
        if self._data is None:
            self._data = []
        return self._data

    @property
    def target(self):
        if self._target is None:
            self._target = []
        return self._target

    @property
    def length(self):
        return len(self.data)

    def add(self, data, target):
        self.data.append(data)
        self.target.append(target)

    def sample(self, num):
        indexes = random.sample(range(num), num)
        new_data = list(map(lambda x: self.data[x], indexes))
        new_target = list(map(lambda x: self.target[x], indexes))
        return Dataset(data=new_data, target=new_target)


class SlotFillingModel:
    def __init__(self, data, target, weight, bias, num_neurons, num_layers, dropout):
        self._data = data
        self._target = target
        self._weight = weight
        self._bias = bias
        self._num_neurons = num_neurons
        self._num_layers = num_layers
        self._dropout = dropout
        self.prediction
        self.train_op
        self.test_op

    @staticmethod
    def length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        return tf.cast(length, tf.int32)

    @lazy_property
    def prediction(self):
        # Recurrent network.
        cell = tf.nn.rnn_cell.GRUCell(self._num_neurons)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self._dropout)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self._num_layers)
        output, _ = tf.nn.dynamic_rnn(cell,
                                      self._data,
                                      sequence_length=self.length(self._data),
                                      dtype=tf.float32)

        # Softmax layer.
        num_cell = int(self._target.get_shape()[1])
        num_classes = int(self._target.get_shape()[2])

        # Flatten to apply same weights to all time steps.
        output = tf.reshape(output, [-1, self._num_neurons])
        prediction = tf.nn.softmax(tf.matmul(output, self._weight) + self._bias)
        prediction = tf.reshape(prediction, [-1, num_cell, num_classes])
        return prediction

    @lazy_property
    def train_op(self):
        cross_entropy = -tf.reduce_sum(self._target * tf.log(self.prediction), [1, 2])
        cross_entropy = tf.reduce_mean(cross_entropy)

        learning_rate = 0.003
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        train_op = optimizer.minimize(cross_entropy)
        return train_op

    @lazy_property
    def test_op(self):
        test_op = tf.not_equal(tf.argmax(self._target, 2), tf.argmax(self.prediction, 2))
        return tf.reduce_mean(tf.cast(test_op, tf.float32))


def read_config(file_path):
    with open(file_path, 'r') as f:
        config = json.loads(f.read())
    return config['word_embedding'], config['training_data'], config['epoch']


def init_word_embedding(path):
    dimension = -1
    with open(path, 'r') as f:
        voca = eval(f.read())
        for word in voca.items():
            voca[word[0]] = np.frombuffer(word[1], dtype=np.float32)
            if dimension == -1:
                dimension = len(voca[word[0]])

    return voca, dimension


def transform_dataset(items, slots, voca):
    dataset = Dataset()
    missings = []
    num_slot = len(slots)

    unk = np.zeros([embedding_dimension], dtype=np.float32)
    unk.fill(-1)

    for item in items:
        vectors = []
        for word in item[0].split():
            if word in voca:
                vectors.append(voca[word])
            else:
                vectors.append(unk)
                missings.append(word)

        labels = []
        for tag in item[1].split():
            value = np.zeros([num_slot], dtype=np.float32)
            value[slots.index(tag)] = 1
            labels.append(value)

        if len(vectors) is not len(labels):
            raise Exception('not matching: %s' % item)

        dataset.add(vectors, labels)

    return dataset, missings


def read_dataset(info, voca):
    slots = ['o']
    for slot in info['slots']:
        slots.append('b-' + slot)
        slots.append('i-' + slot)

    with open(info['train'], 'r') as train_file:
        train_data = [[sentence['uttr'].strip().lower(), sentence['iob'], sentence['length']]
                      for sentence in json.load(train_file)]
    with open(info['test'], 'r') as test_file:
        test_data = [[sentence['uttr'].strip().lower(), sentence['iob'], sentence['length']]
                     for sentence in json.load(test_file)]

    train_data, _ = transform_dataset(train_data, slots, voca)
    test_data, _ = transform_dataset(test_data, slots, voca)

    return slots, train_data, test_data


word_embedding_info, dataset_info, num_epochs = read_config(args.train)
print('-- Config --')
print('Word embedding file path: %s' % word_embedding_info)

voca, embedding_dimension = init_word_embedding(word_embedding_info)
slots, train_data, test_data = read_dataset(dataset_info, voca)

num_neurons = 128
num_layers = 3
num_classes = len(slots)
num_train = train_data.length
num_test = test_data.length
max_length = 100

for i in range(len(train_data.data)):
    train_data.data[i] = np.pad(train_data.data[i], [[0, max_length - len(train_data.data[i])], [0, 0]], 'constant')
    train_data.target[i] = np.pad(train_data.target[i], [[0, max_length - len(train_data.target[i])], [0, 0]],
                                  'constant')

for i in range(len(test_data.data)):
    test_data.data[i] = np.pad(test_data.data[i], [[0, max_length - len(test_data.data[i])], [0, 0]], 'constant')
    test_data.target[i] = np.pad(test_data.target[i], [[0, max_length - len(test_data.target[i])], [0, 0]], 'constant')

print('-- Environment --')
print('Word embedding dimension: %d' % embedding_dimension)
print('slots: %s' % len(slots))
print('train: %d, test: %d' % (num_train, num_test))
print('Neurons: %d' % num_neurons)
print('Epochs: %d' % num_epochs)

print('-- Training --')
print('initializing...')
data = tf.placeholder(tf.float32, [None, max_length, embedding_dimension])
target = tf.placeholder(tf.float32, [None, max_length, num_classes])
dropout = tf.placeholder(tf.float32)
weight = tf.Variable(tf.truncated_normal([num_neurons, num_classes], stddev=0.01))
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
model = SlotFillingModel(data, target, weight, bias, num_neurons, num_layers, dropout)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(num_epochs):
        epoch += 1
        print('epoch #%d' % epoch)
        print('training...')
        for i in range(500):
            train = train_data.sample(10)
            sess.run(model.train_op, feed_dict={data: train.data, target: train.target, dropout: 0.5})
            if i % 100 is 99:
                print('\tbatch %d' % (i + 1))

        print('testing...')
        score = sess.run(model.test_op, feed_dict={data: test_data.data, target: test_data.target, dropout: 1.0})
        print('\t%s' % score)
