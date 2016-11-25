import argparse
import json
import random

import numpy as np

import tensorflow as tf

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument('--train', metavar='CONFIG', help='input a config file for train')
group.add_argument('--test', metavar='CONFIG', help='input a config file for test')
args = parser.parse_args()

if args.train is None and args.test is None:
    parser.print_help()
    exit()


class Dataset:
    def __init__(self, uttr=None, iob=None, data=None, target=None):
        self._uttr = uttr
        self._iob = iob
        self._data = data
        self._target = target

    @property
    def uttr(self):
        if self._uttr is None:
            self._uttr = []
        return self._uttr

    @property
    def iob(self):
        if self._iob is None:
            self._iob = []
        return self._iob

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

    def add(self, uttr, iob, data, target):
        self.uttr.append(uttr)
        self.iob.append(iob)
        self.data.append(data)
        self.target.append(target)

    def sample(self, num):
        indexes = random.sample(range(num), num)
        new_uttr = list(map(lambda x: self.uttr[x], indexes))
        new_iob = list(map(lambda x: self.iob[x], indexes))
        new_data = list(map(lambda x: self.data[x], indexes))
        new_target = list(map(lambda x: self.target[x], indexes))
        return Dataset(uttr=new_uttr, iob=new_iob, data=new_data, target=new_target)

    def extend(self, size):
        for i in range(self.length):
            self.data[i] = np.pad(self.data[i], [[0, size - len(self.data[i])], [0, 0]], 'constant')
            self.target[i] = np.pad(self.target[i], [[0, size - len(self.target[i])], [0, 0]], 'constant')


def read_config(file_path):
    with open(file_path, 'r') as f:
        config = json.loads(f.read())
    return config['word_embedding'], config['training_data']


def init_word_embedding(path):
    dimension = -1
    with open(path, 'r') as f:
        voca = eval(f.read())
        for word in voca.items():
            voca[word[0]] = np.frombuffer(word[1], dtype=np.float32)
            if dimension == -1:
                dimension = len(voca[word[0]])

    return voca, dimension


def generate_dataset(items, slots, voca):
    dataset = Dataset()
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

        labels = []
        for tag in item[1].split():
            value = np.zeros([num_slot], dtype=np.float32)
            value[slots.index(tag)] = 1
            labels.append(value)

        if len(vectors) is not len(labels):
            raise Exception('not matching: %s' % item)

        dataset.add(item[0], item[1], vectors, labels)

    return dataset


def read_dataset(info, voca):
    slots = ['padding', 'o']
    for slot in info['slots']:
        slots.append('b-' + slot)
        slots.append('i-' + slot)

    with open(info['train'], 'r') as train_file:
        train_data = [[sentence['uttr'].strip().lower(), sentence['iob'], sentence['length']]
                      for sentence in json.load(train_file)]
    with open(info['test'], 'r') as test_file:
        test_data = [[sentence['uttr'].strip().lower(), sentence['iob'], sentence['length']]
                     for sentence in json.load(test_file)]

    train_data = generate_dataset(train_data, slots, voca)
    test_data = generate_dataset(test_data, slots, voca)

    return slots, train_data, test_data


word_embedding_info, dataset_info = read_config(args.train)
voca, embedding_dimension = init_word_embedding(word_embedding_info)
slots, train_data, test_data = read_dataset(dataset_info, voca)

num_classes = len(slots)
num_train = train_data.length
num_test = test_data.length
num_neurons = 128
num_layers = 3
num_epochs = 1
num_batch = 1000
max_length = 100

print('-- Environment --')
print('Word embedding dimension: %d' % embedding_dimension)
print('slots: %s' % num_classes)
print('train: %d, test: %d' % (num_train, num_test))
print('Neurons: %d' % num_neurons)
print('Epochs: %d' % num_epochs)
print('Batches: %d' % num_batch)

train_data.extend(max_length)
test_data.extend(max_length)

x = tf.placeholder(tf.float32, [None, max_length, embedding_dimension])
y = tf.placeholder(tf.float32, [None, max_length, num_classes])
dropout = tf.placeholder(tf.float32)
weight = tf.Variable(tf.truncated_normal([num_neurons * 2, num_classes], stddev=0.01))
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

mask = tf.sign(tf.reduce_max(tf.abs(y), reduction_indices=2))
length = tf.cast(tf.reduce_sum(mask, reduction_indices=1), tf.int32)

# Recurrent network.
cell = tf.nn.rnn_cell.GRUCell(num_neurons)
cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell,
                                             cell,
                                             x,
                                             sequence_length=length,
                                             dtype=tf.float32)
output = tf.concat(2, outputs)

# Flatting
output = tf.reshape(output, [-1, num_neurons * 2])
softmax = tf.nn.softmax(tf.matmul(output, weight) + bias)
prediction = tf.reshape(softmax, [-1, max_length, num_classes])

cross_entropy = -tf.reduce_sum(y * tf.log(prediction), reduction_indices=2) * mask
cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1) / tf.cast(length, tf.float32)
cost = tf.reduce_mean(cross_entropy)

learning_rate = 0.003
train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

prediction = tf.argmax(prediction, 2)
target = tf.argmax(y, 2)

correct = tf.cast(tf.equal(prediction, target), tf.float32) * mask
correct = tf.reduce_sum(correct, reduction_indices=1) / tf.cast(length, tf.float32)
score = tf.reduce_mean(correct)

with tf.Session() as sess:
    print('-- Training --')
    print('initializing...')
    sess.run(tf.initialize_all_variables())
    for epoch in range(num_epochs):
        epoch += 1
        print('epoch #%d' % epoch)
        print('training...')
        for i in range(num_batch):
            train = train_data.sample(10)
            _, cost_output = sess.run([train_op, cost],
                                      feed_dict={x: train.data,
                                                 y: train.target,
                                                 dropout: 0.5})
            if i % 100 is 99:
                print('\tbatch %d: cost(%s)' % ((i + 1), cost_output))

        print('testing...')
        prediction_output, target_output, score_output, length_output = sess.run(
            [prediction, target, score, length],
            feed_dict={x: test_data.data,
                       y: test_data.target,
                       dropout: 1.0})
        print('\tscore: %s' % score_output)

        with open('slot_tagger.result', 'w') as f:
            f.write('epoch #%d, score: %s\n' % (epoch, score_output))
            for i in range(num_test):
                f.write('%s\n' % test_data.uttr[i])
                f.write('%s\n' % test_data.iob[i])
                f.write('%s\n' % target_output[i][:length_output[i]])
                f.write('%s\n' % prediction_output[i][:length_output[i]])
