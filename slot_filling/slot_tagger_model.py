import os
import sys

import numpy as np
import tensorflow as tf

from dataset import Dataset
from vocabulary import Vocabulary


class SlotTaggerModel:
    MAX_LENGTH = 100

    def __init__(self, session, vocab: Vocabulary, slots: list):
        self.__step = 0
        self.__num_neurons = 300
        self.__num_layers = 1
        self.__batch_size = 64
        self.__step_per_checkpoints = 100
        self.__vocab = vocab
        self.__slots = slots
        num_slots = len(self.__slots)

        self.__x = tf.placeholder(tf.float32, [None, SlotTaggerModel.MAX_LENGTH, self.__vocab.get_dimension()])
        self.__y = tf.placeholder(tf.float32, [None, SlotTaggerModel.MAX_LENGTH, num_slots])
        self.__dropout = tf.placeholder(tf.float32)

        weight = tf.Variable(tf.truncated_normal([self.__num_neurons, num_slots], stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[num_slots]))
        mask = tf.sign(tf.reduce_max(tf.abs(self.__y), reduction_indices=2))
        self.__length = tf.cast(tf.reduce_sum(mask, reduction_indices=1), tf.int32)

        # Recurrent network.
        cell = tf.nn.rnn_cell.GRUCell(self.__num_neurons)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.__dropout)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.__num_layers)
        outputs, _ = tf.nn.dynamic_rnn(cell,
                                       self.__x,
                                       sequence_length=self.__length,
                                       dtype=tf.float32)
        # Flatting
        output = tf.reshape(outputs, [-1, self.__num_neurons])
        softmax = tf.nn.softmax(tf.matmul(output, weight) + bias)
        prediction = tf.reshape(softmax, [-1, SlotTaggerModel.MAX_LENGTH, num_slots])

        cross_entropy = -tf.reduce_sum(self.__y * tf.log(prediction), reduction_indices=2) * mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1) / tf.cast(self.__length, tf.float32)
        self.__cost = tf.reduce_mean(cross_entropy)

        learning_rate = 0.01
        self.__optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.__cost)

        self.__prediction = tf.argmax(prediction, 2)
        self.__target = tf.argmax(self.__y, 2)

        correct = tf.cast(tf.equal(self.__prediction, self.__target), tf.float32) * mask
        correct = tf.reduce_sum(correct, reduction_indices=1) / tf.cast(self.__length, tf.float32)
        self.__score = tf.reduce_mean(correct)
        self.__saver = tf.train.Saver(tf.global_variables())

        print('[SlotTaggerModel]')
        print('Neurons: %d' % self.__num_neurons)
        print('Multi layers: %d' % self.__num_layers)
        print('Word embedding dimension: %d' % self.__vocab.get_dimension())
        print('Slots: %d' % num_slots)
        print('Learning rate: %f' % learning_rate)

        if not os.path.exists('./model'):
            os.mkdir('./model')

        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.__saver.restore(session, ckpt.model_checkpoint_path)

        else:
            print("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())

        print()
        sys.stdout.flush()

    def train(self, session, dataset: Dataset):
        word_vectors = []
        for tokens in dataset.get_tokens():
            word_vectors.append(self.encode_word_vector(tokens))

        slot_vectors = []
        for iob in dataset.get_iob():
            slot_vectors.append(self.encode_slot_vector(iob))

        cost_output = float('inf')
        for _ in range(self.__step_per_checkpoints):
            indexes = np.random.choice(len(word_vectors), self.__batch_size, replace=False)
            x = [word_vectors[index] for index in indexes]
            y = [slot_vectors[index] for index in indexes]

            self.__step += 1
            _, cost_output = session.run([self.__optimizer, self.__cost],
                                         feed_dict={self.__x: x,
                                                    self.__y: y,
                                                    self.__dropout: 0.5})

        checkpoint_path = os.path.join('./model', "slot_filling_model.ckpt")
        self.__saver.save(session, checkpoint_path, global_step=self.__step)

        return cost_output

    def test(self, session, dataset: Dataset):
        word_vectors = []
        for tokens in dataset.get_tokens():
            word_vectors.append(self.encode_word_vector(tokens))

        slot_vectors = []
        for iob in dataset.get_iob():
            slot_vectors.append(self.encode_slot_vector(iob))

        prediction_output, score_output, length_output = session.run(
            [self.__prediction, self.__score, self.__length],
            feed_dict={self.__x: word_vectors,
                       self.__y: slot_vectors,
                       self.__dropout: 1.0})

        predict = []
        for i in range(dataset.length()):
            predict.append(self.decode_slot_vector(prediction_output[i][:length_output[i]]))

        return score_output, predict

    def encode_word_vector(self, tokens: list):
        result = []
        for token in tokens:
            result.append(self.__vocab.get(token))

        if len(result) < self.MAX_LENGTH:
            for _ in range(self.MAX_LENGTH - len(result)):
                result.append(self.__vocab.get_zeros())
        elif len(result) < self.MAX_LENGTH:
            raise OverflowError("Length should be smaller than %d. - %s" % (self.MAX_LENGTH, tokens))

        return result

    def encode_slot_vector(self, iobs: list):
        result = []
        for iob in iobs:
            result.append(self.__slots.index(iob))

        if len(result) < self.MAX_LENGTH:
            for _ in range(self.MAX_LENGTH - len(result)):
                result.append(-1)
        elif len(result) < self.MAX_LENGTH:
            raise OverflowError("Length should be smaller than %d. - %s" % (self.MAX_LENGTH, iobs))

        vectors = np.zeros([len(result), len(self.__slots)], dtype=np.int32)
        for i in range(len(result)):
            if result[i] >= 0:
                vectors[i][result[i]] = 1

        return vectors

    def decode_slot_vector(self, vector: list):
        result = []
        for v in vector:
            result.append(self.__slots[v])

        return result

    def get_global_step(self):
        return self.__step
