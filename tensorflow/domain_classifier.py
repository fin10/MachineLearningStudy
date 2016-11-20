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
import random
import json


def parse_config(file_path):
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


def init_training_data(data):
    domains = []
    train_data = []
    test_data = []
    for domain in data:
        domains.append(domain['domain'])
        with open(domain['train'], 'r') as train_file:
            train = [[sentence['uttr'].strip().lower(), domains.index(domain['domain'])] for sentence in
                     json.load(train_file)]
        with open(domain['test'], 'r') as test_file:
            test = [[sentence['uttr'].strip().lower(), domains.index(domain['domain'])] for sentence in
                    json.load(test_file)]
        print('[%s] train: %d, test: %d' % (domain['domain'], len(train), len(test)))
        print('[%s] train: %d, test: %d' % (domain['domain'], len(train), len(test)))
        train_data += train
        test_data += test
    random.shuffle(train_data)
    random.shuffle(test_data)

    return domains, train_data, test_data


word_embedding_info, domain_infos = parse_config(args.train)
print('-- Config --')
print('Word embedding file path: %s' % word_embedding_info)
print('Domains: %d' % len(domain_infos))

print('-- Environment --')
voca, embedding_dimension = init_word_embedding(word_embedding_info)
print('Word embedding dimension: %d' % embedding_dimension)
domains, train_data, test_data = init_training_data(domain_infos)

LSTM_SIZE = 128
BATCH_SIZE = 3000
EPOCH_SIZE = 2
label_count = len(domains)
test_count = len(test_data)

UNK = np.zeros([embedding_dimension], dtype=np.float32)
UNK.fill(-1)


def init_batch(data, batch_size, label_count):
    batch = []
    labels = np.zeros([batch_size, label_count], dtype=np.int)

    for i in range(batch_size):
        sentence = []
        for word in data[i][0].split():
            if word in voca:
                sentence.append(voca[word])
            else:
                sentence.append(UNK)
        batch.append(sentence)
        labels[i][data[i][1]] = 1

    return batch, labels


def find_longest_length(items):
    longest_length = 0
    for item in items:
        length = len(item)
        if longest_length < length:
            longest_length = length

    return longest_length


def extend(items, size):
    for item in items:
        item.extend([np.zeros(embedding_dimension, dtype=np.float32) for _ in range(size - len(item))])


batch_for_train, labels_for_train = init_batch(train_data, BATCH_SIZE, label_count)
batch_for_test, labels_for_test = init_batch(test_data, test_count, label_count)
longest_length = find_longest_length(batch_for_train + batch_for_test)
print('longest length:%d' % longest_length)

extend(batch_for_train, longest_length)
extend(batch_for_test, longest_length)


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, W, B, lstm_size, step_size):
    X_split = tf.split(0, step_size, X)

    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    outputs, _states = tf.nn.rnn(lstm, X_split, dtype=tf.float32)

    return tf.matmul(outputs[-1], W) + B, lstm.state_size


X = tf.placeholder("float", [None, embedding_dimension])
Y = tf.placeholder("float", [label_count])

W = init_weights([LSTM_SIZE, label_count])
B = init_weights([label_count])

py_x, state_size = model(X, W, B, LSTM_SIZE, longest_length)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

saver = tf.train.Saver()

with tf.Session() as sess:
    print('-- Training --')
    print('initializing...')
    tf.initialize_all_variables().run()

    failed_list = []
    for epoch in range(EPOCH_SIZE):
        print('epoch #%d' % (epoch + 1))
        print('training...')
        for i in range(len(batch_for_train)):
            sess.run(train_op, feed_dict={X: batch_for_train[i], Y: labels_for_train[i]})
            if (i % 1000) == 999:
                print('  loop %d' % (i + 1))

        model_path = saver.save(sess, './domain_classifier_model.ckpt', global_step=(epoch + 1))
        print('  model saved in file: %s' % model_path)

        print('testing...')
        failed_list.clear()
        failed_list.append('epoch #%d' % (epoch + 1))
        count = 0
        for i in range(len(batch_for_test)):
            result = sess.run(predict_op, feed_dict={X: batch_for_test[i]})
            correct = np.argmax(labels_for_test[i], axis=0)
            if result == correct:
                count += 1
            else:
                failed_list.append('matching_failed: %s %s' % (test_data[i], result))
        print('  score: %s (%d/%d)' % ('{:.2%}'.format(count / len(batch_for_test)), count, len(batch_for_test)))

    with open('matching_failed.out', 'w') as f:
        f.write('\n'.join(failed_list))
