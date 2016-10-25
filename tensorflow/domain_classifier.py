import tensorflow as tf
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--word2vec', help='input file for pretrained vocabulary vectors by word2vec')
parser.add_argument('--train', help='input file for training data')
args = parser.parse_args()

voca = dict()
with open(args.word2vec, 'r') as f:
    voca = eval(f.read())
    
for word in voca.items():
    voca[word[0]] = np.frombuffer(word[1], dtype=np.float32)

domains = ['unknown']
train_data = []
test_data = []

with open(args.train, 'r') as f:
    for line in f:
        input = line.split()
        domains.append(input[0])
        with open(input[1], 'r') as train_file:
            train_data += [[d.strip().lower(), domains.index(input[0])] for d in train_file]

        with open(input[2], 'r') as test_file:
            test_data += [[d.strip().lower(), domains.index(input[0])] for d in test_file]
            
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    print('domain:%s' % domains)
    print('train_data:%d' % len(train_data))
    print('test_data:%d' % len(test_data))

"""    
    for domain in domains:
        train_count = 0
        test_count = 0
        for data in train_data:
            if data[1] == domains.index(domain):
                train_count += 1
        for data in test_data:
            if data[1] == domains.index(domain):
                test_count += 1
                
        print('%s:[train:%d, test:%d]' % (domain, train_count, test_count))
"""

label_count = len(domains)
lstm_size = 128
batch_size = 5000
test_size = len(test_data)
epoch_size = 5

batch = []
labels = np.zeros([batch_size, 1, label_count], dtype=np.int)

for i in range(batch_size):
    batch.append([voca[word] for word in train_data[i][0].split()])
    labels[i][0][train_data[i][1]] = 1

batch_for_test = []
labels_for_test = np.zeros([test_size, 1, label_count], dtype=np.int)

for i in range(test_size):
    batch_for_test.append([voca[word] for word in test_data[i][0].split()])
    labels_for_test[i][0][test_data[i][1]] = 1

longest_length = 0
for d in batch:
    if longest_length < len(d):
        longest_length = len(d)

for d in batch_for_test:
    if longest_length < len(d):
        longest_length = len(d)
        
print('longest:%d' % longest_length)

for d in batch:
    d.extend([np.zeros(128, dtype=np.float32) for _ in range(longest_length - len(d))])

for d in batch_for_test:
    d.extend([np.zeros(128, dtype=np.float32) for _ in range(longest_length - len(d))])
    
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, W, B, lstm_size, step_size):
    X_split = tf.split(0, step_size, X)

    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    outputs, _states = tf.nn.rnn(lstm, X_split, dtype=tf.float32)

    return tf.matmul(outputs[-1], W) + B, lstm.state_size # State size to initialize the stat

X = tf.placeholder("float", [None, 128])
Y = tf.placeholder("float", [None, label_count])

W = init_weights([lstm_size, label_count])
B = init_weights([label_count])

py_x, state_size = model(X, W, B, lstm_size, longest_length)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    print('initializing...')
    tf.initialize_all_variables().run()

    for i in range(epoch_size):
        print('epoch #%d' % i)
        for j in range(batch_size):
            sess.run(train_op, feed_dict={X: batch[j], Y: labels[j]})
            if (j % 1000) == 0 and j != 0:
                print('loop %d' % j)
    
        count = 0
        for j in range(test_size):
            result = sess.run(predict_op, feed_dict={X: batch_for_test[j]})
            correct = np.argmax(labels_for_test[j], axis=1)
            if result == correct:
                count += 1
            else:
                print('%s result:%s' % (test_data[j], result))
        print('score:%d/%d' % (count, test_size))
