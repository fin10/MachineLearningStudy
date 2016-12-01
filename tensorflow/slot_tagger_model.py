import tensorflow as tf
from dataset import Dataset


class SlotTaggerModel:
    def __init__(self, word_embedding_dimension: int, num_classes: int, max_length: int):
        self.__num_neurons = 300
        self.__num_layers = 1
        self.__dimension = word_embedding_dimension
        self.__num_classes = num_classes

        self.__x = tf.placeholder(tf.float32, [None, max_length, self.__dimension])
        self.__y = tf.placeholder(tf.float32, [None, max_length, self.__num_classes])
        self.__dropout = tf.placeholder(tf.float32)

        weight = tf.Variable(tf.truncated_normal([self.__num_neurons, self.__num_classes], stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
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
        prediction = tf.reshape(softmax, [-1, max_length, self.__num_classes])

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

        print('[SlotTaggerModel]')
        print('Neurons: %d' % self.__num_neurons)
        print('Multi layers: %d' % self.__num_layers)
        print('Word embedding dimension: %d' % self.__dimension)
        print('Slots: %d' % self.__num_classes)
        print('Learning rate: %d' % learning_rate)

    def train(self, train_data: Dataset, model=None):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            if model is None:
                sess.run(tf.initialize_all_variables())
            else:
                saver.restore(sess, model)

            cost_output = float('inf')
            for _ in range(100):
                train = train_data.sample(20)
                _, cost_output = sess.run([self.__optimizer, self.__cost],
                                          feed_dict={self.__x: train.vectors,
                                                     self.__y: train.target,
                                                     self.__dropout: 0.5})

            return cost_output, saver.save(sess, 'slot_tagger_model.ckpt')

    def test(self, test_data: Dataset, model: str, result_report=None):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, model)
            prediction_output, target_output, score_output, length_output = sess.run(
                [self.__prediction, self.__target, self.__score, self.__length],
                feed_dict={self.__x: test_data.vectors,
                           self.__y: test_data.target,
                           self.__dropout: 1.0})

            if result_report is not None:
                with open(result_report, 'w') as f:
                    f.write('score: %s\n' % score_output)
                    for i in range(test_data.length):
                        f.write('%s\n' % test_data.uttr[i])
                        f.write('%s\n' % test_data.iob[i])
                        f.write('%s\n' % target_output[i][:length_output[i]])
                        f.write('%s\n' % prediction_output[i][:length_output[i]])
            return score_output
