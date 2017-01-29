import tensorflow as tf


class SlotTaggerTest(tf.test.TestCase):
    def testTrain(self):
        with self.test_session():
            x = tf.constant(1)
            self.assertEqual(x.eval(), 1)


if __name__ == '__main__':
    tf.test.main()
