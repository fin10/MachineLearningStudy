import argparse
import os
import sys
import time

import tensorflow as tf

from dataset import Dataset
from slot_tagger_model import SlotTaggerModel
from vocabulary import Vocabulary


def read_file(path: str):
    with open(path, 'r', encoding='utf-8') as file:
        result = [x.strip() for x in file]

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode')
    parser.add_argument('--slot')
    parser.add_argument('dir')
    args = parser.parse_args()

    slots = []
    files = os.listdir(args.slot)
    for file in files:
        if file.endswith('.slot'):
            with open(os.path.join(args.slot, file), 'r', encoding='utf-8') as f:
                slots += [slot.strip() for slot in f if len(slot.strip()) > 0]

    Dataset.init_slot(set(slots))
    vocab = Vocabulary('./word2vec.embeddings')

    with tf.Session() as sess:
        model = SlotTaggerModel(sess, vocab, Dataset.get_slots())

        if args.mode == 'train':
            train = []
            dev = []
            files = os.listdir(args.dir)
            for file in files:
                if file.endswith('.train'):
                    train += read_file(os.path.join(args.dir, file))
                elif file.endswith('.dev'):
                    dev += read_file(os.path.join(args.dir, file))

            train_dataset = Dataset.create_dataset(train)
            dev_dataset = Dataset.create_dataset(dev)
            print('Data size (train: %d, dev: %d)' % (train_dataset.length(), dev_dataset.length()))

            total_time = time.time()
            while True:
                print('Training...')
                start_time = time.time()
                cost = model.train(sess, train_dataset)
                accuracy, _ = model.test(sess, dev_dataset)
                print('global step %d, accuracy %s, cost %s, duration: %s' % (
                model.get_global_step(), accuracy, cost, round(time.time() - start_time, 2)))
                sys.stdout.flush()

        elif args.mode == 'test':
            test = []
            files = os.listdir(args.dir)
            for file in files:
                if file.endswith('.test'):
                    test += read_file(os.path.join(args.dir, file))

            test_dataset = Dataset.create_dataset(test)
            print('Data size (test: %d)' % test_dataset.length())

            print('Testing...')
            accuracy, predict = model.test(sess, test_dataset)
            print('Total accuracy: %s' % accuracy)

            slots = {}
            for i in range(test_dataset.length()):
                expected = test_dataset.get_iob(i)
                actual = predict[i]

                for j in range(len(expected)):
                    if expected[j] not in slots:
                        slots[expected[j]] = {'correct': 0, 'total': 0}

                    slots[expected[j]]['total'] += 1
                    if expected[j] == actual[j]:
                        slots[expected[j]]['correct'] += 1

            if not os.path.exists('./out'):
                os.mkdir('./out')

            domains = test_dataset.get_domains()
            for domain in domains:
                with open('./out/{}_report.txt'.format(domain), 'w', encoding='utf-8') as file:
                    indexes = [i for i in range(test_dataset.length()) if test_dataset.get_domain(i) == domain]
                    for index in indexes:
                        if test_dataset.get_iob(index) != predict[index]:
                            file.write('{}\n'.format(' '.join(test_dataset.get_tokens(index))))
                            file.write('expected : {}\n'.format(test_dataset.get_iob(index)))
                            file.write('actual   : {}\n'.format(predict[index]))

            with open('./out/slot_report.txt', 'w', encoding='utf-8') as output:
                for key in slots.keys():
                    slot = slots[key]
                    output.write('%s -> %.2f (%d/%d)\n' % (
                    key, 100 * slot['correct'] / slot['total'], slot['correct'], slot['total']))
        else:
            raise NotImplementedError('Not implemented yet.')
