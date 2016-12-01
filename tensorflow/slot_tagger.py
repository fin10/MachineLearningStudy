import argparse
import json

import numpy as np

from dataset import Dataset
from slot_tagger_model import SlotTaggerModel
from vocabulary import Vocabulary


def read_dataset(file_path):
    with open(file_path, 'r') as f:
        config = json.loads(f.read())

    slots = ['padding', 'o']
    for slot in config['training_data']['slots']:
        slots.append('b-' + slot)
        slots.append('i-' + slot)

    with open(config['training_data']['train'], 'r') as train_file:
        train_data = [[sentence['uttr'].strip().lower(), sentence['iob'], sentence['length']]
                      for sentence in json.load(train_file)]
    with open(config['training_data']['test'], 'r') as test_file:
        test_data = [[sentence['uttr'].strip().lower(), sentence['iob'], sentence['length']]
                     for sentence in json.load(test_file)]

    num_valid_dataset = int(len(train_data) * 0.3)

    voca = Vocabulary(config['word_embedding'])
    train_dataset = generate_dataset(train_data[num_valid_dataset:], slots, voca)
    valid_dataset = generate_dataset(train_data[:num_valid_dataset], slots, voca)
    test_dataset = generate_dataset(test_data, slots, voca)

    max_length = 100
    train_dataset.extend(max_length)
    valid_dataset.extend(max_length)
    test_dataset.extend(max_length)

    return voca, slots, max_length, train_dataset, valid_dataset, test_dataset


def init_slot_vectors(voca, dataset):
    slots = dict()
    dummy = np.zeros(shape=voca.dimension(), dtype=np.float32)
    for data in dataset:
        words = data[0].split()
        labels = data[1].split()
        length = data[2]
        for i in range(length):
            if labels[i] is not 'o':
                if labels[i] not in slots:
                    slots[labels[i]] = []
                slots[labels[i]].append(np.concatenate([dummy if i - 1 < 0 else voca.get(words[i - 1]),
                                                        voca.get(words[i]),
                                                        dummy if i + 1 >= length else voca.get(words[i + 1])]))

    for item in slots.items():
        slots[item[0]] = np.mean(item[1], axis=0)

    return slots


def generate_dataset(items, slots, voca: Vocabulary):
    dataset = Dataset()
    for item in items:
        vectors = []
        for word in item[0].split():
            vectors.append(voca.get(word))

        labels = []
        for tag in item[1].split():
            value = np.zeros([len(slots)], dtype=np.float32)
            value[slots.index(tag)] = 1
            labels.append(value)

        dataset.add(item[0], item[1], vectors, labels)

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train', metavar='CONFIG', help='input a config file for train')
    args = parser.parse_args()

    voca, slots, max_length, train_data, valid_data, test_data = read_dataset(args.train)
    model = SlotTaggerModel(voca.dimension(), len(slots), max_length)
    num_epochs = 10
    print('Epochs: %d' % num_epochs)
    model_path = None
    try:
        for i in range(1, num_epochs + 1):
            print('epoch #%d Training...' % i)
            cost, model_path = model.train(train_data, model=model_path)
            accuracy = model.test(valid_data, model=model_path)
            print('accuracy: %s, cost: %s' % (accuracy, cost))
    except KeyboardInterrupt:
        print('\n[err] interrupt occurs')

    print('Testing...')
    accuracy = model.test(test_data, model=model_path, result_report='slot_tagger.result')
    print('final accuracy: %s' % accuracy)
