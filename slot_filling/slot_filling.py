import argparse
import json
import time

from dataset import Dataset
from slot_tagger_model import SlotTaggerModel
from vocabulary import Vocabulary


def read_dataset(file_path: str):
    with open(file_path, 'r') as f:
        config = json.loads(f.read())
        train, valid, test = Dataset.create_dataset(config['dataset'])
        return train, valid, test, Vocabulary(config['word_embedding'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train', metavar='CONFIG', help='input a config file for train')
    args = parser.parse_args()

    train_dataset, valid_dataset, test_dataset, vocab = read_dataset(args.train)

    model = SlotTaggerModel(vocab, Dataset.get_slots())
    num_epochs = 10
    print('Epochs: %d' % num_epochs)
    model_path = None
    total_time = time.time()
    try:
        for i in range(1, num_epochs + 1):
            print('epoch #%d Training...' % i)
            start_time = time.time()
            cost, model_path = model.train(train_dataset, model=model_path)
            accuracy = model.test(valid_dataset, model=model_path)
            print('accuracy: %s, cost: %s, duration: %s' % (accuracy, cost, round(time.time() - start_time, 2)))
    except KeyboardInterrupt:
        print('\ninterrupt occurs')

    print('Testing...')
    accuracy = model.test(test_dataset, model=model_path, result_report='slot_tagger_result.output')
    print('final accuracy: %s, duration: %s' % (accuracy, round(time.time() - total_time, 2)))
