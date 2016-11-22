import argparse
import json
import pickle
import re

parser = argparse.ArgumentParser()
parser.add_argument('input', help='Input a pkl file')
args = parser.parse_args()


def invert_dict(dic):
    newDict = dict()
    for item in dic.items():
        newDict[item[1]] = item[0]
    return newDict


def parse_slots(labels):
    slots = set()
    pattern = re.compile('[ib]-(\S+)')
    for label in labels:
        found = re.findall(pattern, label.lower())
        if found:
            slots.add(found[0])
    return list(slots)


def parse_data(uttr, named_entity, iob):
    result = []
    for i in range(len(uttr)):
        item = dict()
        item['uttr'] = ' '.join(list(map(lambda x: idx2words[x], uttr[i]))).lower() \
            .replace('\'d', 'would').replace('\'s', 'is').replace('\'m', 'am').replace('\'ll', 'will') \
            .replace('\'re', 'are')
        item['named_entity'] = ' '.join(list(map(lambda x: idx2tables[x], named_entity[i]))).lower()
        item['iob'] = ' '.join(list(map(lambda x: idx2labels[x], iob[i]))).lower()
        item['length'] = len(uttr[i])
        result.append(item)
    return result


with open(args.input, 'rb') as f:
    train, test, dicts = pickle.load(f, encoding='latin1')

idx2words = invert_dict(dicts['words2idx'])
idx2labels = invert_dict(dicts['labels2idx'])
idx2tables = invert_dict(dicts['tables2idx'])

slots = parse_slots(list(idx2labels.values()))
print('slots: %d' % len(slots))

train_data = parse_data(train[0], train[1], train[2])
test_data = parse_data(test[0], test[1], test[2])
print('generated train data: %d, test data: %d' % (len(train_data), len(test_data)))

with open(args.input + '.slots', 'w') as output:
    output.write(json.dumps(slots))

with open(args.input + '.train', 'w') as output:
    output.write(json.dumps(train_data))

with open(args.input + '.test', 'w') as output:
    output.write(json.dumps(test_data))
