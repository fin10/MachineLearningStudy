import argparse
import json
import random
import re

parser = argparse.ArgumentParser()
parser.add_argument('input', help='Input a grammar file')
parser.add_argument('num_train', type=int, help='Number of training data')
parser.add_argument('num_test', type=int, help='Number of test data')
args = parser.parse_args()


def alternate(text, words, output):
    try:
        word = words.pop(0)
        ws = word[1].split('|')
        for w in ws:
            alternate(text.replace(word[0], w, 1), words[:], output)
    except IndexError:
        output.append(text)
        return


def replace(text, entities, output):
    try:
        entity = entities.pop(0)
        slot_file = open('./' + entity[1], 'r')
        for word in slot_file:
            word = word.strip().lower()
            if len(word) > 0:
                replace(text.replace(entity[0], word, 1), entities[:], output)
    except IndexError:
        output.append(text)
        return
    except OSError as e:
        print(e)
        return


def extract_iob(input):
    uttr = []
    iob = []
    pattern = re.compile('(\S+)/([b|i-]\S+)')
    for word in input.split():
        found = re.findall(pattern, word)
        if not found:
            uttr.append(word)
            iob.append('o')
        else:
            uttr.append(found[0][0])
            iob.append(found[0][1])

    return ' '.join(uttr), ' '.join(iob), len(uttr)


with open(args.input, 'r') as input:
    replacePattern = re.compile('(<(\S+)>)')
    alternatePattern = re.compile('(\(([^)]+)\))')
    slotPattern = re.compile('(\{([^\}]+)\}\[(\S+)\])')

    result = []
    for line in input:
        utterance = line.strip().lower()
        words = re.findall(alternatePattern, utterance)
        alternate(utterance, words, result)

    result2 = []
    for item in result:
        entities = re.findall(replacePattern, item)
        replace(item, entities, result2)

    random.shuffle(result2)
    result2 = result2[:args.num_train + args.num_test]

    result3 = []
    for item in result2:
        slots = re.findall(slotPattern, item)
        for slot in slots:
            words = slot[1]
            name = slot[2]
            word = words.split(' ')
            ret = ' '.join(['{0}/{1}-{2}'.format(word[i], i == 0 and 'b' or 'i', name) for i in range(len(word))])
            item = item.replace(slot[0], ret)
        result3.append(item)

    result4 = []
    for item in result3:
        data = dict()
        data['raw'] = item
        data['uttr'], data['iob'], data['length'] = extract_iob(item)
        result4.append(data)

    train = result4[:args.num_train]
    test = result4[args.num_train:]

    with open(args.input + '.train', 'w') as output:
        output.write(json.dumps(train))

    with open(args.input + '.test', 'w') as output:
        output.write(json.dumps(test))

print('%s\nUtterance generated: %d for training, %d for test' % (args.input, len(train), len(test)))
