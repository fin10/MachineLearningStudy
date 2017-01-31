import argparse
import os
import random


def write_output(name: str, extension: str, subset):
    if not os.path.exists('./output'):
        os.mkdir('./output')

    filename = name + '.' + extension
    with open('./output/' + filename, 'w', encoding='utf-8') as output:
        output.write('\n'.join(subset))
        print('[%s] %d is generated.' % (filename, len(subset)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    parser.add_argument('input')
    parser.add_argument('count', nargs='?', default=-1, type=int)
    args = parser.parse_args()

    items = []
    with open(args.input, 'r', encoding='utf-8') as input:
        for line in input:
            item = line.strip()
            if len(item) is not 0:
                items.append(item)

    random.shuffle(items)
    print('%d exists.' % len(items))

    items = items[:int(args.count)]

    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.2
    TEST_RATIO = 0.1

    train_index = int(len(items) * TRAIN_RATIO)
    valid_index = int(len(items) * VALID_RATIO) + train_index
    test_index = int(len(items) * TEST_RATIO) + valid_index

    write_output(args.output, 'train', items[:train_index])
    write_output(args.output, 'valid', items[train_index:valid_index])
    write_output(args.output, 'test', items[valid_index:test_index])
