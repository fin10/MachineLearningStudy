import argparse
import os
import random
import re

from tools.data_splitter import DataSplitter


class GrammarGenerator:
    __alternate_pattern = re.compile('(\{([^}]+)\})')
    __replace_pattern = re.compile('(<(\S+)>)')

    def __init__(self, path: str):
        self.__vocab = {}
        for file in os.listdir(path):
            name = os.path.splitext(file)
            if name[-1] == '.vocab':
                with open(os.path.join(path, file), 'r', encoding='utf-8') as lines:
                    self.__vocab[name[0]] = set(filter(lambda x: len(x) > 0, [line.strip() for line in lines]))

    def generate(self, grammar: str):
        alters = GrammarGenerator.__alternate_pattern.findall(grammar)
        items = GrammarGenerator.__alternate(grammar, alters)

        result = []
        for item in items:
            entities = GrammarGenerator.__replace_pattern.findall(item)
            result += GrammarGenerator.__replace(item, entities, self.__vocab)

        return result

    @staticmethod
    def __alternate(text: str, alters: list):
        result = []
        if len(alters) > 0:
            alter = alters.pop(0)
            ws = alter[1].split('|')
            for w in ws:
                result += GrammarGenerator.__alternate(text.replace(alter[0], w, 1), alters[:])
        else:
            result.append(text)

        return result

    @staticmethod
    def __replace(text: str, entities: list, vocab: dict):
        result = []
        if len(entities) > 0:
            entity = entities.pop(0)
            slots = vocab[entity[1]]
            for slot in slots:
                result += GrammarGenerator.__replace(text.replace(entity[0], slot, 1), entities[:], vocab)
        else:
            result.append(text)

        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--vocab')
    parser.add_argument('--count', type=int)
    args = parser.parse_args()

    generator = GrammarGenerator(args.vocab)

    for root, sub, files in os.walk(args.input):
        for file in files:
            if file.endswith('.grammar'):
                result = []
                domain = os.path.splitext(file)[0]
                with open(os.path.join(root, file), 'r', encoding='utf-8') as lines:
                    for line in lines:
                        line = line.strip()
                        if len(line) > 0:
                            result += generator.generate(line)

                print('-- %s --' % domain)
                print('%d are generated from %s.' % (len(result), file))

                random.shuffle(result)
                result = result[:args.count]

                slots, train, dev, test = DataSplitter.split(domain, result)
                print('slot: %d' % len(slots))
                print('train: %d' % len(train))
                print('dev: %d' % len(dev))
                print('test: %d\n' % len(test))

                with open('./output/{}.slot'.format(domain), 'w', encoding='utf-8') as output:
                    output.write('\n'.join(slots))
                with open('./output/{}.train'.format(domain), 'w', encoding='utf-8') as output:
                    output.write('\n'.join(['[{}] {}'.format(domain, x) for x in train]))
                with open('./output/{}.dev'.format(domain), 'w', encoding='utf-8') as output:
                    output.write('\n'.join(['[{}] {}'.format(domain, x) for x in dev]))
                with open('./output/{}.test'.format(domain), 'w', encoding='utf-8') as output:
                    output.write('\n'.join(['[{}] {}'.format(domain, x) for x in test]))
