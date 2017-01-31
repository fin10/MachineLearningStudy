import argparse
import os
import re


class GrammarModel:
    @staticmethod
    def alternate(grammars: list):
        alternate_pattern = re.compile('(\{([^}]+)\})')

        result = []
        for grammar in grammars:
            words = re.findall(alternate_pattern, grammar)
            result += GrammarModel.__alternate(grammar, words)

        return result

    @staticmethod
    def __alternate(text: str, words: list):
        result = []
        if len(words) > 0:
            word = words.pop(0)
            ws = word[1].split('|')
            for w in ws:
                result += GrammarModel.__alternate(text.replace(word[0], w, 1), words[:])
        else:
            result.append(text)

        return result

    @staticmethod
    def replace(grammars: list, slot_dict: dict):
        replace_pattern = re.compile('(<(\S+)>)')

        result = []
        for grammar in grammars:
            entities = re.findall(replace_pattern, grammar)
            result += GrammarModel.__replace(grammar, entities, slot_dict)

        return result

    @staticmethod
    def __replace(text: str, entities: list, slot_dict: dict):
        result = []
        if len(entities) > 0:
            entity = entities.pop(0)
            slots = slot_dict[entity[1]]
            for slot in slots:
                result += GrammarModel.__replace(text.replace(entity[0], slot, 1), entities[:], slot_dict)
        else:
            result.append(text)

        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input a grammar file')
    args = parser.parse_args()

    grammars = []
    with open(args.input, 'r') as file:
        for line in file:
            line = line.strip().lower()
            if len(line) > 0:
                grammars.append(line)

    slot_dict = {}
    dir_name = os.path.dirname(args.input)
    files = os.listdir(dir_name)
    for file in files:
        name = os.path.splitext(file)
        if name[-1] == '.slot':
            with open(os.path.join(dir_name, file), 'r', encoding='utf-8') as f:
                slot_dict[name[0]] = set(filter(lambda x: len(x) > 0, [line.strip() for line in f]))

    result = GrammarModel.alternate(grammars)
    result = GrammarModel.replace(result, slot_dict)

    with open(args.input + '.output', 'w') as output:
        output.write('\n'.join(result))

    print('%d are generated.' % len(result))
