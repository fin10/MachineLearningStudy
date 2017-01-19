import argparse
import re
from operator import itemgetter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('input', help='folder of ann files')
args = parser.parse_args()

path = Path(args.input)

total = 0
files = list(path.glob('**/*.ann'))
with open('corpus_with_tag.output', 'w', encoding='utf-8') as output:
    for file in files:
        with file.open('r', encoding='utf-8') as ann:
            ann.readline()
            ann.readline()

            while True:
                sentence = re.sub('[()\[\]]', ' ', ann.readline().rstrip())
                if sentence.isdigit():
                    break

                count = int(ann.readline())
                if count <= 0:
                    continue

                indexes = []
                for i in range(count):
                    tag = re.search('\sETC\t(\d+)\t(\d+)', ann.readline())
                    indexes.append((int(tag.group(1)), int(tag.group(2))))
                indexes.sort(key=itemgetter(0))

                sentence_with_tag = sentence[0:indexes[0][0]]
                for i in range(len(indexes)):
                    sentence_with_tag += '({0})[dummy]'.format(sentence[indexes[i][0]:indexes[i][1]])
                    sentence_with_tag += sentence[
                                         indexes[i][1]:i + 1 == len(indexes) and len(sentence) - 1 or indexes[i + 1][0]]

                output.write(sentence_with_tag)
                output.write('\n')
                total += 1

print('%d is generated.' % total)
