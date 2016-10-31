import argparse
import re
import random

alternatePattern = re.compile('\([ <>a-zA-Z0-9\|]+\)')
replacePattern = re.compile('<[-_a-zA-Z0-9]+>')

def alternate(text, words, output):
    try:
        word = words.pop(0)
        ws = word[1:-1].split('|')
        for w in ws:
            alternate(text.replace(word, w, 1), words[:], output)
    except IndexError as e:
        output.append(text)
        return
        
def replace(text, slots, output):
    try:
        slot = slots.pop(0)
        slotFile = open('./' + slot[1:-1], 'r')
        for word in slotFile:
            word = word.strip();
            if (len(word) > 0):
                replace(text.replace(slot, word, 1), slots[:], output)
    except IndexError as e:
        output.append(text)
        return
    except OSError as e:
        print(e)
        return

parser = argparse.ArgumentParser()
parser.add_argument('input', help='Input a grammar file')
parser.add_argument('num_train', type=int, help='Number of training data')
parser.add_argument('num_test', type=int, help='Number of test data')
args = parser.parse_args()

with open(args.input, 'r') as input:
    result = []
    result2 = []
    for line in input:
        utterance = line.strip().lower()
        words = re.findall(alternatePattern, utterance)
        alternate(utterance, words, result)
        for item in result:
            slots = re.findall(replacePattern, item)
            replace(item, slots, result2)
        
    random.shuffle(result2)
    train = result2[:args.num_train]
    test = result2[args.num_train:(args.num_train + args.num_test)]
    
    with open((args.input) + '.train', 'w') as output:
        output.write('\n'.join(train))
        
    with open((args.input) + '.test', 'w') as output:
        output.write('\n'.join(test))
        
    print('input: %s\nUtterance geneated: %d for training, %d for test' % (args.input, len(train), len(test)))
