import argparse
import re;

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
parser.add_argument('input', help='input file')
args = parser.parse_args()

input = open(args.input, 'r')
output = open(args.input + '.out', 'w')

try:
    result = []
    result2 = []
    for line in input:
        utterance = line.strip().lower()
        words = re.findall(alternatePattern, utterance)
        alternate(utterance, words, result)
        for item in result:
            slots = re.findall(replacePattern, item)
            replace(item, slots, result2)
        
    output.write('\n'.join(result2))
    print('input: %s\n Utterance geneated: %d' % (args.input, len(result2)))
    
finally:
    input.close()

