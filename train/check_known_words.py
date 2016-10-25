import argparse

parser = argparse.ArgumentParser()
parser.add_argument('w2v', help='input file for pretrained vocabulary vectors by word2vec')
parser.add_argument('input', help='input file to check words')
args = parser.parse_args()

voca = dict()
with open(args.w2v, 'r') as f:
    voca = eval(f.read())
    
known = []
with open(args.input, 'r') as f:
    for word in f:
        if word.strip().lower() in voca:
            known.append(word.strip())
        

with open('./known_words.list', 'w') as f:
    f.write('\n'.join(known))

print('%d are generated.' % len(known))