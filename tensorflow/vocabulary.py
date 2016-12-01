import numpy as np


class Vocabulary:
    def __init__(self, word2vec_file_path):
        self.__voca = dict()
        with open(word2vec_file_path, 'r') as f:
            word_vectors = eval(f.read())
            for word in word_vectors.items():
                self.__voca[word[0]] = np.frombuffer(word[1], dtype=np.float32)

        self.__dimension = len(list(self.__voca.values())[0])
        self.__empty = np.zeros([self.__dimension], dtype=np.float32)
        self.__empty.fill(-1)

    def get(self, word):
        return self.__voca[word] if word in self.__voca else self.__empty

    def dimension(self):
        return self.__dimension
