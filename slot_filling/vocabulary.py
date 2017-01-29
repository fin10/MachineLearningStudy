import numpy as np


class Vocabulary:
    def __init__(self, path: str):
        self.__vocab = {}

        with open(path, 'r') as file:
            word_vectors = eval(file.read())
            for word in word_vectors.items():
                self.__vocab[word[0]] = np.frombuffer(word[1], dtype=np.float32)

        self.__dimension = len(list(self.__vocab.values())[0])

        self.__empty = np.zeros([self.__dimension], dtype=np.float32)
        self.__empty.fill(-1)

        self.__zeros = np.zeros([self.__dimension], dtype=np.float32)

    def get(self, word: str):
        if word in self.__vocab:
            return self.__vocab[word]
        else:
            return self.__empty

            # return word in self.__vocab and self.__vocab[word] or self.__empty

    def get_zeros(self):
        return self.__zeros

    def get_dimension(self):
        return self.__dimension
