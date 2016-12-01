import random

import numpy as np


class Dataset:
    def __init__(self):
        self.__uttr = []
        self.__iob = []
        self.__vectors = []
        self.__target = []
        self.__cursor = 0

    @property
    def uttr(self):
        return self.__uttr

    @property
    def iob(self):
        return self.__iob

    @property
    def vectors(self):
        return self.__vectors

    @property
    def target(self):
        return self.__target

    @property
    def length(self):
        return len(self.vectors)

    def add(self, uttr, iob, vectors, target):
        self.uttr.append(uttr)
        self.iob.append(iob)
        self.vectors.append(vectors)
        self.target.append(target)

    def extend(self, size: int):
        for i in range(self.length):
            self.vectors[i] = np.pad(self.vectors[i], [[0, size - len(self.vectors[i])], [0, 0]], 'constant')
            self.target[i] = np.pad(self.target[i], [[0, size - len(self.target[i])], [0, 0]], 'constant')

    def sample(self, num: int):
        indexes = random.sample([x for x in range(self.length)], num)
        new_dataset = Dataset()
        new_dataset.__uttr = [self.__uttr[i] for i in indexes]
        new_dataset.__iob = [self.__iob[i] for i in indexes]
        new_dataset.__vectors = [self.__vectors[i] for i in indexes]
        new_dataset.__target = [self.__target[i] for i in indexes]
        return new_dataset
