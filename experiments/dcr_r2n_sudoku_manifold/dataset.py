import random
import math
import sys

from keras_ns.dataset import Dataset
from keras_ns.logic import FOL, Domain, Predicate
from itertools import product
from collections import defaultdict,namedtuple
import numpy as np

from sudoku  import Sudoku # pip install py-sudoku
import tensorflow_datasets as tfds



Corruption = namedtuple("Corruption", "head tail")



class SudokuDataHandler():

    def __init__(self, dataset_name = "sudoku",
                 n = 4,
                 ragged=False,
                 format="functional",

                 ):

        assert n in [4,9]
        self.format = format

        name = dataset_name
        self.domain_name = name
        self.ragged = ragged
        self.n = n

        m = int(math.sqrt((n)))

        rows = np.reshape(np.arange(0, n * n), [n, n])

        columns = rows.T

        squares = np.array([np.reshape(rows[i:i + m, j:j + m], -1) for i, j in product(range(0, n, m), range(0, n, m))])

        images = defaultdict(lambda: [])
        ds_train = tfds.load('mnist',split=['train'],shuffle_files=True,as_supervised=True)[0]
        for i, (image, label) in enumerate(tfds.as_numpy(ds_train)):
            images[label].append(image)

        queries = []
        labels = []
        self.images = []
        self.labels_images = []
        for id_sudoku in list(range(0, 4500*n, n*n)): # 4500 because some digits has not 6000. mnist unbalanced?????? #TODO check
            seed = random.randint(0, sys.maxsize-1)
            solution = Sudoku(m, seed=seed).solve().board # set a fixed seed fro random, so I pass a random one from the outside
            board = np.reshape(solution, [-1]) - 1  # zero-based
            r = random.random()
            if r > 0.5: # positive
                labels.append([1])
            else: # negative
                labels.append([0])

                #we corrupt the board #TODO: check that this is a good corruption
                number_corruptions = random.randint(0,n*n)
                t = 0
                while t < number_corruptions:
                    a = random.randint(0,n*n-1)
                    b = random.randint(0,n*n-1)
                    if a != b:
                        c = board[a]
                        board[a] = board[b]
                        board[b] = c
                        t+=1
            # print(np.reshape(board, [n,n]))
            # print(labels[-1])

            for i in board:
                self.images.append(images[i].pop())
                self.labels_images.append(i)
            queries.append(["sudoku(%s)" % id_sudoku] )

        self.queries = queries
        self.labels = labels

        self.images = np.array(self.images)
        self.labels_images = np.array(self.labels_images)



        self.sudokus = list(range(0, 60000, n*n))
        self.domains = [Domain("images", list(range(0, 60000))),
                        Domain("digits", list(range(n))),
                        Domain("sudokus", self.sudokus),
                        Domain("structures", list(range(n)))]
        self.domains_dict = {d.name:d for d in self.domains}
        self.predicates = [Predicate("digit",[self.domains_dict["images"], self.domains_dict["digits"]]),
                           Predicate("column", [self.domains_dict["sudokus"], self.domains_dict["structures"]]),
                           Predicate("row", [self.domains_dict["sudokus"], self.domains_dict["structures"]]),
                           Predicate("square", [self.domains_dict["sudokus"], self.domains_dict["structures"]]),
                           Predicate("sudoku", [self.domains_dict["sudokus"]])]

        self.manifolds = {
            "columns": [(sudoku, i, *(column+sudoku)) for sudoku,(i,column) in product(self.sudokus, enumerate(columns))],
            "rows": [(sudoku, i, *(column+sudoku)) for sudoku,(i,column) in product(self.sudokus, enumerate(rows))],
            "squares": [(sudoku, i, *(column+sudoku)) for sudoku,(i,column) in product(self.sudokus, enumerate(squares))],
            }

        self.fol = FOL(self.domains, self.predicates)

    def get_dataset(self, split:str):
        f = int(len(self.queries) * 0.8)
        s = int(len(self.labels) * 0.9)
        if split == "train":
            queries = self.queries[:f]
            labels = self.labels[:f]
        elif split == "valid":
            queries = self.queries[f:s]
            labels = self.labels[f:s]
        elif split == "test":
            queries = self.queries[s:]
            labels = self.labels[s:]
        else:
            raise Exception("Split %s unknown" %split)


        return Dataset(queries, labels, constants_features={self.domain_name: self.constants_features})








if __name__ == "__main__":
    dh = SudokuDataHandler(n=9)

    for id in range(10):
        label = dh.labels[id]
        print(label)
        print(np.reshape(dh.labels_images[id * (dh.n*dh.n): (id+1)*dh.n*dh.n ], [dh.n,dh.n]))
        import matplotlib.pyplot as plt

        f, axarr = plt.subplots(nrows=dh.n, ncols=dh.n)
        for i, (row,col) in enumerate(product(range(dh.n),range(dh.n))):
            axarr[row, col].imshow(dh.images[id * (dh.n*dh.n): (id+1)*dh.n*dh.n ][i])
        plt.show()