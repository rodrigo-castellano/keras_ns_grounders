import copy
import os.path
import random

from keras_ns.dataset import Dataset
from keras_ns.logic import FOL, Domain, Predicate
from typing import List, Union, Set, Tuple
from os.path import join
from collections import OrderedDict
from keras_ns.utils import read_file_as_lines
from keras_ns.logic.commons import Atom
from itertools import product
from collections import defaultdict,namedtuple
from keras_ns.metrics import MRRMetric
import numpy as np
from functools import lru_cache
import tensorflow as tf

import timeit




Corruption = namedtuple("Corruption", "head tail")



class CollectiveDataHandler():

    def __init__(self, dataset_name,
                 base_path,
                 ragged=False,
                 format="functional",
                 train_file="train.txt",
                 valid_file="valid.txt",
                 test_file="test.txt",
                 facts_file = "facts.txt"
                 ):

        self.format = format

        name = dataset_name
        self.domain_name = name
        self.ragged = ragged

        base_path  = join(base_path, dataset_name)
        train_path = join(base_path, train_file)
        valid_path = join(base_path, valid_file)
        test_path = join(base_path, test_file)
        facts_path = join(base_path, facts_file)
        classes_path = join(base_path, "classes.txt")
        features_path = join(base_path, "features.txt")

        constants, predicates = read_ontology(base_path, [train_path, valid_path, test_path,facts_path])

        self.classes = open(classes_path).readlines()[0].split(",")

        self.constants = sorted(list(constants.keys()))
        self.domains = [Domain(self.domain_name, self.constants), Domain("classes", self.classes)]
        predicates = sorted(list(predicates.items()), key = lambda x: x[0])
        self.predicates = [Predicate("class",[self.domains[0], self.domains[1]]),
                           Predicate("r", [self.domains[0], self.domains[0]])]

        # Transformation from strings to atoms.
        self.train_facts = [Atom(s=s, format=format).toTuple()
                            for s in read_file_as_lines(train_path)]
        self.valid_facts = [Atom(s=s, format=format).toTuple()
                            for s in read_file_as_lines(valid_path)]
        self.test_facts = [Atom(s=s, format=format).toTuple()
                           for s in read_file_as_lines(test_path)]
        self.known_facts = [Atom(s=s, format=format).toTuple()
                           for s in read_file_as_lines(facts_path)]




        with open(features_path) as f:
            lines = f.readlines()
            all_features = {}
            for line in lines:
                (id, features) = line.split(":")
                features = [float(a) for a in features.split(",")]
                all_features[id] = features

            self.constants_features = []
            for c in self.constants:
                self.constants_features.append(all_features[c])
            self.constants_features = tf.constant(self.constants_features)

        self.train_facts_set = set(self.train_facts)
        self.valid_facts_set = set(self.valid_facts)
        self.test_facts_set = set(self.test_facts)
        self.known_facts_set = set(self.known_facts)

        self.train_val_ground_facts_set = set(self.train_facts + self.valid_facts + self.known_facts)
        self.ground_facts_set = set(self.train_facts + self.valid_facts + self.test_facts +self.known_facts)

        self.fol = FOL(self.domains, self.predicates, self.train_facts_set.union(self.known_facts))

    def get_dataset(self, split:str):
        if split == "train":
            facts = self.train_facts
        elif split == "valid":
            facts = self.valid_facts
        elif split == "test":
            facts = self.test_facts
        else:
            raise Exception("Split %s unknown" % split)

        queries = []
        labels = []

        for t in facts:
            if t[0]=="class":
                queries.append([])
                labels.append([])
                for r in self.classes:
                    query = (t[0],t[1], r)
                    label = int(r == t[2])
                    queries[-1].append(query)
                    labels[-1].append(label)
            else:
                r,c1,c2 = t[0],t[1],t[2]
                queries.append([])
                labels.append([])
                query = (r, c1, c2)
                label = 1
                queries[-1].append(query)
                labels[-1].append(label)
                for c in self.domains[0].constants:
                    for corruption in [(r,c,c2), (r,c1,c)]:
                        if corruption not in self.known_facts_set:
                            label = 0
                            queries[-1].append(corruption)
                            labels[-1].append(label)


        return Dataset(queries, labels, constants_features = {self.domain_name: self.constants_features})






def read_ontology(base_path, paths):
    constants_file = join(base_path, "entities.dict")
    relations_file = join(base_path, "relations.dict")

    predicates = OrderedDict()
    constants = OrderedDict()
    if os.path.exists(constants_file) and os.path.exists(relations_file):

        with open(constants_file) as f:
            for line in f:
                line = line.replace("\n", "")
                id, c = line.split("\t")
                constants[c] = id

        with open(relations_file) as f:
            for line in f:
                line = line.replace("\n", "")
                id, r = line.split("\t")
                predicates[r] = id
    else:
        for file in paths:
            with open(file) as f:
                for line in f:
                    p, rest = line.split("(")
                    args = rest.split(")")[0].split(",")
                    for i in range(len(args)):
                        arg = args[i].replace(" ", "")
                        args[i] = arg
                        if arg not in constants:
                            constants[arg] = len(constants)
                    if p not in predicates:
                        predicates[p] = (len(predicates), len(args))
    return constants, predicates


if __name__ == "__main__":
    pass
