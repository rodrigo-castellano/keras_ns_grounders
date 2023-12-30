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


# This implements a double method of negatives: fixed number and all possible
def check_next_corruption(ret, num_negatives, last_idx, num_constants):
    if num_negatives is not None:
        idx = random.randint(0, num_constants - 1)
        last = len(ret) >= num_negatives
        return idx, last
    else:
        last_idx = last_idx + 1
        last = last_idx >= num_constants - 1
        return last_idx, last




def _facts_to_triples(facts):
    triples = []
    for fact in facts:
        p, rest = fact.split("(")
        constants = rest.split(")")[0].split(",")
        for i in range(len(constants)):
            arg = constants[i].replace(" ", "")
            constants[i] = arg
            triples.append((constants[0], p, constants[1]))
    return triples


def _triple_to_fact(triple):
    h, r, t = triple
    return "%s(%s,%s)" % (r, h, t)


def _triples_to_facts(triples):
    return ["%s(%s,%s)" % (r, h, t) for h, r, t in triples]


def read_ntp_ontology_only(file, base_path):
    file = join(base_path, file)
    predicates = OrderedDict()
    constants = OrderedDict()
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
                predicates[p] = len(predicates)
    return constants, predicates


def read_ontology(base_path, train_path, valid_path, test_path):
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
        for file in [train_path, valid_path, test_path]:
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
                        predicates[p] = len(predicates)
    return constants, predicates


def read_ontology_from_files(base_path, constants_file, relations_file):
    constants_file = join(base_path, constants_file)
    relations_file = join(base_path, relations_file)
    predicates = OrderedDict()
    constants = OrderedDict()
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

    return constants, predicates



class KGCTrainingDataset(Dataset):

    def __init__(self, queries, labels, num_negatives, known_facts, constants, constants_features=None, format="functional"):
        super().__init__(queries, labels, constants_features, format=format)
        self.num_negatives = num_negatives
        self.known_facts = set(f if isinstance(f,Tuple) else f.toTuple() if isinstance(f,Atom) else Atom(s=f, format=format).toTuple() for f in known_facts)
        self.constants = constants


    def __getitem__(self, item: Union[int,slice]):
        queries,labels =  self.queries[item], self.labels[item]
        if isinstance(item,int):
            queries = [queries]
            labels = [labels]

        facts_to_corrupt = [q[0] for q in queries]  # Get positive to corrupt
        corruptions_per_query = KGCDataHandler.create_corruptions(
            queries=facts_to_corrupt,
            known_facts=self.known_facts,
            constants=self.constants,
            num_negatives=self.num_negatives)

        # Training corruptions are mixed head/tail corruptions
        Q = []
        L = []
        for q,l,c in zip(queries, labels, corruptions_per_query):
            Q.append(q + c.head + c.tail)
            L.append(l + [0] * (len(c.head)+len(c.tail)))

        if isinstance(item, int):
            Q = Q[0]
            L = L[0]
        return Q, L

class KGCEvalDataset(Dataset):

    """It creates two different test sample per each query: one with only head corruptions
    and one with only test corruptions"""

    def __init__(self, queries, labels, num_negatives, known_facts, constants, constants_features=None, format="functional"):
        super().__init__(queries, labels, constants_features, format=format)
        self.num_negatives = num_negatives
        self.known_facts = set(f if isinstance(f,Tuple) else f.toTuple() if isinstance(f,Atom) else Atom(s=f, format=format).toTuple() for f in known_facts)
        self.constants = constants


    def __getitem__(self, item: Union[int, slice]):

        #Slice the positive
        queries, labels = self.queries[item], self.labels[item]
        if isinstance(item,int):
            queries = [queries]
            labels = [labels]

        #Sample the corruptions
        facts_to_corrupt = [q[0] for q in queries] # Get positive to corrupt
        corruptions_per_query = KGCDataHandler.create_corruptions(
            queries=facts_to_corrupt,
            known_facts=self.known_facts,
            constants=self.constants,
            num_negatives=self.num_negatives)

        # Eval corruptions are split head and tail corruptions
        Q = []
        L = []
        for q,l,c in zip(queries, labels, corruptions_per_query):
            Q.append(q + c.head)
            Q.append(q + c.tail)
            L.append(l + [0] * len(c.head))
            L.append(l + [0] * len(c.tail))
        return Q, L


class KGCDataHandler():

    def __init__(self, dataset_name,
                 base_path,
                 ragged=False,
                 num_negatives=None,
                 format="functional",
                 valid_size=None,
                 train_file="train.txt",
                 valid_file="valid.txt",
                 test_file="test.txt"
                 ):

        self.num_negatives = num_negatives
        self.format = format

        name = dataset_name
        self.domain_name = name
        self.ragged = ragged

        base_path  = join(base_path, dataset_name)
        train_path = join(base_path, train_file)
        valid_path = join(base_path, valid_file)
        test_path = join(base_path, test_file)

        constants, predicates = read_ontology(base_path, train_path, valid_path, test_path)

        self.constants = sorted(list(constants.keys()))
        self.domains = [Domain(self.domain_name, self.constants)]
        predicates = sorted(list(predicates.keys()))
        self.predicates = [Predicate(p,[self.domains[0], self.domains[0]])
                           for p in predicates]

        # Transformation from strings to atoms.
        self.train_facts = [Atom(s=s, format=format).toTuple()
                            for s in read_file_as_lines(train_path)]
        self.valid_facts = [Atom(s=s, format=format).toTuple()
                            for s in read_file_as_lines(valid_path)]
        self.test_facts = [Atom(s=s, format=format).toTuple()
                           for s in read_file_as_lines(test_path)]

        if valid_size is not None:
            self.valid_facts = self.valid_facts[:valid_size]
            self.train_facts = self.train_facts + self.valid_facts[valid_size:]

        self.train_facts_set = set(self.train_facts)
        self.valid_facts_set = set(self.valid_facts)
        self.test_facts_set = set(self.test_facts)

        self.train_val_ground_facts_set = set(self.train_facts + self.valid_facts)
        self.ground_facts_set = set(self.train_facts + self.valid_facts + self.test_facts)

        self.fol = FOL(self.domains, self.predicates, self.train_facts_set)

    @staticmethod
    def create_corruptions(queries: List[Tuple],
                           known_facts: Set[Tuple],
                           constants: List[str],
                           num_negatives: int=None) -> List[Corruption]:
        if num_negatives is None:
            # print("creating all corruptions\n")
            return KGCDataHandler.create_all_corruptions(queries, known_facts, constants)
        else:
            # print("creating sampled corruptions\n")
            return KGCDataHandler.create_sampled_corruptions(queries, known_facts, constants, num_negatives)


    @staticmethod
    def create_all_corruptions(queries: List[Tuple],
                               known_facts: Set[Tuple],
                               constants: List[str]) -> List[Corruption]:
        # print("Creating all the corruptions for batch...")
        # start = timeit.default_timer()
        Q=[]

        for i, q in enumerate(queries):
            ret1=[]
            ret2=[]
            r_idx, s_idx, o_idx = q

            for entity in constants:
                a1 = (r_idx,s_idx,entity)
                if a1 not in known_facts:
                    ret1.append(a1)

                a2 = (r_idx, entity, o_idx)
                if a2 not in known_facts:
                    ret2.append(a2)

            Q.append(Corruption(head=ret1, tail=ret2))

        return Q


    @staticmethod
    def create_sampled_corruptions(queries: List[Tuple],
                                   known_facts: Set[Tuple],
                                   constants: List[str],
                                   num_negatives: int) -> List[Corruption]:
        n_constants = len(constants)
        Q = []

        for q in queries:
            num_corruptions_head = 0
            num_corruptions_tail = 0

            ret1=[]
            ret2=[]
            r_idx, s_idx, o_idx = q

            # Head corruptions
            while num_corruptions_head < num_negatives:
                idx = random.randint(0, n_constants - 1)
                entity = constants[idx]
                a1 = (r_idx, s_idx, entity)
                if a1 not in known_facts:
                    ret1.append(a1)
                    num_corruptions_head += 1

            # Tail corruptions
            while num_corruptions_tail < num_negatives:
                idx = random.randint(0, n_constants - 1)
                entity = constants[idx]
                a2 = (r_idx, entity, o_idx)
                if a2 not in known_facts:
                    ret2.append(a2)
                    num_corruptions_tail += 1


            Q.append(Corruption(head=ret1, tail=ret2))

        return Q

    def get_dataset(self, split:str, number_negatives:int=None) -> Union[
        KGCTrainingDataset, KGCEvalDataset]:
        train_known_facts = self.train_facts_set
        if split == "train":
            queries, labels = [[q] for q in self.train_facts], [[1] for _ in self.train_facts]
            return KGCTrainingDataset(queries=queries, labels=labels,
                                      num_negatives=number_negatives,
                                      known_facts=self.train_facts_set,
                                      constants=self.constants)
        else:
            if split == "valid":
                queries, labels = [[q] for q in self.valid_facts], [[1] for _ in self.valid_facts]
            else:
                queries, labels = [[q] for q in self.test_facts], [[1] for _ in self.test_facts]
            return KGCEvalDataset(queries=queries, labels=labels,
                                      num_negatives=number_negatives,
                                      known_facts=self.ground_facts_set, #TODO: Check for baselines w.r.t. train_facts_set
                                      constants=self.constants)



if __name__ == "__main__":
    pass
