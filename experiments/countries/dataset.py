import copy
import os.path
import random

from keras_ns.dataset import Dataset
from keras_ns.logic import FOL, Domain, Predicate
from typing import Dict, List, Union, Set, Tuple
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

def read_atoms(paths: List[str], format: str):
    #constants_file = join(base_path, "entities.dict")
    #relations_file = join(base_path, "relations.dict")
    atoms = []
    for file in paths:
        if not os.path.exists(file):
            print('Skipping reading %s because missing' % file, flush=True)
            continue
        atoms += [Atom(s=s, format=format).toTuple()
                  for s in read_file_as_lines(file)]
    return list(set(atoms))

def read_domains(path: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    # If a domain file is provided, the constants can belong to different
    # domains. In this case the file is expected to be in format:
    # domain_name1 space_separated_constants_in_the_domain
    # domain_name2 space_separated_constants_in_the_domain
    # ...
    constant2domain = {}
    domain2constants = {}
    domain_lines = read_file_as_lines(path)
    for line in domain_lines:
        t = line.split(' ')
        domain_name = t[0]
        domain2constants[domain_name] = []
        for c in sorted(list(set(t[1:]))):
            assert c not in constant2domain, 'Repeated constant %s' % c
            constant2domain[c] = domain_name
            domain2constants[domain_name].append(c)
    return constant2domain, domain2constants

def build_domains(filename: str) -> Dict[str, Domain]:
    _, domain2constants = read_domains(os.path.join(filename))
    name2domain = {name : Domain(name=name, constants=constants)
                   for name, constants in domain2constants.items()}
    return name2domain

def read_ontology(paths: List[str], format: str):
    predicates = OrderedDict()
    constants = OrderedDict()
    atoms = read_atoms(paths, format)
    # print('atoms', len(atoms),atoms[:10], flush=True)
    for a in atoms:
        p = a[0]
        if p not in predicates:
            predicates[p] = len(predicates)
        for c in a[1:]:
            if c not in constants:
                constants[c] = len(constants)
    return constants, predicates

# Computes a dict predicate_name -> (domain_name1, domain_name2, ...)
def Predicate2Domains(
    atoms: List[Tuple[str, str, str]],
    constant2domain: Dict[str, str]) -> Dict[str, List[Tuple[str]]]:
    predictate2domains = {}
    for a in atoms:
        p = a[0]
        for c in a[1:]:
            assert c in constant2domain, 'Unknown domain for %s' % c
        domain_tuple = tuple([constant2domain[c] for c in a[1:]])
        if (p in predictate2domains and
            domain_tuple not in predictate2domains[p]):
            predictate2domains[p].append(domain_tuple)
        else:
            predictate2domains[p] = [domain_tuple]
    return predictate2domains


class KGCTrainingDataset(Dataset):

    def __init__(self, queries, labels, num_negatives, known_facts,
                 constant2domain, domain2constants,
                 constants_features=None, format="functional",
                 corrupt_mode: str='HEAD_AND_TAIL'):
        super().__init__(queries, labels, constants_features, format=format)
        self.num_negatives = num_negatives
        self.known_facts = set(f if isinstance(f,Tuple) else f.toTuple()
                               if isinstance(f,Atom) else
                               Atom(s=f, format=format).toTuple()
                               for f in known_facts)
        self.constant2domain=constant2domain
        self.domain2constants=domain2constants
        self.corrupt_mode = corrupt_mode

    def __getitem__(self, item: Union[int,slice]):
        queries,labels =  self.queries[item], self.labels[item]
        if isinstance(item, int):
            queries = [queries]
            labels = [labels]

        facts_to_corrupt = [q[0] for q in queries]  # Get positive to corrupt
        corruptions_per_query = KGCDataHandler.create_corruptions(
            queries=facts_to_corrupt,
            known_facts=self.known_facts,
            constant2domain=self.constant2domain,
            domain2constants=self.domain2constants,
            num_negatives=self.num_negatives,
            corrupt_mode=self.corrupt_mode)

        # Training corruptions are mixed head/tail corruptions
        Q = []
        L = []
        for q,l,c in zip(queries, labels, corruptions_per_query):
            Q.append(q + c.head + c.tail)
            L.append(l + [0] * (len(c.head)+len(c.tail)))

        if isinstance(item, int):
            Q = Q[0]
            L = L[0]
        # tf.print('len Q', len(Q), 'len L', len(L))
        # tf.print('QUERY TRAIN', Q[0], 'LABEL', L[0])
        return Q, L

class KGCEvalDataset(Dataset):

    """It creates two different test sample per each query: one with only head corruptions
    and one with only test corruptions"""

    def __init__(self, queries, labels, num_negatives, known_facts,
                 constant2domain, domain2constants,
                 constants_features=None, format="functional",
                 corrupt_mode: str='HEAD_AND_TAIL'):
        super().__init__(queries, labels, constants_features, format=format)
        self.num_negatives = num_negatives
        self.known_facts = set(f if isinstance(f,Tuple) else f.toTuple()
                               if isinstance(f, Atom) else
                               Atom(s=f, format=format).toTuple()
                               for f in known_facts)
        self.constant2domain = constant2domain
        self.domain2constants = domain2constants
        self.corrupt_mode = corrupt_mode

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
            constant2domain=self.constant2domain,
            domain2constants=self.domain2constants,
            num_negatives=self.num_negatives,
            corrupt_mode=self.corrupt_mode)

        # Eval corruptions are split head and tail corruptions
        Q = []
        L = []
        for q,l,c in zip(queries, labels, corruptions_per_query):
            Q.append(q + c.head)
            Q.append(q + c.tail)
            L.append(l + [0] * len(c.head))
            L.append(l + [0] * len(c.tail))
        # tf.print('len Q', len(Q), 'len L', len(L))
        # tf.print('QUERY VAL/TEST', Q[0], 'LABEL', L[0])
        # tf.print('QUERY VAL/TEST', Q[1], 'LABEL', L[1])
        return Q, L


class KGCDataHandler():

    def __init__(self, dataset_name,
                 base_path,
                 ragged=False,
                 num_negatives=None,
                 format="functional",
                 valid_size=None,
                 domain_file=None,
                 train_file="train.txt",
                 valid_file="valid.txt",
                 test_file="test.txt",
                 fact_file="facts.txt"):

        self.num_negatives = num_negatives
        self.format = format

        name = dataset_name
        base_path  = join(base_path, dataset_name)

        self.ragged = ragged
        train_path = join(base_path, train_file)
        valid_path = join(base_path, valid_file)
        test_path = join(base_path, test_file)
        fact_path = join(base_path, fact_file)

        constants, predicates = read_ontology(
            # [train_path, valid_path, test_path], format)
            [train_path, valid_path, test_path, fact_path], format)

        self.constants = sorted(list(constants.keys()))
        predicates = sorted(list(predicates.keys()))

        # Transformation from strings to atoms.
        self.train_facts = [Atom(s=s, format=format).toTuple()
                            for s in read_file_as_lines(train_path)]
        self.valid_facts = [Atom(s=s, format=format).toTuple()
                            for s in read_file_as_lines(valid_path)]
        self.test_facts = [Atom(s=s, format=format).toTuple()
                           for s in read_file_as_lines(test_path)]
        self.known_facts = [Atom(s=s, format=format).toTuple()
                            for s in read_file_as_lines(fact_path)]

        if valid_size is not None:
            self.valid_facts = self.valid_facts[:valid_size]
            self.train_facts = self.train_facts + self.valid_facts[valid_size:]

        self.train_facts_set = set(self.train_facts)
        self.valid_facts_set = set(self.valid_facts)
        self.test_facts_set = set(self.test_facts)
        self.known_facts_set = set(self.known_facts)

        self.train_known_facts_set = set(self.train_facts + self.known_facts)
        self.ground_facts_set = set(self.train_facts +
                                    self.valid_facts +
                                    self.test_facts +
                                    self.known_facts)

        # Create one global domain, this is still used by the serializer.
        self.default_domain_name = 'default'
        # Global domain.
        self.domains: List[Domain] = []
        # Domain(self.default_domain_name, self.constants)]

        if domain_file is not None:
            self.constant2domain, self.domain2constants = read_domains(
                join(base_path, domain_file))
            constants_set = set(self.constants)
            for c in self.constant2domain.keys():
                assert c in constants_set, (
                    '%s constant missing in the ontology constraits' % c)
            self.domains += [
                Domain(name, constants)
                for name,constants in self.domain2constants.items()]
        else:
            self.constant2domain = {}
            self.domain2constants = {}

        self.domain2constants[self.default_domain_name] = []

        # Missing constants are added to the default domain.
        for c in self.constants:
            if c not in self.domain2constants:
                self.domain2constants[self.default_domain_name].append(c)
            if c not in self.constant2domain:
                self.constant2domain[c] = self.default_domain_name

        name2domain: Dict[str, Domain] = {d.name:d for d in self.domains}

        self.predicates = []
        predicate2domains: Dict[str, List[Tuple[str]]] = Predicate2Domains(
            atoms=list(self.ground_facts_set),
            constant2domain=self.constant2domain)
        # print('predicate2domains', predicate2domains, flush=True)
        # Computes the domains for each positional input of a predicate, checking
        # that the possible domains are univocally determined.
        for p,domain_list in predicate2domains.items():
            # print('p', p, 'domain_list', domain_list, flush=True)
            assert len(domain_list) > 0
            num_possible_domains = len(domain_list)
            arity = len(domain_list[0])
            assert num_possible_domains == 1, '%s %s'%(p, domain_list)
            domains = [name2domain[d] for d in domain_list[0]]
            self.predicates.append(Predicate(p, tuple(domains)))
            # print('appended ',Predicate(p, tuple(domains)), flush=True)

        self.fol = FOL(self.domains, self.predicates, self.train_facts_set,
                       constant2domain_name=self.constant2domain)

    @staticmethod
    def create_corruptions(queries: List[Tuple],
                           known_facts: Set[Tuple],
                           domain2constants: Dict[str, List[str]],
                           constant2domain: Dict[str, str],
                           num_negatives: int=None,
                           corrupt_mode: str='HEAD_AND_TAIL') -> List[Corruption]:
        if num_negatives is None:
            # print("creating all corruptions\n")
            return KGCDataHandler.create_all_corruptions(
                queries, known_facts, domain2constants, constant2domain,
                corrupt_mode)
        else:
            # print("creating sampled corruptions\n")
            return KGCDataHandler.create_sampled_corruptions(
                queries, known_facts, domain2constants, constant2domain,
                num_negatives, corrupt_mode)


    @staticmethod
    def create_all_corruptions(queries: List[Tuple],
                               known_facts: Set[Tuple],
                               domain2constants: Dict[str, List[str]],
                               constant2domain: Dict[str, str],
                               corrupt_mode: str) -> List[Corruption]:
        # print("Creating all the corruptions for batch...")
        # start = timeit.default_timer()
        Q=[]

        for i, q in enumerate(queries):
            ret1=[]
            ret2=[]
            r_idx, s_idx, o_idx = q

            if corrupt_mode == 'HEAD_AND_TAIL' or corrupt_mode == 'TAIL':
                o_domain = constant2domain[o_idx]
                for entity in domain2constants[o_domain]:
                    a1 = (r_idx,s_idx,entity)
                    if a1 not in known_facts:
                        ret1.append(a1)

            if corrupt_mode == 'HEAD_AND_TAIL' or corrupt_mode == 'HEAD':
                s_domain = constant2domain[s_idx]
                for entity in domain2constants[s_domain]:
                    a2 = (r_idx, entity, o_idx)
                    if a2 not in known_facts:
                        ret2.append(a2)

            Q.append(Corruption(head=ret1, tail=ret2))

        return Q


    @staticmethod
    def create_sampled_corruptions(queries: List[Tuple],
                                   known_facts: Set[Tuple],
                                   domain2constants: Dict[str, List[str]],
                                   constant2domain: Dict[str, str],
                                   num_negatives: int,
                                   corrupt_mode: str) -> List[Corruption]:
        Q = []

        for q in queries:
            num_corruptions_head = 0
            num_corruptions_tail = 0

            ret1=[]
            ret2=[]
            r_idx, s_idx, o_idx = q

            if corrupt_mode == 'HEAD_AND_TAIL' or corrupt_mode == 'HEAD':
                # Head corruptions
                o_domain = constant2domain[o_idx]
                n_constants = len(domain2constants[o_domain])
                constants = domain2constants[o_domain]
                while num_corruptions_head < num_negatives:
                    idx = random.randint(0, n_constants - 1)
                    entity = constants[idx]
                    a1 = (r_idx, s_idx, entity)
                    if a1 not in known_facts:
                        ret1.append(a1)
                        num_corruptions_head += 1

            if corrupt_mode == 'HEAD_AND_TAIL' or corrupt_mode == 'TAIL':
                # Tail corruptions
                s_domain = constant2domain[s_idx]
                n_constants = len(domain2constants[s_domain])
                constants = domain2constants[s_domain]
                while num_corruptions_tail < num_negatives:
                    idx = random.randint(0, n_constants - 1)
                    entity = constants[idx]
                    a2 = (r_idx, entity, o_idx)
                    if a2 not in known_facts:
                        ret2.append(a2)
                        num_corruptions_tail += 1

            Q.append(Corruption(head=ret1, tail=ret2))

        return Q

    def get_dataset(self, split:str, number_negatives:int=None,
                    corrupt_mode: str='HEAD_AND_TAIL') -> Union[
        KGCTrainingDataset, KGCEvalDataset]:
        train_known_facts = self.train_facts_set
        if split == "train":
            queries, labels = [[q] for q in self.train_facts], [[1] for _ in self.train_facts]
            return KGCTrainingDataset(queries=queries, labels=labels,
                                      num_negatives=number_negatives,
                                      known_facts=self.train_known_facts_set,
                                      constant2domain=self.constant2domain,
                                      domain2constants=self.domain2constants,
                                      corrupt_mode=corrupt_mode)
        else:
            if split == "valid":
                queries, labels = [[q] for q in self.valid_facts], [[1] for _ in self.valid_facts]
            else:
                queries, labels = [[q] for q in self.test_facts], [[1] for _ in self.test_facts]
            return KGCEvalDataset(queries=queries, labels=labels,
                                  num_negatives=number_negatives,
                                  known_facts=self.ground_facts_set,
                                  constant2domain=self.constant2domain,
                                  domain2constants=self.domain2constants,
                                  corrupt_mode=corrupt_mode)


if __name__ == "__main__":
    pass
