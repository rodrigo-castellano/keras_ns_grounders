import os 
import sys
import re
import random
from typing import Tuple

def add_domain_to_locIn(queries, domain2constants):
    '''Add the domain to the locatedIn predicates'''
    countries = domain2constants['countries']
    regions = domain2constants['regions']
    subregions = domain2constants['subregions']
    new_q = []
    for query in queries:
        if query[0] == 'locatedIn':
            c1, c2 = query[1], query[2]
            if c1 in countries and c2 in regions:
                predicate = 'locatedInCR'
            elif c1 in countries and c2 in subregions:
                predicate = 'locatedInCS'
            elif c1 in subregions and c2 in regions:
                predicate = 'locatedInSR'
            else:
                raise ValueError('The query', query, 'has a wrong constant')
            new_q.append((predicate, query[1], query[2]))
        else:
            new_q.append(query)
    return new_q

def get_domain2constants(ctes_path):
    '''Get the domain2constants dictionary'''
    domain2constants = {}
    with open(ctes_path, 'r') as f:
        for line in open(ctes_path, 'r'):
            domain = line.split(' ')[0]
            ctes = line.split(' ')[1:]
            # remove the '\n' character
            ctes = [cte.replace('\n', '') for cte in ctes]
            domain2constants[domain] = ctes
    return domain2constants

def get_constants_predicates_queries(dataset_path,):
    '''Get the set of constants, predicates and queries from the dataset'''
    constants = set()
    predicates = set()
    queries = []
    with open(dataset_path, 'r') as f: 
        for line in open(dataset_path, 'r'):
            predicate = line.split('(')[0]
            consant1 = line.split('(')[1].split(',')[0]
            consant2 = line.split('(')[1].split(',')[1].split(')')[0]
            constants.add(consant1)
            constants.add(consant2)
            predicates.add(predicate)
            queries.append((predicate, consant1, consant2))
    # if the constants are numbers, order them by value
    # if they are strings, order them alphabetically. 
    # Order the queries by predicates and then by the first constant
    constants = sorted(constants, key=lambda x: int(x) if x.isdigit() else x)
    predicates = sorted(predicates, key=lambda x: int(x) if x.isdigit() else x)
    queries = sorted(queries, key=lambda x: (x[0], x[1], x[2]))
    return set(constants), set(predicates), queries

root = './experiments/data/countries_ablation/'
train_path = root+'train.txt'
val_path = root+'valid.txt'
test_path = root+'test.txt'
dataset_path = root + 'dataset.txt'

constants_dataset, predicates_dataset, dataset = get_constants_predicates_queries(dataset_path)
dataset = set(dataset)
print('number of queries in dataset ablation:', len(dataset))

root = './experiments/data/countries_michelangelo/countries_s1/'
train_path = root+'train.txt'
val_path = root+'valid.txt'
test_path = root+'test.txt'
dataset_path = root + 'dataset.txt'


constants_train, predicates_train, train = get_constants_predicates_queries(train_path)
constants_val, predicates_val, val = get_constants_predicates_queries(val_path)
constants_test, predicates_test, test = get_constants_predicates_queries(test_path)

dataset_2 = set(train + val + test)
print('number of queries in dataset michelangelo:', len(dataset_2))

print('number of queries not in common:', len(dataset.difference(dataset_2)))


root = './experiments/data/countries_dataset_giuseppe/'
dataset_path = root + 'dataset.txt'
domain2constants_path = root+'domain2constants.txt'

constants_dataset, predicates_dataset, dataset = get_constants_predicates_queries(dataset_path)
domain2constants = get_domain2constants(domain2constants_path)
dataset = add_domain_to_locIn(dataset, domain2constants)
dataset = set(dataset)
print('number of queries in dataset giuseppe:', len(dataset))

print('queries in giuseppe but not in michelangelo:', len(dataset.difference(dataset_2)), dataset.difference(dataset_2)) 