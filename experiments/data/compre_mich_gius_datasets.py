import os 
import sys
import re
import random
from typing import Tuple

from  data.verify_dataset import get_constants_predicates_queries, get_domain2constants, add_domain_to_locIn

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