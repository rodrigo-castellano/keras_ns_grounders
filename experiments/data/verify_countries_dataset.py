import os 
import sys
import re
import random
from typing import Tuple
from tqdm import tqdm
# Check that all the countries have their locatedInCR, neighborOf and remove all locatedInCS


# For that, first get the set of constants
root = './experiments/data/countries_s1/'
train_path = root+'train.txt'
val_path = root+'valid.txt'
test_path = root+'test.txt'
# dataset_path = root + 'dataset.txt'
ctes_path = root+'domain2constants.txt'
predicates_path = root+'relations.txt'

def write_queries_to_file(queries, path):
    with open(path, 'w') as f:
        for query in queries:
            f.write(query[0]+'('+query[1]+','+query[2]+').\n')

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

def remove_locatedInCS_locatedInSR(queries):
    return {query for query in queries if query[0] != 'locatedInCS' and query[0] != 'locatedInSR'}

def add_domain_to_locIn(train, val, test, domain2constants):
    '''Add the domain to the locatedIn predicates'''

    def find_predicate(query,countries, regions, subregions):
        c1, c2 = query[1], query[2]
        if c1 in countries and c2 in regions:
            return 'locatedInCR'
        elif c1 in countries and c2 in subregions:
            return 'locatedInCS'
        elif c1 in subregions and c2 in regions:
            return 'locatedInSR'
        else:
            raise ValueError('The query', query, 'has a wrong constant')
    
    countries = domain2constants['countries']
    regions = domain2constants['regions']
    subregions = domain2constants['subregions']
    train_q,val_q,test_q = [],[],[]
    for query in train:
        if query[0] == 'locatedIn':
            predicate = find_predicate(query,countries, regions, subregions)
            train_q.append((predicate, query[1], query[2]))
        else:
            train_q.append(query)
    for query in val:
        if query[0] == 'locatedIn':
            predicate = find_predicate(query,countries, regions, subregions)
            val_q.append((predicate, query[1], query[2]))
        else:
            val_q.append(query)
    for query in test:
        if query[0] == 'locatedIn':
            predicate = find_predicate(query,countries, regions, subregions)
            test_q.append((predicate, query[1], query[2]))     
        else:
            test_q.append(query)
            
    return train_q, val_q, test_q

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

def print_query(query):
    print(query[0]+'('+query[1]+','+query[2]+')')

def check_properties_of_dataset(constants, domain2constants,queries):
    ''' Check that:
        1. there are 23 queries locatedInSR
        2. all countries have locatedInCS, locatedInCR and neighborOf predicates. 
        3. The countries that do not have neighborOf are islands
    '''
    countries = domain2constants['countries']
    regions = domain2constants['regions']
    subregions = domain2constants['subregions']

    # check if there is any repeated query
    repeated_queries = [query for query in queries if queries.count(query) > 1]
    if repeated_queries:
        raise ValueError('There are repeated queries:', [print_query(query) for query in repeated_queries])
        # raise ValueError('There are repeated queries:', [print_query(query) for query in repeated_queries],sep='\n')

    # 0. Check that no query has both constants the same
    repeated_constants = {query for query in queries if query[1] == query[2]}
    if repeated_constants:
        raise ValueError('There are queries with the same constant:', repeated_constants)

    # CHECK CORRECT DOMAINS
    # In the queries, neighbors have countries in the first and second constants
    for query in queries:
        if query[0] == 'neighborOf' and (query[1] not in countries or query[2] not in countries):
            raise ValueError('The query', query, 'has a wrong constant')
    
    # In the queries, locatedInCR has a country in the first constants, and a region in the second constant
    for query in queries:
        if query[0] == 'locatedInCR' and (query[1] not in countries or query[2] not in regions):
            raise ValueError('The query', query, 'has a wrong constant')
    
    # In the queries, locatedInSR has a subregion in the first constants, and a region in the second constant
    for query in queries:
        if query[0] == 'locatedInSR' and (query[1] not in subregions or query[2] not in regions):
            raise ValueError('The query', query, 'has a wrong constant')
    
    # In the queries, locatedInCS has a country in the first constants, and a subregion in the second constant
    for query in queries:
        if query[0] == 'locatedInCS' and (query[1] not in countries or query[2] not in subregions):
            raise ValueError('The query', query, 'has a wrong constant')

    # In the queries, check that neighbour has always symmetric relations, e.g., if A is neighbor of B, then B is neighbor of A
    for query in queries:
        if query[0] == 'neighborOf':
            c1 = query[1]
            c2 = query[2]
            symmetric_query = ('neighborOf', c2, c1)    
            if symmetric_query not in queries:
                print('Symmetric query not found:', query)

    # 1. SR: its count should be 23
    count = sum(1 for query in queries if query[0] == 'locatedInSR')
    if count != 23:
        raise ValueError('There are not 23 locatedInSR queries:', count)
    # for every subregion, there should be a locatedInSR query
    subregions_in_SR_queries = {query[1] for query in queries if query[0] == 'locatedInSR'}
    missing_subregions = set(subregions) - subregions_in_SR_queries
    if missing_subregions:
        raise ValueError('There are missing subregions in locatedInSR:', missing_subregions)

    # 2. CS: every country should have a locatedInCS predicate
    countries_in_CS_queries = {query[1] for query in queries if query[0] == 'locatedInCS'}
    missing_countries = set(countries) - countries_in_CS_queries
    for country in missing_countries:
        print(country, 'not found in locatedInCS')
    print()

    # 3. CR: every country should have a locatedInCR predicate
    countries_in_CR_queries = {query[1] for query in queries if query[0] == 'locatedInCR'}
    missing_countries = set(countries) - countries_in_CR_queries
    for country in missing_countries:
        print(country, 'not found in locatedInCR')
        # print('locatedInCR('+str(country)+',)')
    print()

    # 4. neighborOf: every country should have a neighborOf predicate, except for the islands
    # print('\n')
    neighboring_countries = set(query[1] for query in queries if query[0] == 'neighborOf') | set(query[2] for query in queries if query[0] == 'neighborOf')
    islands = set(countries) - neighboring_countries

    for country in islands:
        print(country, 'not found in neighborOf (island)')

    # 5. get the queries with LocatedInCR and an island as the first constant
    islands_CR_queries = {query for query in queries if query[0] == 'locatedInCR' and query[1] in islands}
    # print('islands:', len(islands_CR_queries), islands_CR_queries)

    return islands, islands_CR_queries


def check_constants_in_domain(constants_train, constants_val, constants_test, constants,domain2constants):
    # 1. check that all test,val constants are in train
    print('\ntest - train:', constants_test - constants_train)
    print('val - train:', constants_val - constants_train)

    # 2. check that all test,val,train constants are in the domain2constants
    all_domain2constants = {constant  for domain, constants in domain2constants.items() for constant in constants}
    print('\ntest - domain:', constants_test - all_domain2constants)
    print('val - domain:', constants_val - all_domain2constants)
    print('train - domain:', constants_train - all_domain2constants)
    print('\ndomain-train,val,test:', all_domain2constants - constants)
    print('\ntrain,val,test-domain:', constants-all_domain2constants)

# Get the constants, predicates and queries from the dataset
constants_train, predicates_train, train = get_constants_predicates_queries(train_path)
constants_val, predicates_val, val = get_constants_predicates_queries(val_path)
constants_test, predicates_test, test = get_constants_predicates_queries(test_path)
constants =  constants_train | constants_val | constants_test

domain2constants = get_domain2constants(ctes_path)

# modify locIn to add a domain
train,val,test = add_domain_to_locIn(train, val, test, domain2constants)
queries_all = train + val + test

# make sure that the constants  in train, val and test are in the domain2constants, and all the way round. 
# check all test and val queries are in train
check_constants_in_domain(constants_train, constants_val, constants_test, constants, domain2constants)

check_properties_of_dataset(constants, domain2constants, queries_all)

# s1: remove all test queries in train. remove all valid queries in train. check that all test,valid queries can be proven with s1 rule

