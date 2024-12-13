import os 
import sys
import re
import random
from typing import Tuple


def write_queries_to_file(queries, path):
    with open(path, 'w') as f:
        for query in queries:
            f.write(query[0]+'('+query[1]+','+query[2]+').\n')

def get_domain2constants(ctes_path: str) -> dict:
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

def add_domain_to_locIn(data: list, domain2constants: dict) -> list:
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
    data_q = []
    for query in data:
        if query[0] == 'locatedIn':
            predicate = find_predicate(query,countries, regions, subregions)
            data_q.append((predicate, query[1], query[2]))
        else:
            data_q.append(query)
            
    return data_q

def get_constants_predicates_queries(dataset_path: str) -> Tuple[set, set, list]:
    '''Get the set of constants, predicates and queries from the dataset'''
    constants = set()
    predicates = set()
    queries = []
    with open(dataset_path, 'r') as f: 
        for line in open(dataset_path, 'r'):
            predicate = line.split('(')[0]
            constant1 = line.split('(')[1].split(',')[0]
            constant2 = line.split('(')[1].split(',')[1].split(')')[0]
            constants.add(constant1)
            constants.add(constant2)
            predicates.add(predicate)
            queries.append((predicate, constant1, constant2))

    # if the constants are numbers, order them by value
    # if they are strings, order them alphabetically. 
    # Order the queries by predicates and then by the first constant
    constants = sorted(constants, key=lambda x: int(x) if x.isdigit() else x)
    predicates = sorted(predicates, key=lambda x: int(x) if x.isdigit() else x)
    queries = sorted(queries, key=lambda x: (x[0], x[1], x[2]))
    return set(constants), set(predicates), queries

def print_query(query):
    print(query[0]+'('+query[1]+','+query[2]+')')

def check_properties_of_dataset(domain2constants: dict, queries: list) -> Tuple[set, set]:
    ''' Check that:
        1. there are 23 queries locatedInSR
        2. all countries have locatedInCS, locatedInCR and neighborOf predicates. 
        3. The countries that do not have neighborOf are islands
    '''
    countries = domain2constants['countries']
    regions = domain2constants['regions']
    # check if there are subregions
    if 'subregions' in domain2constants:
        subregions = domain2constants['subregions']
    else:
        print('No subregions found in the dataset')
        subregions = None

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
    if subregions:
        for query in queries:
            if query[0] == 'locatedInSR' and (query[1] not in subregions or query[2] not in regions):
                raise ValueError('The query', query, 'has a wrong constant')
    
    # In the queries, locatedInCS has a country in the first constants, and a subregion in the second constant
    if subregions:
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
    if subregions:
        count = sum(1 for query in queries if query[0] == 'locatedInSR')
        if count != 23:
            raise ValueError('There are not 23 locatedInSR queries:', count)
        # for every subregion, there should be a locatedInSR query
        subregions_in_SR_queries = {query[1] for query in queries if query[0] == 'locatedInSR'}
        missing_subregions = set(subregions) - subregions_in_SR_queries
        if missing_subregions:
            raise ValueError('There are missing subregions in locatedInSR:', missing_subregions)

    # 2. CS: every country should have a locatedInCS predicate
    if subregions:
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
    neighboring_countries = set(query[1] for query in queries if query[0] == 'neighborOf') | set(query[2] for query in queries if query[0] == 'neighborOf')
    islands = set(countries) - neighboring_countries

    # for country in islands:
        # print(country, 'not found in neighborOf (island)')

    # 5. get the queries with LocatedInCR and an island as the first constant
    islands_CR_queries = {query for query in queries if query[0] == 'locatedInCR' and query[1] in islands}
    # print('islands:', len(islands_CR_queries), islands_CR_queries)

    return islands, islands_CR_queries


def check_constants_in_domain(constants,domain2constants,constants_train=None,constants_val=None,constants_test=None):
    '''make sure that the constants  in train, val and test are in the domain2constants, 
    and all the way round. check all test and val queries are in train'''
    # 1. check that all test,val constants are in train
    if constants_train is not None and constants_val is not None and constants_test is not None:
        print('\ntest - train:', constants_test - constants_train)
        print('val - train:', constants_val - constants_train)

    # 2. check that all test,val,train constants are in the domain2constants
    all_domain2constants = {constant  for domain, constants in domain2constants.items() for constant in constants}
    # print('\ntest - domain:', constants_test - all_domain2constants)
    # print('val - domain:', constants_val - all_domain2constants)
    # print('train - domain:', constants_train - all_domain2constants)
    print('\ndomain - dataset constants:', all_domain2constants - constants)
    print('dataset constants - domain:', constants-all_domain2constants)
    print()


def get_neighbors(country: str, data: set) :
    '''Get the neighbors of a country in the data set'''
    return {query[1] if query[2] == country else query[2] 
            for query in data 
            if query[0] == 'neighborOf' and (query[1] == country or query[2] == country)}


def get_neighbors_from_countries(countries: set, data: set):
    """For every country, get the neighbors countries in a list"""
    return [get_neighbors(country, data) for country in countries]
        

def get_locatedInCR(country: str, data: set) -> bool:
    '''Check if the country has a LocatedInCR query in the data set'''
    return any(query[0] == 'locatedInCR' and query[1] == country for query in data)

def get_locatedInCR_from_countries(countries: set, data: set):
    """Return the countries that have LocatedInCR query in the data set"""
    return {country for country in countries if get_locatedInCR(country, data)}


def s1_condition(val_test, train):
    '''All the val,test queries need to have a locInCS query in train with a locInSR query in train'''
    print('s1 condition...')
    # countries that appear in a CS query in train, and their subregions
    countries_in_CS_queries = {query[1]: query[2] for query in train if query[0] == 'locatedInCS'}
    # subregions that appear in a SR query in train
    subregions_in_SR_queries = {query[1] for query in train if query[0] == 'locatedInSR'}
    for query in val_test:
        assert query[0] == 'locatedInCR', f'query {query} is not locatedInCR'
        country = query[1]
        subregion = countries_in_CS_queries[country]
        if not subregion:
            raise ValueError('The query', query, 'does not have a locatedInCS query in train')
        if subregion not in subregions_in_SR_queries:
            raise ValueError('The subregion', subregion, 'does not have a locatedInSR query in train')

def s2_condition(val_test, train):
    '''All the val,test queries need to have a neighborOf query in train with a locatedInCR query'''
    print('s2 condition...')
    Ne_queries = {query for query in train if query[0] == 'neighborOf'}
    for query in val_test:
        country = query[1]
        neighbors = get_neighbors(country, Ne_queries)
        if not neighbors:
            raise ValueError('The country', country, 'has no neighbors in train')
        neighbors_with_cr = get_locatedInCR_from_countries(neighbors, train)
        if not neighbors_with_cr:
            raise ValueError('The neighbors of', country, 'have no locatedInCR queries in train')
    return True
    
def s3_condition(val_test, train):
    '''All the val,test queries with a neighborOf query in train: remove its LocatedInCR in train and make sure the 2nd neigh has a locatedInCR in train'''
    print('s3 condition...')
    Ne_queries = {query for query in train if query[0] == 'neighborOf'}
    for query in val_test:
        country = query[1]
        neighbors = get_neighbors(country, Ne_queries)
        if not neighbors:
            raise ValueError('The country', country, 'has no neighbors in train')
        neighbors_with_cr = get_locatedInCR_from_countries(neighbors, train)
        if neighbors_with_cr:
            raise ValueError('The neighbors of', country, 'have no locatedInCR queries in train')
        # Get the neighbors of the neighbors
        neighbors_of_neighbors = [get_neighbors(ne, Ne_queries) for ne in neighbors]
        # substract the country from the neighbors
        neighbors_of_neighbors = [ne_set-{country} for ne_set in neighbors_of_neighbors]
        if not neighbors_of_neighbors:
            raise ValueError('no neighbors_of_neighbors',country,neighbors,neighbors_of_neighbors)
        # Check if any neighbor of the neighbors has a 'locatedInCR' entry
        neighbors_of_neighbors_with_cr = [get_locatedInCR_from_countries(ne_set, train) for ne_set in neighbors_of_neighbors]
        if not any(neighbors_of_neighbors_with_cr):
            print('Not provable, no neighbor of the neighbors has locatedInCR:',country)
                            #  ,neighbors,neighbors_of_neighbors,neighbors_of_neighbors_with_cr)
    return True


if __name__ == '__main__':
    # COUNTRIES AND ABLATION DATA
    root = './experiments/data/countries_s3/'
    train_path,val_path,test_path,domain2constants_path = root+'train.txt',root+'valid.txt',root+'test.txt',root+'domain2constants.txt'

    constants_train, predicates_train, train = get_constants_predicates_queries(train_path)
    constants_val, predicates_val, val = get_constants_predicates_queries(val_path)
    constants_test, predicates_test, test = get_constants_predicates_queries(test_path)
    constants =  constants_train | constants_val | constants_test
    domain2constants = get_domain2constants(domain2constants_path)
    dataset = train + val + test
    check_constants_in_domain(constants, domain2constants, constants_train, constants_val, constants_test)
    check_properties_of_dataset(domain2constants, dataset)
    s1_condition(val+test, train)
    s2_condition(val+test, train)
    s3_condition(val+test, train)


    # # GIUSEPPE's data
    # root = './experiments/data/countries_dataset_giuseppe/'
    # dataset_path, domain2constants_path = root + 'dataset.txt', root+'domain2constants.txt'
    # constants, predicates, dataset = get_constants_predicates_queries(dataset_path) 
    # domain2constants = get_domain2constants(domain2constants_path)
    # dataset = add_domain_to_locIn(dataset, domain2constants)
    # check_properties_of_dataset(domain2constants, dataset)


    # # GIUSEPPE's data splitted in train, val, test
    # root = './experiments/data/countries_dataset_giuseppe/'
    # train_path,val_path,test_path,domain2constants_path = root+'train.txt',root+'valid.txt',root+'test.txt',root+'domain2constants.txt'

    # constants_train, predicates_train, train = get_constants_predicates_queries(train_path)
    # constants_val, predicates_val, val = get_constants_predicates_queries(val_path)
    # constants_test, predicates_test, test = get_constants_predicates_queries(test_path)
    # constants =  constants_train | constants_val | constants_test
    # domain2constants = get_domain2constants(domain2constants_path)
    # dataset = train + val + test
    # check_constants_in_domain(constants, domain2constants, constants_train, constants_val, constants_test)
    # check_properties_of_dataset(domain2constants, dataset)

    # root = './experiments/data/countries_dataset_giuseppe/'
    # train_path,val_path,test_path,domain2constants_path = root+'train_s3.txt',root+'valid.txt',root+'test.txt',root+'domain2constants.txt'
    # constants_train, predicates_train, train = get_constants_predicates_queries(train_path)
    # constants_val, predicates_val, val = get_constants_predicates_queries(val_path)
    # constants_test, predicates_test, test = get_constants_predicates_queries(test_path)
    # constants =  constants_train | constants_val | constants_test
    # domain2constants = get_domain2constants(domain2constants_path)
    # dataset = train + val + test
    # check_constants_in_domain(constants_train, constants_val, constants_test, constants, domain2constants)
    # check_properties_of_dataset(domain2constants, dataset)