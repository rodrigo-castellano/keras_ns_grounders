import os 
import sys
import re
import random
from typing import Tuple

# Check that all the countries have their locatedInCR, neighborOf and remove all locatedInCS


# For that, first get the set of constants
root = './experiments/data/countries_ablation/'
train_path = root+'train.txt'
val_path = root+'valid.txt'
test_path = root+'test.txt'
dataset_path = root + 'dataset.txt'
ctes_path = root+'domain2constants.txt'
predicates_path = root+'relations.txt'


def write_queries_to_file(queries, path):
    with open(path, 'w') as f:
        for query in queries:
            f.write(query[0]+'('+query[1]+','+query[2]+').\n')

def get_constants_predicates_queries(dataset_path, ctes_path):
    '''Get the set of constants, predicates and queries from the dataset'''
    constants = set()
    predicates = set()
    queries = set()
    with open(dataset_path, 'r') as f: 
        for line in open(dataset_path, 'r'):
            predicate = line.split('(')[0]
            consant1 = line.split('(')[1].split(',')[0]
            consant2 = line.split('(')[1].split(',')[1].split(')')[0]
            constants.add(consant1)
            constants.add(consant2)
            predicates.add(predicate)
            queries.add((predicate, consant1, consant2))

    # if the constants are numbers, order them by value
    # if they are strings, order them alphabetically. 
    # Order the queries by predicates and then by the first constant
    constants = sorted(constants, key=lambda x: int(x) if x.isdigit() else x)
    predicates = sorted(predicates, key=lambda x: int(x) if x.isdigit() else x)
    queries = sorted(queries, key=lambda x: (x[0], x[1], x[2]))
    return constants, predicates, queries

constants, predicates, queries = get_constants_predicates_queries(dataset_path, ctes_path)
print('queries:', len(queries))
print(len(constants),'constants, predicates: ', predicates,'\n')


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

    # print the len of each domain
    for domain in domain2constants:
        print(domain, len(domain2constants[domain]))
    print('')
    return domain2constants

domain2constants = get_domain2constants(ctes_path)


# Check that all the constants from domain are in train, and later check that all the constants from train are in domains
# Check that in train.txt all the countries have the locatedInCR, locatedInCS, locatedInSR and neighborOf predicates

def check_constants_in_domain(constants, domain2constants):
    ''' Check that:
        1. there are 23 queries locatedInSR
        2. all countries have locatedInCS, locatedInCR and neighborOf predicates. 
        3. The countries that do not have neighborOf are islands
    '''
    countries = domain2constants['countries']

    # 0. Check that no query has both constants the same
    repeated_constants = {query for query in queries if query[1] == query[2]}
    if repeated_constants:
        raise ValueError('There are queries with the same constant:', repeated_constants)

    # 1. SR: its count should be 23
    count = sum(1 for query in queries if query[0] == 'locatedInSR')
    if count != 23:
        raise ValueError('There are not 23 locatedInSR queries:', count)

    # 2. CS: every country should have a locatedInCS predicate
    countries_in_CS_queries = {query[1] for query in queries if query[0] == 'locatedInCS'}
    missing_countries = set(countries) - countries_in_CS_queries
    # for country in missing_countries:
        # print(country, 'not found in locatedInCS')

    # 3. CR: every country should have a locatedInCR predicate
    countries_in_CR_queries = {query[1] for query in queries if query[0] == 'locatedInCR'}
    missing_countries = set(countries) - countries_in_CR_queries
    # for country in missing_countries:
    #     print(country, 'not found in locatedInCR')


    # 4. neighborOf: every country should have a neighborOf predicate, except for the islands
    # print('\n')
    neighboring_countries = set(query[1] for query in queries if query[0] == 'neighborOf') | set(query[2] for query in queries if query[0] == 'neighborOf')
    islands = set(countries) - neighboring_countries

    # for country in islands:
    #     print(country, 'not found in neighborOf (island)')

    # 5. get the queries with LocatedInCR and an island as the first constant
    islands_CR_queries = {query for query in queries if query[0] == 'locatedInCR' and query[1] in islands}
    # print('islands:', len(islands_CR_queries), islands_CR_queries)

    return islands, islands_CR_queries

islands, islands_CR_queries = check_constants_in_domain(constants, domain2constants)




'''
Get the dataset d1, whereby using the rule LocatedInCR(x,z) => LocatedInCR(w,z), NeighborOf(x,w),
we can prove all queries in val and test by using the queries in train 
    - data: set of queries without locatedInCS
    - train, val, test
    - islands: set of countries that are islands (no neighborOf query)
    - islands_queries: set of queries with LocatedInCR and an island as the first constant
Split the CR queries in 80/10/10 for train/val/test, and add the rest of the non-CR queries to train
'''
def remove_locatedInCS_locatedInSR(queries):
    return {query for query in queries if query[0] != 'locatedInCS' and query[0] != 'locatedInSR'}

print('data with locatedInCS:', len(queries))
data = remove_locatedInCS_locatedInSR(queries)
print('data without locatedInCS:', len(data))



# def split_queries_by_CR(data, islands_CR_queries):
#     '''Split the CR queries in 80/10/10 for train/val/test, and add the rest of the non-CR queries to train'''
#     train, val, test = set(), set(), set()

#     CR_queries = {query for query in data if query[0] == 'locatedInCR'}
#     CR_queries_without_islands = list(CR_queries - islands_CR_queries)

#     # shuffle the queries, except for the islands. Put them at the beginning and shuffle the rest
#     random.shuffle(CR_queries_without_islands)
#     CR_queries = list(islands_CR_queries) + CR_queries_without_islands


#     # split the CR_queries in 80/10/10
#     n = len(CR_queries)
#     train_size = int(n*0.8)
#     val_size = int(n*0.1)
#     test_size = int(n*0.1)


#     # split the CR_queries
#     train = set(CR_queries[:train_size])
#     val = set(CR_queries[train_size:train_size+val_size])
#     test = set(CR_queries[train_size+val_size:])
#     # add to the train the rest of the queries
#     train = train.union(data - set(CR_queries))
#     return train, val, test

# train, val, test = split_queries_by_CR(data, islands_CR_queries)
# print('\nlen train:', len(train), 'len val:', len(val), 'len test:', len(test), 'total:', len(train)+len(val)+len(test),'len data:', len(data))

 

# def has_neighbor_locatedInCR(query, data):
#     """
#     Checks if a query has a neighbor in the data that also has a 'locatedInCR' query.

#     Args:
#         query: A tuple representing the query.
#         data: A list of queries.

#     Returns:
#         A tuple of two boolean values:
#             - `has_neighbor`: True if the query has a neighbor in the data.
#             - `has_locatedInCR`: True if the neighbor has a 'locatedInCR' query in the data.
#     """

#     country = query[1]
#     neighbors = {query[1] for query in data if query[0] == 'neighborOf' and (query[1] == country or query[2] == country)}

#     for neighbor in neighbors:
#         for data_query in data:
#             if data_query[0] == 'locatedInCR' and data_query[1] == neighbor:
#                 return True, True

#     return bool(neighbors), False



# def test_d1_v2(train: set, val_test: list) -> bool:
#     """
#     Checks if a set of queries (val_test) have neighbors in the training data that also have 'locatedInCR' queries.

#     Args:
#         val_test: A set of queries to be checked.
#         train: A list of training queries.

#     Returns:
#         A list of tuples, where each tuple represents a query and its corresponding
#         (has_neighbor, has_locatedInCR) result.
#     """
#     neighbor_queries = set(query for query in train if query[0] == 'neighborOf')

#     for query in val_test:
#         country = query[1]
#         # print('country:',country)

#         n1 = {query_[1] for query_ in neighbor_queries if query_[2] == country}
#         n2 = {query_[2] for query_ in neighbor_queries if query_[1] == country}
#         if country in n1 or country in n2:
#             raise ValueError('Country in neighbors',country,n1,n2)
#         neighbors = n1 | n2
#         # print('n1:',n1)
#         # print('n2:',n2)
#         # print('neighbors:',neighbors)
#         # neighbors_q = {query_ for query_ in neighbor_queries if query_[1] == country} | {query_ for query_ in neighbor_queries if query_[2] == country}
#         # print('NE queries:', neighbors_q)
        
#         neighbor_has_locatedInCR = False
#         for neighbor in neighbors:
#             # has_neighbor_locatedInCR = any(data_query[0] == 'locatedInCR' and data_query[1] == neighbor for data_query in train)    
#             # print('     neighbor:',neighbor)
#             for data_query in train:
#                 if data_query[0] == 'locatedInCR' and data_query[1] == neighbor:
#                     # print('         has neighbor locatedInCR:',data_query)
#                     neighbor_has_locatedInCR = True
#                     break
            
#         if not neighbor_has_locatedInCR:
#             # print('         has no neighbor locatedInCR','!'*20)
#             return False
#     return True

# def test_d1_v1(train, val_test):
#     """
#     Checks if validation and test queries i) are not in the training set, ii) have neighbors and iii) those neighbors have 'locatedInCR' queries.

#     Args:
#         train: Set of training queries.
#         val: Set of validation queries.
#         test: Set of test queries.
#     """
#     test_passed = True
#     # Create a dictionary to store 'locatedInCR' queries by country
#     located_in_cr_queries = {query[1]: query for query in train if query[0] == 'locatedInCR'}

#     # Create a set of 'neighborOf' queries for faster membership checks
#     neighbor_queries = set(query for query in train if query[0] == 'neighborOf')

#     for query in val_test:
#         if query in train: # i)
#             return False
#             # test_passed = False
#             # print(f"Query {query} is in the training set.")
#             # continue

#         country = query[1]
#         if not any(country in query_ for query_ in neighbor_queries): # ii)
#             return False
#             # test_passed = False
#             # print(f"Query {query} has no neighbor in the training set.")
#             # continue


#         for neighbor_query in neighbor_queries: # iii)
#             if country in neighbor_query: 
#                 other_country = neighbor_query[1] if neighbor_query[1] != country else neighbor_query[2]
#                 if other_country not in located_in_cr_queries:
#                     return False
#                     # test_passed = False
#                     # print(f"Query {query} has neighbor {neighbor_query} but no 'locatedInCR' query for {other_country}")
#                     # break
#     return test_passed

from tqdm import tqdm

def test_d1(train: set, val_test: list) -> bool:
    """
    Checks if each query in val_test has neighbors in the training data, and those neighbors have 'locatedInCR' queries.
    """
    # Extract 'locatedInCR' queries for quick lookup
    located_in_cr_by_country = {query[1]: query for query in train if query[0] == 'locatedInCR'}
    
    # Extract 'neighborOf' queries into a set
    neighbor_queries = set(query for query in train if query[0] == 'neighborOf')

    for query in val_test:
        if query in train:
            return False  # Query shouldn't be in training data

        country = query[1]

        # Check for neighbors of `country`
        neighbors = {
            query_[1] if query_[2] == country else query_[2]
            for query_ in neighbor_queries
            if country in query_
        }

        # Check if any neighbor has a 'locatedInCR' entry
        if not any(neighbor in located_in_cr_by_country for neighbor in neighbors):
            return False  # Neighbor doesn't have 'locatedInCR' query

    return True

def get_d1_dataset(data: set, islands_CR_queries: set) -> Tuple[set, set, set]:
    '''
    Get the d1 dataset. I want to remove CR queries from train and add them to val_test data in a 80/10/10 split.
    All the CR queries in val and test need to have a neighbour in train, and that neighbour needs to have a LocatedInCR query in train
    The way to do it in this case is to do a maximum of 10^6 iters or until we complete the length of val_test set (val_test is len_val+len_test)
    - generate a random permutation of the CR queries, and for each query in the permutation, remove it from train and add it to val_test
    - for every query in val_test, check that it has a neighbour in train, and that neighbour has a LocatedInCR query in train
        - if it  not the case, break the while loop and start again with another permutation
    '''
    CR_queries = {query for query in data if query[0] == 'locatedInCR'}
    CR_queries_without_islands = list(CR_queries - islands_CR_queries)
    Ne_queries = {query for query in data if query[0] == 'neighborOf'}

    n = len(CR_queries)
    train_size = int(n*0.8)
    val_size = int(n*0.1)
    test_size = n - train_size - val_size
    val_test_size = val_size + test_size
    print('\nCR queries. Total:', n,', distributed in - train_size:', train_size, 'val_size:', val_size, 'test_size:', test_size, 'val_test_size:', val_test_size, )
    print('data without CR queries:', len(data - CR_queries))
    print('data:', len(data))
    max_iters = 2000
    iter = 0
    max_len_val_test = 0

    with tqdm(total=max_iters, desc="Processing iterations") as pbar:
        while iter < max_iters and max_len_val_test<val_test_size:
            val_test = []
            train = CR_queries.copy()
            random.shuffle(CR_queries_without_islands)
            for query in CR_queries_without_islands:
                train_set = train - {query}
                # if test_d1_v2(train_set | Ne_queries, val_test + [query]) != test_d1_v3(train_set| Ne_queries, val_test + [query]):
                #     raise ValueError('Test failed')                
                if not test_d1(train_set | Ne_queries, val_test + [query]):
                    break  # Restart if criteria are not met
                train.remove(query)
                val_test.append(query)
            
            if len(val_test) > max_len_val_test:
                max_len_val_test = len(val_test)
                best_train, best_val_test = set(train), set(val_test)
                print('\niter',iter,'max_len_val_test:', max_len_val_test,'/',val_test_size)
            iter += 1
            pbar.update(1)
    train = best_train.union(data - CR_queries) # append the rest of the non-CR queries. 
    test = set(list(best_val_test)[:test_size]) # append best_val_test up to test_size
    val = set(list(best_val_test)[test_size:test_size+val_size]) if len(best_val_test) > test_size else set()
    if len(best_val_test) > test_size+val_size: # If there are too many queries in val_test, add the rest to train
        train = train.union(list(best_val_test)[test_size+val_size:])

    print('New distribution: len train(with non-CR):', len(train), 'len val:', len(val), 'len test:', len(test), 'total:', len(train)+len(val)+len(test))
    print('Keeps the length:', len(data)==len(train)+len(val)+len(test))
  
   
    return train, val, test

# train, val, test = get_d1_dataset(data, islands_CR_queries)


# passed = test_d1(train, val | test)
# print('Final test passed:', passed)


# dataset_type = 'd1'
# write_queries_to_file(train, root+'train_'+dataset_type+'.txt')
# write_queries_to_file(val, root+'val_'+dataset_type+'.txt')
# write_queries_to_file(test, root+'test_'+dataset_type+'.txt')




def get_neighbors(country: str, data: set) :
    '''Get the neighbors of a country in the data set'''
    return {query[1] if query[2] == country else query[2] 
            for query in data 
            if query[0] == 'neighborOf' and (query[1] == country or query[2] == country)}

def get_neighbors_queries(countries: set, data: set):
    '''For every country, get the neighbors countries in a list'''
    neighbors = []
    for country in countries:
        neighbors.append(get_neighbors(country, data))
    return list(neighbors)

def get_locatedInCR(country: str, data: set) -> bool:
    '''Check if the country has a LocatedInCR query in the data set'''
    return any(query[0] == 'locatedInCR' and query[1] == country for query in data)

def get_locatedInCR_queries(countries: set, data: set):
    '''Return the countries that have LocatedInCR query in the data set'''
    locatedInCR = []
    for country in countries:
        locatedInCR.append(get_locatedInCR(country, data))
    # return the countries where the LocatedInCR query is True
    return {country for i, country in enumerate(countries) if locatedInCR[i]}


def q_remove_from_train(train: set, Ne_queries: set,  country: str) -> set:
    '''For that country, return the LocatedInCR queries to remove'''
    # print('country:',country)   
    neighbors = get_neighbors(country, Ne_queries)
    # print('neighbors:',neighbors)
    ne_with_cr = get_locatedInCR_queries(neighbors, train)
    # print('ne_with_cr:',ne_with_cr)
    # print('q_remove:',{query for query in train if query[0] == 'locatedInCR' and query[1] in ne_with_cr})
    return {query for query in train if query[0] == 'locatedInCR' and query[1] in ne_with_cr}

def test_d2(train: set, Ne_queries: set, val_test: list) -> bool:
    """
    Checks if each query in val_test has neighbors in the training data with no CR queries, and the neighbors of those neighbors have CR queries.
    For every country in val_test, check that:
        - the country has no neighbors in train
        - the neighbors of the country have no LocatedInCR queries in train
        - the neighbors of the neighbors of the country have LocatedInCR queries in train
    """
    # Extract 'locatedInCR' queries for quick lookup
    located_in_cr_by_country = {query[1]: query for query in train if query[0] == 'locatedInCR'}

    for query in val_test:
        # 1. Query shouldn't be in training data
        if query in train:
            # print('query in train')
            return False 

        country = query[1]

        # 2. 
        # Check for neighbors of `country`
        neighbors = get_neighbors(country, Ne_queries)
        neighbors_with_cr = get_locatedInCR_queries(neighbors, train)
        # print('neighbors:',neighbors)
        # print('neighbors with CR:',neighbors_with_cr)
        # Check if any neighbor has a 'locatedInCR' entry
        if any(neighbors_with_cr):            
            # print('at least one neighbor has locatedInCR',country,neighbors,neighbors_with_cr)
            return False  # Neighbor doesn't have 'locatedInCR' query
        
        # 3.
        # Get the neighbors of the neighbors
        neighbors_of_neighbors = [get_neighbors(ne, Ne_queries) for ne in neighbors]
        # substract the country from the neighbors
        neighbors_of_neighbors = [ne_set-{country} for ne_set in neighbors_of_neighbors]
        # print('neighbors of neighbors:',neighbors_of_neighbors)

        # Check if any neighbor of the neighbors has a 'locatedInCR' entry
        neighbors_of_neighbors_with_cr = [get_locatedInCR_queries(ne_set, train) for ne_set in neighbors_of_neighbors]
        # print('neighbors of neighbors with CR:',neighbors_of_neighbors_with_cr)
        if not any(neighbors_of_neighbors_with_cr):
            # print('no neighbor of the neighbors has locatedInCR',country,neighbors,neighbors_of_neighbors,neighbors_of_neighbors_with_cr)
            return False

    return True


def get_d2_dataset(data: set, islands_CR_queries: set) -> Tuple[set, set, set]:
    '''
    Get the d1 dataset. 
    c,c1,c2,.. are countries
    NE, CR are queries. 
    NE is a neighbour query, NE_c1_c2 is a neighbour query that contains c1,c2 (no matter the order)
    CR is a LocatedInCR query, CR_c is a LocatedInCR query that contains c
    q_c is a CR_c query in val_test

    Rule:    ∀ q_ci in val_test, ∀ NE_ci(∄ CR_ci, ∃NE_ci_cj(∃CR_cj, )). 
    Meaning that for every CR query in val_test, all the neighbour queries in train that 
        i.i) do not have a CR query in train, i.ii) have a neighbour query in train that ii) has a CR query in train. 
    - I need a function that given queries/countries and trainset, it returns its NeighborOf queries in train / its list of neighbors
    - I need a function that given queries/countries and trainset, it returns its LocatedInCR queries in train / bool: if they have LocatedInCR queries
    Algo: 
    - while iter < max_iters and max_len_val_test<val_test_size:
        - start a val_test=[] and train. Generate a random permutation (to change)
            - for query in CR_queries_without_islands:
                - atoms_remove: get the CR atoms to remove from train
                - updated_train_test = train - {atoms_remove} - {query}
                - Updated_val_test = val_test + query
                - check if the updated_val_test passes the test_d1 function
    '''
    CR_queries = {query for query in data if query[0] == 'locatedInCR'}
    CR_queries_without_islands = list(CR_queries - islands_CR_queries)
    Ne_queries = {query for query in data if query[0] == 'neighborOf'}

    n = len(CR_queries)
    train_size = int(n*0.8)
    val_size = int(n*0.1)
    test_size = n - train_size - val_size
    val_test_size = val_size + test_size
    print('\nCR queries. Total:', n,', distributed in - train_size:', train_size, 'val_size:', val_size, 'test_size:', test_size, 'val_test_size:', val_test_size, )
    print('data without CR queries:', len(data - CR_queries))
    print('data:', len(data),'\n')
    max_iters = 2000
    iter = 0
    max_len_val_test = 0

    with tqdm(total=max_iters, desc="Processing iterations") as pbar:
        while iter < max_iters and max_len_val_test<val_test_size:
            # print('\n')
            val_test = []
            train = CR_queries.copy()
            random.shuffle(CR_queries_without_islands)
            for query in CR_queries_without_islands:
                # print('\nquery:',query)
                country = query[1]
                # Get the train queries to remove, and the neighbors whose CR have been removed
                q_remove = q_remove_from_train(train, Ne_queries, country)

                # do the test with the updated train and val_test
                train_set = train - {query} - q_remove
                if not test_d2(train_set, Ne_queries, val_test + [query]):
                    break
                train -= {query} | q_remove
                val_test.append(query)

            if len(val_test) > max_len_val_test:
                max_len_val_test = len(val_test)
                best_train, best_val_test = set(train), set(val_test)
                print('\niter',iter,'max_len_val_test:', max_len_val_test,'/',val_test_size)
            iter += 1
            pbar.update(1)
    train = best_train.union(data - CR_queries) # append the rest of the non-CR queries. 
    test = set(list(best_val_test)[:test_size]) # append best_val_test up to test_size
    val = set(list(best_val_test)[test_size:test_size+val_size]) if len(best_val_test) > test_size else set()
    if len(best_val_test) > test_size+val_size: # If there are too many queries in val_test, add the rest to train
        train = train.union(list(best_val_test)[test_size+val_size:])

    print('New distribution: len train(with non-CR):', len(train), 'len val:', len(val), 'len test:', len(test), 'total:', len(train)+len(val)+len(test))
    print('Keeps the length:', len(data)==len(train)+len(val)+len(test),'. Decrease:',len(data)-(len(train)+len(val)+len(test)))
  
   
    return train, val, test


train, val, test = get_d2_dataset(data, islands_CR_queries)
Ne_queries = {query for query in data if query[0] == 'neighborOf'}
passed = test_d2(train, Ne_queries, val | test)
print('Final test passed:', passed)


dataset_type = 'd2'
write_queries_to_file(train, root+'train_'+dataset_type+'.txt')
write_queries_to_file(val, root+'val_'+dataset_type+'.txt')
write_queries_to_file(test, root+'test_'+dataset_type+'.txt')

