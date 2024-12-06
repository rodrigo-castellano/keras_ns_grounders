import os 
import sys
import re
import random
from typing import Tuple
from tqdm import tqdm


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
    print(str(query[0])+'('+str(query[1])+','+str(query[2])+').')

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

    # CHECK SYMMETRY OF NEIGHBOR RELATION 
    # In the queries, check that neighbour has always symmetric relations, e.g., if A is neighbor of B, then B is neighbor of A
    for query in queries:
        if query[0] == 'neighborOf':
            c1 = query[1]
            c2 = query[2]
            symmetric_query = ('neighborOf', c2, c1)    
            if symmetric_query not in queries:
                print('Symmetric query not found:', query)
                # print_query(symmetric_query)

    # SR: its count should be 23
    count = sum(1 for query in queries if query[0] == 'locatedInSR')
    if count != 23:
        raise ValueError('There are not 23 locatedInSR queries:', count)
    
    # 1. SR: every subregion should have a locatedInSR query
    subregions_in_SR_queries = {query[1] for query in queries if query[0] == 'locatedInSR'}
    missing_subregions = set(subregions) - subregions_in_SR_queries
    if missing_subregions:
        raise ValueError('There are missing subregions in locatedInSR:', missing_subregions)

    # 2. CS: every country should have a locatedInCS query
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
    # for country in islands:
    #     print(country, 'not found in neighborOf (island)')

    # 5. get the queries with LocatedInCR and an island as the first constant
    islands_CR_queries = {query for query in queries if query[0] == 'locatedInCR' and query[1] in islands}
    # print('islands:', len(islands_CR_queries), islands_CR_queries)

    return islands, islands_CR_queries


def check_transitivity(constants_train, constants_val, constants_test, constants,domain2constants):
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


def check_correspondence_domain2constants(constants, domain2constants):
    all_domain2constants = {constant  for domain, constants in domain2constants.items() for constant in constants}
    print('\nconstants - domain:', constants - all_domain2constants)
    print('domain - constants:', all_domain2constants - constants)



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

def test_countries(train: set, Ne_queries: set, val_test: list[set]) -> bool:
    """
    Checks if each query in val_test has at least 1 neighbor in the training data with a CR queries, and at least one 2nd neighbor has a CR query.
    For every country in val_test, check that:
        - the country has neighbors in train
        - at least one neighbor has a locatedInCR in train
        - get the 2nd neighbours of the country
        - at least one 2nd neighbor of the country has a LocatedInCR queries in train
    """
    # Extract 'locatedInCR' queries for quick lookup
    # located_in_cr_by_country = {query[1]: query for query in train if query[0] == 'locatedInCR'}

    for query in val_test:
        # 1. Query shouldn't be in training data
        if query in train:
            # print('query in train',query)
            return False 

        # 2. 
        # Check for neighbors of `country`
        country = query[1]
        neighbors = get_neighbors(country, Ne_queries)
        if not neighbors:
            # print('no neighbors',country,neighbors)
            return False
        neighbors_with_cr = get_locatedInCR_from_countries(neighbors, train)
        if not neighbors_with_cr:
            # print('No neighbors have locatedInCR')
            return False
        
        # 3.
        # Get the neighbors of the neighbors
        neighbors_of_neighbors = [get_neighbors(ne, Ne_queries) for ne in neighbors]
        # substract the country from the neighbors
        neighbors_of_neighbors = [ne_set-{country} for ne_set in neighbors_of_neighbors]
        if not neighbors_of_neighbors:
            # print('no neighbors_of_neighbors',country,neighbors,neighbors_of_neighbors)
            return False
        # Check if any neighbor of the neighbors has a 'locatedInCR' entry
        neighbors_of_neighbors_with_cr = [get_locatedInCR_from_countries(ne_set, train) for ne_set in neighbors_of_neighbors]
        if not any(neighbors_of_neighbors_with_cr):
            # print('no neighbor of the neighbors has locatedInCR',country,neighbors,neighbors_of_neighbors,neighbors_of_neighbors_with_cr)
            return False

    return True


def get_d_dataset(data: set, islands_CR_queries: set) -> Tuple[set, set, set]:
    '''
    Try with different permutations of the CR queries, and in each permutation, see how many queries you can put in val_test,
        until you reach the maximum number of queries in val_test. Select a query, if valid_test is still valid with that query, add it. 
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

    Ne_queries = {query for query in data if query[0] == 'neighborOf'}
    CR_queries = {query for query in data if query[0] == 'locatedInCR'}
    val_test_candidates = set(CR_queries - islands_CR_queries)

    n = len(CR_queries)
    train_size = int(n*0.8)
    val_size = int(n*0.1)
    test_size = n - train_size - val_size
    val_test_size = val_size + test_size
    print('\nCR queries. Total:', n,', distributed in - train_size:', train_size, 'val_size:', val_size, 'test_size:', test_size, 'val_test_size:', val_test_size, )
    print('data:', len(data),'val_test_candidates:', len(data - val_test_candidates),'\n')

    max_iters = 1000
    iter = 0
    max_len_val_test = 0

    # with tqdm(total=max_iters, desc="Processing iterations") as pbar:
    while iter < max_iters and max_len_val_test<val_test_size:
        percent_complete = (iter + 1) / max_iters * 100
        print(f"Iteration: {iter+1}/{max_iters} ({percent_complete:.2f}%)", end="\r")
        # Initialise the train and val_test. Generate a random permutation of the CR_queries_without_islands
        val_test = []
        train = CR_queries.copy()
        val_test_permutation = list(val_test_candidates.copy())
        random.shuffle(val_test_permutation)
        val_test_permutation_iter = val_test_permutation.copy() # to not modify the original list
        for i,query in enumerate(val_test_permutation_iter):
            # print('query:',i,query)
            # remove the query from the permutation
            val_test_permutation.remove(query) 
            # Do a test with the query, and the updated train set. If it fails, move to the next query
            # q_remove = remove_from_train_countries(train, Ne_queries, query)
            # train_set = train - {query} - q_remove
            train_set = train - {query}
            if not test_countries(train_set, Ne_queries,[query]):
                continue
            # Check if the val/test passes the test. If it fails, move to the next query. Maybe I can move to the next iteration
            if not test_countries(train_set, Ne_queries, val_test + [query]):
                continue
            # if it passes the test, update train and val
            train -= {query} # | q_remove
            val_test.append(query)
        
        # Update the best train and val_test
        if len(val_test) > max_len_val_test:
            max_len_val_test = len(val_test)
            best_train, best_val_test = set(train), set(val_test)
            print('\niter',iter,'max_len_val_test:', max_len_val_test,'/',val_test_size,'. Ratio of success:', max_len_val_test,'/',len(val_test_permutation_iter))
        iter += 1
        # pbar.update(1)

    # Take the best train and val_test, and append the rest of the non-CR queries to train
    train = best_train.union(data - CR_queries) # append the rest of the non-CR queries. 
    test = set(list(best_val_test)[:test_size]) # append best_val_test up to test_size
    val = set(list(best_val_test)[test_size:test_size+val_size]) if len(best_val_test) > test_size else set()
    if len(best_val_test) > test_size+val_size: # If there are too many queries in val_test, add the rest to train
        train = train.union(list(best_val_test)[test_size+val_size:])

    print('New distribution: len train(with non-CR):', len(train), 'len val:', len(val), 'len test:', len(test), 'total:', len(train)+len(val)+len(test))
    print('Keeps the length:', len(data)==len(train)+len(val)+len(test),'. Decrease:',len(data)-(len(train)+len(val)+len(test)))
     
    return train, val, test


def s1_condition(val_test, train):
    '''All the val,test queries need to have a locInCS query in train with a locInSR query in train'''
    print('s1 condition...')
    # countries that appear in a CS query in train, and their subregions
    countries_in_CS_queries = {query[1]: query[2] for query in train if query[0] == 'locatedInCS'}
    # subregions that appear in a SR query in train
    subregions_in_SR_queries = {query[1] for query in train if query[0] == 'locatedInSR'}
    for query in val_test:
        assert query[0] == 'locatedInCR'
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

    '''(GIUSEPPE'S DATA)'''

    # #  VERIFY PROPERTIES OF THE DATA AS A WHOLE 
    # # For that, first get the set of constants
    # root = './experiments/data/countries_dataset_giuseppe/'
    # train_path, val_path, test_path = root+'train.txt', root+'valid.txt', root+'test.txt'
    # dataset_path = root + 'dataset.txt'
    # domain2constants_path = root+'domain2constants.txt'

    # constants, predicates, queries = get_constants_predicates_queries(dataset_path)
    # print('number of queries:', len(queries))
    # domain2constants = get_domain2constants(domain2constants_path)
    # # modify locIn to add a domain
    # queries = add_domain_to_locIn(queries, domain2constants)
    # check_correspondence_domain2constants(constants, domain2constants)
    # islands, islands_CR_queries = check_properties_of_dataset(constants, domain2constants, queries)




    # # CREATE DATASET
    
    # '''
    # VAL,TEST: I NEED TO CHOOSE VAL, TEST QUERIES THAT HAVE 
    #       AT LEAST A NEIGH WITH A LOCATEDINCR. 
    #       AT LEAST ONE NEIGH OF A NEIGH HAS A LOCATEDINCR
    # TRAIN IN S3: REMOVE THE LOCATEDINCR OF THE 1ST NEIGH
    # TRAIN IN S2: DONT REMOVE ANYTHING
    # TRAIN IN S1: ALL THE VAL,TEST QUERIES NEED TO HAVE A LOCATEDINCS QUERY IN TRAIN WITH A LOCATEDINSR QUERY IN TRAIN
    # '''

    # train, val, test = get_d_dataset(set(queries), islands_CR_queries)

    # # Need to modify the train dataset for S3
    # train_s3 = set(train.copy())
    # removed_CR_queries = set()
    # for query in val|test:
    #     neighbors = get_neighbors(query[1], {query for query in train if query[0] == 'neighborOf'})
    #     neighbors_with_cr = get_locatedInCR_from_countries(neighbors, train)
    #     CR_queries_from_neighbors = {query for query in train if query[0] == 'locatedInCR' and query[1] in neighbors}
    #     # delete the neighbors with locatedInCR
    #     train_s3 = train_s3 - CR_queries_from_neighbors
    #     removed_CR_queries = removed_CR_queries.union(CR_queries_from_neighbors)
    # print('removed_CR_queries for s3 train:', len(removed_CR_queries))

    # # save the queries to the files
    # write_queries_to_file(train, train_path)
    # write_queries_to_file(train_s3, train_path.replace('train','train_s3'))
    # write_queries_to_file(val, val_path)
    # write_queries_to_file(test, test_path)




    # CHECK THE CORRECTNESS OF THE DATASET

    def check_correctness(train_path, val_path, test_path, domain2constants_path):

        # load the queries from the files
        constants_train, predicates_train, train = get_constants_predicates_queries(train_path)
        constants_val, predicates_val, val = get_constants_predicates_queries(val_path)
        constants_test, predicates_test, test = get_constants_predicates_queries(test_path)
        predicates = predicates_train | predicates_val | predicates_test
        constants = constants_train | constants_val | constants_test
        dataset = train + val + test

        domain2constants = get_domain2constants(domain2constants_path)
        check_correspondence_domain2constants(constants, domain2constants)

        problematic_queries = set(val + test) & set(train)
        if problematic_queries:
            raise ValueError('There are queries in val or test that are in train:', [print_query(query) for query in problematic_queries], sep='\n')
        
        check_properties_of_dataset(constants, domain2constants, dataset)
        return train, val, test




    # # GIUSEPPE'S DATA
    # root = './experiments/data/countries_dataset_giuseppe/'
    # train_path,val_path,test_path,domain2constants_path = root+'train.txt',root+'valid.txt',root+'test.txt',root+'domain2constants.txt'
    # train, val, test = check_correctness(train_path, val_path, test_path, domain2constants_path)
    # s1_condition(val+test, train)
    # s2_condition(val+test, train)
    # _,_, train_s3 = get_constants_predicates_queries(train_path.replace('train','train_s3'))
    # s3_condition(val+test, train_s3)


    # S1
    # S1. should be able to solve the queries in val and test with the rule locatedInCS(X,W), locatedInSR(W,Z) -> locatedInCR(X,Z)
    # All the val,test queries need to have a locInCS query in train with a locInSR query in train
    # root = './experiments/data/countries_s1/'
    # train_path,val_path,test_path,domain2constants_path = root+'train.txt',root+'valid.txt',root+'test.txt',root+'domain2constants.txt'
    # train, val, test = check_correctness(train_path, val_path, test_path, domain2constants_path)
    # s1_condition(val+test, train)



    # # S2
    # # S2. should be able to solve the queries in val and test with the rule neighborOf(X,Y), locatedInCR(Y,Z) -> locatedInCR(X,Z) (THE ONLY RULE)
    # # All the val,test queries need to have a neighborOf query in train with a locatedInCR query
    # root = './experiments/data/countries_s2/'
    # train_path,val_path,test_path,domain2constants_path = root+'train.txt',root+'valid.txt',root+'test.txt',root+'domain2constants.txt'
    # train, val, test = check_correctness(train_path, val_path, test_path, domain2constants_path)
    # s2_condition(val+test, train)

    # S3
    # S3. should be able to solve the queries in val and test with the rule neighborOf(X,Y), neighborOf(Y,K), locatedInCR(K,Z) -> locatedInCR(X,Z)
    # All the val,test queries with a neighborOf query in train: remove its LocatedInCR in train and make sure the 2nd neigh has a locatedInCR in train
    root = './experiments/data/countries_s3/'
    train_path,val_path,test_path,domain2constants_path = root+'train.txt',root+'valid.txt',root+'test.txt',root+'domain2constants.txt'
    train, val, test = check_correctness(train_path, val_path, test_path, domain2constants_path)
    s3_condition(val+test, train)