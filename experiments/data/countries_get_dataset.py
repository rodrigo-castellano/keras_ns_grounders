import os 
import sys
import re
import random
from typing import Tuple
from tqdm import tqdm

from verify_dataset import get_constants_predicates_queries, get_domain2constants, check_constants_in_domain, check_properties_of_dataset,get_neighbors, get_locatedInCR_from_countries, add_domain_to_locIn, write_queries_to_file
from ablation_get_dataset import get_dataset

# def test_countries(train: set, Ne_queries: set, val_test: list[set]) -> bool:
#     """
#     Checks if each query in val_test has at least 1 neighbor in the training data with a CR queries, and at least one 2nd neighbor has a CR query.
#     For every country in val_test, check that:
#         - the country has neighbors in train
#         - at least one neighbor has a locatedInCR in train
#         - get the 2nd neighbours of the country
#         - at least one 2nd neighbor of the country has a LocatedInCR queries in train
#     """

#     for query in val_test:
#         # 1. Query shouldn't be in training data
#         if query in train:
#             # print('query in train',query)
#             return False 

#         # 2. 
#         # Check for neighbors of `country`
#         country = query[1]
#         neighbors = get_neighbors(country, Ne_queries)
#         if not neighbors:
#             # print('no neighbors',country,neighbors)
#             return False
#         neighbors_with_cr = get_locatedInCR_from_countries(neighbors, train)
#         if not neighbors_with_cr:
#             # print('No neighbors have locatedInCR')
#             return False
        
#         # 3.
#         # Get the neighbors of the neighbors
#         neighbors_of_neighbors = [get_neighbors(ne, Ne_queries) for ne in neighbors]
#         # substract the country from the neighbors
#         neighbors_of_neighbors = [ne_set-{country} for ne_set in neighbors_of_neighbors]
#         if not neighbors_of_neighbors:
#             # print('no neighbors_of_neighbors',country,neighbors,neighbors_of_neighbors)
#             return False
#         # Check if any neighbor of the neighbors has a 'locatedInCR' entry
#         neighbors_of_neighbors_with_cr = [get_locatedInCR_from_countries(ne_set, train) for ne_set in neighbors_of_neighbors]
#         if not any(neighbors_of_neighbors_with_cr):
#             # print('no neighbor of the neighbors has locatedInCR',country,neighbors,neighbors_of_neighbors,neighbors_of_neighbors_with_cr)
#             return False

#     return True


# def get_dataset(data: set, islands_CR_queries: set) -> Tuple[set, set, set]:
#     '''
#     Try with different permutations of the CR queries, and in each permutation, see how many queries you can put in val_test,
#         until you reach the maximum number of queries in val_test. Select a query, if valid_test is still valid with that query, add it. 
#     Get the d1 dataset. 
#     c,c1,c2,.. are countries
#     NE, CR are queries. 
#     NE is a neighbour query, NE_c1_c2 is a neighbour query that contains c1,c2 (no matter the order)
#     CR is a LocatedInCR query, CR_c is a LocatedInCR query that contains c
#     q_c is a CR_c query in val_test

#     Rule:    ∀ q_ci in val_test, ∀ NE_ci(∄ CR_ci, ∃NE_ci_cj(∃CR_cj, )). 
#     Meaning that for every CR query in val_test, all the neighbour queries in train that 
#         i.i) do not have a CR query in train, i.ii) have a neighbour query in train that ii) has a CR query in train. 
#     - I need a function that given queries/countries and trainset, it returns its NeighborOf queries in train / its list of neighbors
#     - I need a function that given queries/countries and trainset, it returns its LocatedInCR queries in train / bool: if they have LocatedInCR queries
#     Algo: 
#     - while iter < max_iters and max_len_val_test<val_test_size:
#         - start a val_test=[] and train. Generate a random permutation (to change)
#             - for query in CR_queries_without_islands:
#                 - atoms_remove: get the CR atoms to remove from train
#                 - updated_train_test = train - {atoms_remove} - {query}
#                 - Updated_val_test = val_test + query
#                 - check if the updated_val_test passes the test_d1 function
#     '''

#     Ne_queries = {query for query in data if query[0] == 'neighborOf'}
#     CR_queries = {query for query in data if query[0] == 'locatedInCR'}
#     val_test_candidates = set(CR_queries - islands_CR_queries)

#     n = len(CR_queries)
#     train_size = int(n*0.8)
#     val_size = int(n*0.1)
#     test_size = n - train_size - val_size
#     val_test_size = val_size + test_size
#     print('\nCR queries. Total:', n,', distributed in - train_size:', train_size, 'val_size:', val_size, 'test_size:', test_size, 'val_test_size:', val_test_size, )
#     print('data:', len(data),'val_test_candidates:', len(data - val_test_candidates),'\n')

#     max_iters = 1000
#     iter = 0
#     max_len_val_test = 0

#     # with tqdm(total=max_iters, desc="Processing iterations") as pbar:
#     while iter < max_iters and max_len_val_test<val_test_size:
#         percent_complete = (iter + 1) / max_iters * 100
#         print(f"Iteration: {iter+1}/{max_iters} ({percent_complete:.2f}%)", end="\r")
#         # Initialise the train and val_test. Generate a random permutation of the CR_queries_without_islands
#         val_test = []
#         train = CR_queries.copy()
#         val_test_permutation = list(val_test_candidates.copy())
#         random.shuffle(val_test_permutation)
#         val_test_permutation_iter = val_test_permutation.copy() # to not modify the original list
#         for i,query in enumerate(val_test_permutation_iter):
#             # print('query:',i,query)
#             # remove the query from the permutation
#             val_test_permutation.remove(query) 
#             # Do a test with the query, and the updated train set. If it fails, move to the next query
#             # q_remove = remove_from_train_countries(train, Ne_queries, query)
#             # train_set = train - {query} - q_remove
#             train_set = train - {query}
#             if not test_countries(train_set, Ne_queries,[query]):
#                 continue
#             # Check if the val/test passes the test. If it fails, move to the next query. Maybe I can move to the next iteration
#             if not test_countries(train_set, Ne_queries, val_test + [query]):
#                 continue
#             # if it passes the test, update train and val
#             train -= {query} # | q_remove
#             val_test.append(query)
        
#         # Update the best train and val_test
#         if len(val_test) > max_len_val_test:
#             max_len_val_test = len(val_test)
#             best_train, best_val_test = set(train), set(val_test)
#             print('\niter',iter,'max_len_val_test:', max_len_val_test,'/',val_test_size,'. Ratio of success:', max_len_val_test,'/',len(val_test_permutation_iter))
#         iter += 1
#         # pbar.update(1)

#     # Take the best train and val_test, and append the rest of the non-CR queries to train
#     train = best_train.union(data - CR_queries) # append the rest of the non-CR queries. 
#     test = set(list(best_val_test)[:test_size]) # append best_val_test up to test_size
#     val = set(list(best_val_test)[test_size:test_size+val_size]) if len(best_val_test) > test_size else set()
#     if len(best_val_test) > test_size+val_size: # If there are too many queries in val_test, add the rest to train
#         train = train.union(list(best_val_test)[test_size+val_size:])

#     print('New distribution: len train(with non-CR):', len(train), 'len val:', len(val), 'len test:', len(test), 'total:', len(train)+len(val)+len(test))
#     print('Keeps the length:', len(data)==len(train)+len(val)+len(test),'. Decrease:',len(data)-(len(train)+len(val)+len(test)))
     
#     return train, val, test



if __name__ == '__main__':

    '''(GIUSEPPE'S DATA)'''

    # OBTAIN THE WHOLE DATA
    root = './experiments/data/countries_dataset_giuseppe/'
    train_path, val_path, test_path = root+'train.txt', root+'valid.txt', root+'test.txt'
    dataset_path, domain2constants_path = root+'dataset.txt', root+'domain2constants.txt'

    constants, predicates, dataset = get_constants_predicates_queries(dataset_path)
    domain2constants = get_domain2constants(domain2constants_path)
    dataset = add_domain_to_locIn(dataset, domain2constants)
    check_constants_in_domain(constants, domain2constants)
    islands, islands_CR_queries = check_properties_of_dataset(domain2constants, dataset)


    # CREATE SPLITS
    '''
    S1: condition related to CS, SR, so not relevant to build the dataset
        all the queries in val,test need to have a CS query in TRAIN with a SR query in TRAIN
    S2: all queries have a neghbor with a CR query
    S3: all queries dont have a neighbor with a CR query, and at least one neighbor of a neighbor has a CR query 
    Build an S3 dataset., and to fulfill the S2 condition, for each query in val,test, add a 
        CR query to the negih in train if there's none

    VAL,TEST: I NEED TO CHOOSE VAL, TEST QUERIES THAT HAVE 
          AT LEAST A NEIGH WITH A LOCATEDINCR. 
          AT LEAST ONE NEIGH OF A NEIGH HAS A LOCATEDINCR
    TRAIN IN S3: REMOVE THE LOCATEDINCR OF THE 1ST NEIGH
    TRAIN IN S2: DONT REMOVE ANYTHING
    TRAIN IN S1: ALL THE VAL,TEST QUERIES NEED TO HAVE A LOCATEDINCS QUERY IN TRAIN WITH A LOCATEDINSR QUERY IN TRAIN

    I can call the get_dataset function to get the data for s1,s2, and maximize for s3 (ask max of queries that have a neigh2 with a CR query and no neigh with a CR query)
    '''
    from ablation_get_dataset import get_dataset, build_neighbor_map, get_located_in_cr_countries
    train_s3, val, test = get_dataset(dataset, islands_CR_queries, type='d2')

    # To get the train set of s2 (also s1), for each query in val_test, get the neighbors, and add a CR query
    #   randomly from one neighbour if there's none
    train = set(train_s3.copy())
    
    ne_queries = {query for query in train if query[0] == 'neighborOf'}
    neighbor_map = build_neighbor_map(ne_queries)
    country_with_cr = get_located_in_cr_countries(train)
    countries_in_val_test = {query[1] for query in val|test}
    countries2regions = {query[1]:query[2] for query in dataset if query[0] == 'locatedIn'}

    for query in val|test:
        print('query:',query,query[1])
        neighbors = get_neighbors(query[1],neighbor_map)
        neighbors_with_cr = neighbors & country_with_cr
        assert neighbors, f'No neighbors for {query}'
        if not neighbors_with_cr:
            # choose a neighbor that doesnt have a CR query in val|test, and add a CR query in train
            neighbors_not_in_val_test = neighbors - countries_in_val_test
            if neighbors_not_in_val_test:
                neighbor = random.choice(list(neighbors_not_in_val_test))
                train.add(('locatedInCR',neighbor,countries2regions[neighbor]))
            else:
                # all the neighbors have a CR query in val|test, choose a random neighbor and add a CR query in train
                neighbor = random.choice(list(neighbors))
                train.add(('locatedInCR',neighbor,countries2regions[neighbor]))
                # delete the CR query of that neighbour from the val|test
                val = val - {query} if query[1] == neighbor else val
                test = test - {query} if query[1] == neighbor else test
                print('removed query:',query)

    # train, val, test = get_dataset(set(dataset), islands_CR_queries)
    # # Need to modify the train dataset for S3
    # train_s3 = set(train.copy())
    # ne_queries = {query for query in train_s3 if query[0] == 'neighborOf'}
    # removed_CR_queries = set()
    # for query in val|test:
    #     neighbors = get_neighbors(query[1], ne_queries)
    #     neighbors_with_cr = get_locatedInCR_from_countries(neighbors, train_s3)
    #     CR_queries_from_neighbors = {query for query in train_s3 if query[0] == 'locatedInCR' and query[1] in neighbors}
    #     # delete the neighbors with locatedInCR
    #     train_s3 = train_s3 - CR_queries_from_neighbors
    #     removed_CR_queries = removed_CR_queries.union(CR_queries_from_neighbors)
    # print('removed_CR_queries for s3 train:', len(removed_CR_queries))

    from verify_dataset import s1_condition, s2_condition, s3_condition
    s1_condition(val | test, train)
    s2_condition(val | test, train)
    s3_condition(val | test, train_s3)


    # save the queries to the files
    # write_queries_to_file(train, train_path)
    # write_queries_to_file(train_s3, train_path.replace('train','train_s3'))
    # write_queries_to_file(val, val_path)
    # write_queries_to_file(test, test_path)




