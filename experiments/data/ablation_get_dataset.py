import os 
import sys
import re
import random
from typing import Tuple
from tqdm import tqdm
from verify_dataset import write_queries_to_file
from verify_dataset import get_constants_predicates_queries, get_domain2constants, check_constants_in_domain, remove_locatedInCS_locatedInSR, get_neighbors, get_locatedInCR, get_locatedInCR_from_countries, get_neighbors_from_countries
from typing import List, Set, Tuple, Optional
from collections import defaultdict




def build_neighbor_map(data: Set[Tuple[str, str, str]]) -> dict:
    """Builds a neighbor map for efficient neighbor lookups."""
    neighbor_map = defaultdict(set)
    for relation, country1, country2 in data:
        if relation == 'neighborOf':
            neighbor_map[country1].add(country2)
            neighbor_map[country2].add(country1)
    return neighbor_map

def get_neighbors(country: str, neighbor_map: dict) -> Set[str]:
    """Efficiently gets neighbors using the pre-built map."""
    if country in neighbor_map:
        return neighbor_map[country]
    else:
        return set()

def get_located_in_cr_countries(data: Set[Tuple[str, str, str]]) -> Set[str]:
    """Efficiently extracts countries with 'locatedInCR' relations."""
    return {country for relation, country, _ in data if relation == 'locatedInCR'}

def test_d1(train: Set[Tuple[str, str, str]],
            val_test: List[Tuple[str, str, str]],
            neighbor_map: Optional[dict]=None,
            country_with_cr: Optional[Set[str]]=None,
            verbose=0) -> bool:
    """
    Checks if each query in val_test has at least one neighbor with a CR query.
    """
    for query in val_test:
        if query in train:
            if verbose: print('query in train', query)
            return False

        country = query[1]
        neighbors = get_neighbors(country, neighbor_map)
        if not neighbors:
            if verbose: print('no neighbors', country, neighbors)
            return False

        if not (neighbors & country_with_cr):
            if verbose: print('no neighbors with locatedInCR', country, neighbors)
            return False
    return True

def test_d2(train: Set[Tuple[str, str, str]], 
            val_test: List[Tuple[str, str, str]], 
            neighbor_map: Optional[dict]=None,
            country_with_cr: Optional[Set[str]]=None,
            verbose=0) -> bool:
    """
    Checks if each query in val_test has in training data no neighbors with CR queries, 
        and at least one neighbors of neighbors with a CR queries.
    """
    for query in val_test:
        if query in train:
            if verbose: print('query in train', query)
            return False

        country = query[1]
        
        # Step 1: Get neighbors and Check that no of them has a 'locatedInCR' entry
        neighbors = get_neighbors(country, neighbor_map)
        if not neighbors:
            if verbose: print('no neighbors', country, neighbors)
            return False

        if neighbors & country_with_cr:
            if verbose: print('Some neighbors have locatedInCR', country, neighbors)
            return False

        # Step 2: Get neighbors of neighbors and Check that at leasts one of them has a 'locatedInCR' entry
        neighbors_of_neighbors = set()
        for neighbor in neighbors:
            neighbors_of_neighbors.update(get_neighbors(neighbor, neighbor_map))
        neighbors_of_neighbors.discard(country)  # Remove the original country

        if not neighbors_of_neighbors:
            if verbose: print('no neighbors_of_neighbors', country, neighbors, neighbors_of_neighbors)
            return False

        if not (neighbors_of_neighbors & country_with_cr):
            if verbose: print('no neighbor of the neighbors has locatedInCR', country, neighbors, neighbors_of_neighbors)
            return False

    return True

def test_d3(train: Set[Tuple[str, str, str]], 
            val_test: List[Tuple[str, str, str]], 
            neighbor_map: Optional[dict]=None,
            country_with_cr: Optional[Set[str]]=None,
            verbose=0) -> bool:
    """
    Checks if each query in val_test has no neighbors in the training data with CR queries, 
    and no neighbors of neighbors have CR queries, 
    and at least one of the neighbors of the neighbors of the neighbors have a CR query.
    """
    for query in val_test:
        if query in train:
            if verbose: print('query in train', query)
            return False

        country = query[1]
        
        # Step 1: Get neighbors and Check that no of them has a 'locatedInCR' entry
        neighbors = get_neighbors(country, neighbor_map)
        if not neighbors:
            if verbose: print('no neighbors', country, neighbors)
            return False

        if neighbors & country_with_cr:
            if verbose: print('Some neighbors have locatedInCR', country, neighbors)
            return False

        # Step 2: Get neighbors of neighbors and Check that no of them has a 'locatedInCR' entry
        neighbors_of_neighbors = set()
        for neighbor in neighbors:
            neighbors_of_neighbors.update(get_neighbors(neighbor, neighbor_map))
        neighbors_of_neighbors.discard(country)  # Remove the original country

        if not neighbors_of_neighbors:
            if verbose: print('no neighbors_of_neighbors', country, neighbors, neighbors_of_neighbors)
            return False
        
        if neighbors_of_neighbors & country_with_cr:
            if verbose: print('Some neighbors_of_neighbors have locatedInCR', country, neighbors, neighbors_of_neighbors)
            return False

        # Step 3: Get neighbors of neighbors of neighbors and Check that at least one of them has a 'locatedInCR' entry
        neighbors_of_neighbors_of_neighbors = set()
        for neighbor_of_neighbor in neighbors_of_neighbors:
            neighbors_of_neighbors_of_neighbors.update(get_neighbors(neighbor_of_neighbor, neighbor_map))
        
        neighbors_of_neighbors_of_neighbors -= neighbors
        neighbors_of_neighbors_of_neighbors.discard(country)

        if not (neighbors_of_neighbors_of_neighbors & country_with_cr):
            if verbose: print('no neighbors_of_neighbors_of_neighbors has locatedInCR', country, neighbors, neighbors_of_neighbors, neighbors_of_neighbors_of_neighbors)
            return False

    return True

def remove_from_train_d3(train: Set[Tuple[str, str, str]], 
                        query: Tuple[str, str, str],
                        neighbor_map: dict, 
                        country_with_cr: Set[str],
                        verbose=0) -> Set[Tuple[str, str, str]]:
    '''For the country in a query, remove the CR queries from i) its neighbors and ii) the neighbors of the neighbors'''
    print('\nRemoving from train:') if verbose else None
    country = query[1]
    neighbors = get_neighbors(country, neighbor_map)
    neighbors_with_cr = neighbors & country_with_cr
    print('neighbors',len(neighbors),neighbors) if verbose else None
    print('neighbors_with_cr',len(neighbors_with_cr),neighbors_with_cr) if verbose else None
    neighbors_of_neighbors = set()
    for neighbor in neighbors:
        neighbors_of_neighbors.update(get_neighbors(neighbor, neighbor_map))
    neighbors_of_neighbors.discard(country)

    neighbors_of_neighbors_with_cr = neighbors_of_neighbors & country_with_cr
    print('neighbors_of_neighbors',len(neighbors_of_neighbors),neighbors_of_neighbors)  if verbose else None
    print('neighbors_of_neighbors_with_cr',len(neighbors_of_neighbors_with_cr),neighbors_of_neighbors_with_cr) if verbose else None
    countries_to_remove = neighbors_with_cr | neighbors_of_neighbors_with_cr
    print('countries_to_remove',len(countries_to_remove),countries_to_remove) if verbose else None
    return {(relation, c, cr) for relation, c, cr in train if relation == 'locatedInCR' and c in countries_to_remove}

def remove_from_train_d2(train: Set[Tuple[str, str, str]],
                        query: Tuple[str, str, str],
                        neighbor_map: dict,
                        country_with_cr: Set[str],
                        verbose=0) -> Set[Tuple[str, str, str]]:
    '''For the country in a query, remove the CR queries from its neighbors'''
    country = query[1]
    neighbors = get_neighbors(country, neighbor_map)
    neighbors_with_cr = neighbors & country_with_cr 
    return {(relation, c, cr) for relation, c, cr in train if relation == 'locatedInCR' and c in neighbors_with_cr}

def get_provable_queries(train: Set[Tuple[str, str, str]],
                        q_set: Set[Tuple[str, str, str]],
                        type='d3',
                        ne_queries: Optional[Set[Tuple[str, str, str]]]=None,
                        neighbor_map: Optional[dict]=None,
                        country_with_cr: Optional[Set[str]]=None) -> int:
    '''Checks if each query in q_set fulfills the conditions of d by calling the test_d function
    before calling it, remove the q_set query from the train set'''
    if neighbor_map is None:
        assert ne_queries is not None, "ne_queries must be provided if neighbor_map is not"
        neighbor_map = build_neighbor_map(ne_queries)
    if country_with_cr is None:
        country_with_cr = get_located_in_cr_countries(train)
    n_provables = 0
    for query in q_set:
        # 1. remove the query temporarily from the train set
        train_temp = train - {query} 
        # 2. Check if the query passes the test_d2 function
        if test_dataset(train_temp, [query], neighbor_map=neighbor_map, country_with_cr=country_with_cr, type=type):
            n_provables += 1
    return n_provables

def test_dataset(train: Set[Tuple[str, str, str]], 
                val_test: List[Tuple[str, str, str]],
                ne_queries: Optional[Set[Tuple[str, str, str]]]=None,
                neighbor_map: Optional[dict]=None,
                country_with_cr: Optional[Set[str]]=None,
                type='d3',
                verbose=0) -> bool:
    if neighbor_map is None:
        assert ne_queries is not None, "ne_queries must be provided if neighbor_map is not"
        neighbor_map = build_neighbor_map(ne_queries)
    if country_with_cr is None:
        country_with_cr = get_located_in_cr_countries(train)
    if type == 'd1':
        return test_d1(train, val_test, neighbor_map=neighbor_map, country_with_cr=country_with_cr,verbose=verbose)
    elif type == 'd2':
        return test_d2(train, val_test, neighbor_map=neighbor_map, country_with_cr=country_with_cr,verbose=verbose)
    elif type == 'd3':
        return test_d3(train, val_test, neighbor_map=neighbor_map, country_with_cr=country_with_cr,verbose=verbose)
    else:
        raise ValueError(f"Invalid type: {type}")


def remove_from_train(train: Set[Tuple[str, str, str]], 
                    query: Tuple[str, str, str], 
                    ne_queries: Optional[Set[Tuple[str, str, str]]]=None,
                    neighbor_map: Optional[dict]=None,
                    country_with_cr: Optional[Set[str]]=None,
                    type='d3',
                    verbose=0) -> Set[Tuple[str, str, str]]:
    if neighbor_map is None:
        assert ne_queries is not None, "ne_queries must be provided if neighbor_map is not"
        neighbor_map = build_neighbor_map(ne_queries)
    if country_with_cr is None:
        country_with_cr = get_located_in_cr_countries(train)
    if type == 'd1':
        return set()
    elif type == 'd2':
        return remove_from_train_d2(train, query, neighbor_map, country_with_cr, verbose=verbose)
    elif type == 'd3':
        return remove_from_train_d3(train, query, neighbor_map, country_with_cr, verbose=verbose)
    else:
        raise ValueError(f"Invalid type: {type}")



def iterate_over_candidates(
    train_candidates: Set,
    ne_queries: Set,
    val_test_candidates: List,
    val_test_size: int,
    train_size: int,
    type: str = 'd3'
) -> Tuple[Set, Set]:
    """
    Iterates over permutations of validation/test candidates to maximize the number of valid queries
    in the validation/test set while ensuring the training set remains provable.

    Args:
        train_candidates (Set): Initial set of training candidates.
        ne_queries (Set): Set of queries to validate against.
        val_test_candidates (List): List of candidates for the validation/test set.
        val_test_size (int): Desired size of the validation/test set.
        train_size (int): Target size for the training set.
        type (str): Type of operation for query processing. Default is 'd3'.

    Returns:
        Tuple[Set, Set]: The best training set and validation/test set found.
    """
    max_iters = 100000
    best_train, best_val_test = set(), set()
    max_len_val_test, max_len_provables_train = 0, 0

    neighbor_map = build_neighbor_map(ne_queries)
    country_with_cr_train = get_located_in_cr_countries(train_candidates)

    for iteration in range(max_iters):
        print(f"Iteration: {iteration + 1}/{max_iters} ({(iteration + 1) / max_iters * 100:.2f}%)", end="\r")

        # Initialize training and validation/test sets
        train = train_candidates.copy()
        country_with_cr = country_with_cr_train.copy()
        # Shuffle candidates for a random permutation
        shuffled_candidates = val_test_candidates.copy()
        random.shuffle(shuffled_candidates)

        val_test = []
        for i,query in enumerate(shuffled_candidates):
            if len(val_test) >= val_test_size:
                break
            # Remove query and associated atoms from the training set
            atoms_to_remove = remove_from_train(train, query,type=type, neighbor_map=neighbor_map,country_with_cr=country_with_cr,verbose=0)
            # create a temporary training set and country_with_cr set so that if the query is not valid, we can revert back
            train_tmp = train - {query} - atoms_to_remove
            country_with_cr_tmp = country_with_cr - {query[1]} - {atom[1] for atom in atoms_to_remove if atom[0] == 'locatedInCR'}
            # Validate the query and updated validation/test set
            if not test_dataset(train_tmp, [query], type=type, neighbor_map=neighbor_map, country_with_cr=country_with_cr_tmp, verbose=0):  
                continue
            if not test_dataset(train_tmp, val_test + [query], type=type, neighbor_map=neighbor_map, country_with_cr=country_with_cr_tmp): 
                continue
            # Update training and validation/test sets
            train -= {query} | atoms_to_remove
            country_with_cr -= {query[1]} | {atom[1] for atom in atoms_to_remove if atom[0] == 'locatedInCR'}
            val_test.append(query)
        # print('\npassed test',test_dataset(train, val_test, type=type, neighbor_map=neighbor_map,country_with_cr=country_with_cr))
        # Update the best sets if criteria are met
        provable_queries_train = get_provable_queries(train, train, type=type, neighbor_map=neighbor_map,country_with_cr=country_with_cr) 
        if len(val_test) > max_len_val_test or (
            len(val_test) == val_test_size and provable_queries_train > max_len_provables_train
        ):
            max_len_val_test = len(val_test)
            max_len_provables_train = provable_queries_train
            best_train, best_val_test = train.copy(), set(val_test)

            print(
                f"\nIteration {iteration + 1}: Updated best sets.\n"
                f"Validation/Test size: {len(val_test)}/{val_test_size}."
                f" Provable queries in train: {provable_queries_train}/{train_size}."
            )

        # Stop early if both criteria are satisfied
        if len(val_test) == val_test_size and max_len_provables_train >= train_size:
            break
    print('\ntest check passed',test_dataset(best_train, best_val_test, type=type, neighbor_map=neighbor_map, country_with_cr=country_with_cr_train))

    return best_train, best_val_test


def get_dataset(
    data: List[Tuple],
    islands_cr_queries: Set[Tuple],
    type: str = 'd3'
) -> Tuple[Set[Tuple], Set[Tuple], Set[Tuple]]:
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        data (List[Tuple]): List of queries.
        islands_cr_queries (Set[Tuple]): Set of isolated CR queries that cannot be used in val/test.
        type (str): Type of operation for query processing. Default is 'd3'.

    Returns:
        Tuple[Set[Tuple], Set[Tuple], Set[Tuple]]: Training, validation, and test sets.
    """
    # Split queries by type
    ne_queries = {query for query in data if query[0] == 'neighborOf'}
    cr_queries = {query for query in data if query[0] == 'locatedInCR'}

    # Separate candidates for training and validation/test
    train_candidates = cr_queries
    val_test_candidates = list(cr_queries - islands_cr_queries)

    # Calculate dataset sizes
    n = len(cr_queries)
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)
    test_size = n - train_size - val_size
    val_test_size = val_size + test_size

    print(
        f"\nCR queries. Total: {n}, distributed as - "
        f"train_size: {train_size}, val_size: {val_size}, "
        f"test_size: {test_size}, val_test_size: {val_test_size}"
    )
    print(f"Data: {len(data)}, val_test_candidates: {len(val_test_candidates)}\n")

    # Iterate to find the best train and val/test split
    best_train, best_val_test = iterate_over_candidates(
        train_candidates, ne_queries, val_test_candidates, val_test_size, train_size, type=type)

    # Allocate the remaining data to train, val, and test sets
    train = best_train.union(set(data) - cr_queries)
    test = set(list(best_val_test)[:test_size])
    val = (
        set(list(best_val_test)[test_size:test_size + val_size])
        if len(best_val_test) > test_size else set()
    )
    assert len(best_val_test) <= test_size + val_size, "len(best_val_test) exceeds val_test_size"

    print(
        f"New distribution: train (with non-CR): {len(train)}, val: {len(val)}, "
        f"test: {len(test)}, total: {len(train) + len(val) + len(test)}"
    )
    print(f'Keeps the length: {len(data) == len(train) + len(val) + len(test)}. Decrease: {len(data) - (len(train) + len(val) + len(test))}')

    return train, val, test


if __name__ == '__main__':

    from verify_dataset import check_properties_of_dataset, add_domain_to_locIn

    dataset_type = 'd3'
    root = './experiments/data/countries_ablation/'
    dataset_path, domain2constants_path = root + 'dataset.txt', root+'domain2constants.txt'
    train_path, val_path, test_path = root+'train_'+dataset_type+'.txt', root+'val_'+dataset_type+'.txt', root+'test_'+dataset_type+'.txt'

    constants, predicates, dataset = get_constants_predicates_queries(dataset_path) 
    domain2constants = get_domain2constants(domain2constants_path)
    dataset = add_domain_to_locIn(dataset, domain2constants)
    islands, islands_CR_queries = check_properties_of_dataset(domain2constants, dataset)

    train, val, test = get_dataset(dataset, islands_CR_queries, type=dataset_type)
    Ne_queries = {query for query in dataset if query[0] == 'neighborOf'}
    # print('Evaluating validation...')
    passed_val = test_dataset(train, val, ne_queries=Ne_queries,type=dataset_type)
    # print('Evaluating test...')
    passed_test = test_dataset(train, test, ne_queries=Ne_queries,type=dataset_type)
    passed = passed_val and passed_test
    print('Final test passed:', passed)

    # write_queries_to_file(train, root+'train_'+dataset_type+'.txt')
    # write_queries_to_file(val, root+'val_'+dataset_type+'.txt')
    # write_queries_to_file(test, root+'test_'+dataset_type+'.txt')

