from verify_dataset import add_domain_to_locIn, get_domain2constants, get_constants_predicates_queries, check_properties_of_dataset


def load_raw_data(path):
    ''' reads the file with only the name of the country, opens giuseppe's dataset and adds the queries with the region'''

    substitutions = {'saint_barthélemy': 'saint_barthelemy', 'são_tomé_and_príncipe': 'sao_tome_and_principe', 'timor-leste': 'timor_leste', 
                    'guinea-bissau': 'guinea_bissau', 'curaçao': 'curacao', 'réunion': 'reunion', 
                    'south-eastern_asia': 'south_eastern_asia', 'åland_islands': 'aland_islands','guinea-bissau': 'guinea_bissau'}
    # read the queries with only the name of the country
    with open(path, 'r') as f:
        countries = f.readlines()
    countries = [query.strip().split('\t') for query in countries]
    constants = set()
    for country in countries:
        constant = country[0]
        if constant in substitutions:
            constant = substitutions[constant]
        constants.add(constant)
    return constants


def apply_substitutions(data):
    substitutions = {'saint_barthélemy': 'saint_barthelemy', 'são_tomé_and_príncipe': 'sao_tome_and_principe', 'timor-leste': 'timor_leste', 
                    'guinea-bissau': 'guinea_bissau', 'curaçao': 'curacao', 'réunion': 'reunion', 
                    'south-eastern_asia': 'south_eastern_asia', 'åland_islands': 'aland_islands','guinea-bissau': 'guinea_bissau'}
    new_data = []
    for query in data:
        _, constant1, constant2 = query
        if constant1 in substitutions:
            constant1 = substitutions[constant1]
        if constant2 in substitutions:
            constant2 = substitutions[constant2]
        new_query = (query[0], constant1, constant2)
        new_data.append(new_query)
    return new_data
    
def process_queries(countries,path):
    # read the dataset with the regions from giuseppe
    queries = []
    dataset_path = 'experiments/data/countries_dataset_giuseppe/dataset.txt'
    with open(dataset_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            predicate = line.split('(')[0]
            constant1 = line.split('(')[1].split(',')[0]
            constant2 = line.split('(')[1].split(',')[1].split(')')[0]
            queries.append((predicate, constant1, constant2))

    # add the queries with the regions to the queries
    final_queries = []
    regions = ['europe', 'africa', 'americas', 'asia', 'oceania']

    for j,country in enumerate(countries):
        found = False
        for q in queries:
            if q[0] == 'locatedIn' and q[1] == country and q[2] in regions:
                final_queries.append(q)
                found = True
                break
        if not found:
            raise ValueError(f'country {country} not found in the queries, look manually for its region')

    assert len(final_queries) == len(countries), f'len(final_queries)={len(final_queries)} != len(countries)={len(countries)}'

    final_queries = add_domain_to_locIn(final_queries, domain2constants)    

    # write the final_queries to the path without the original
    with open( path.replace('_original', ''), 'w') as f:
        for query in final_queries:
            f.write(query[0]+'('+query[1]+','+query[2]+').\n')

    return None 




dataset = 'countries_s3'
domain2constants_path = 'experiments/data/'+dataset+'/domain2constants.txt'
train_path = 'experiments/data/'+dataset+'/train_original.txt'
val_path = 'experiments/data/'+dataset+'/valid_original.txt'
test_path = 'experiments/data/'+dataset+'/test_original.txt'

domain2constants = get_domain2constants(domain2constants_path)

# process the valid and test files by adding LocatedInCR(,region) to the queries and saving the files
valid_countries = load_raw_data(val_path)
test_countries = load_raw_data(test_path)
process_queries(valid_countries, val_path)
process_queries(test_countries, test_path)
constants_val, predicates_val, val = get_constants_predicates_queries(val_path.replace('_original', ''))
constants_test, predicates_test, test = get_constants_predicates_queries(test_path.replace('_original', ''))


# replace in train all the constants that are in subtitutions
constants_train, predicates_train, train = get_constants_predicates_queries(train_path)
train = apply_substitutions(train)
train = add_domain_to_locIn(train, domain2constants)

missing_symmetric =[('neighborOf','south_sudan','chad'),
('neighborOf','united_kingdom','cyprus'),
('neighborOf','afghanistan','india'),
('neighborOf','china','nepal'),
('neighborOf','egypt','palestine'),
('neighborOf','israel','palestine'),
('neighborOf','jordan','palestine'),
('neighborOf','chad','sudan')]

train = train + missing_symmetric
# save the file with the new queries
with open( train_path.replace('_original', ''), 'w') as f:
    for query in train:
        f.write(query[0]+'('+query[1]+','+query[2]+').\n')
constants_train, predicates_train, train = get_constants_predicates_queries(train_path.replace('_original', ''))
# constants_train = set([query[1] for query in train] + [query[2] for query in train])
# predicates_train = set([query[0] for query in train])

# check the constants_train that are not in the domain2constants values
all_constants_domain = set([item for sublist in domain2constants.values() for item in sublist])
print('constants: train-domain',set(constants_train) - all_constants_domain)
print('constants: domain-train',all_constants_domain - set(constants_train))

print('the repeated queries you will see are in common in validation and train!!!')
check_properties_of_dataset(domain2constants, list(train)+list(val)+list(test))



# conclusion: it was missing some symmetric relations in the train set, added them manually
# there are queries in common in the validation and train set
# the query micronesia was repeated
# the query neighbour (cyprus,united_kingdom) was removed
# not all countries have a locatedinCR and locatedinCS relation
# denmark and timor_leste in countries s2 dont have any neighbour with a locatedInCR relation