# take all the constants from train.txt for every line, with format predicate(constant1,constant2).
# write the unique constants to domain2constants.txt
dataset = 'pharmkg_supersmall'
# path = './train.txt'
# ctes_path = './domain2constants.txt'

path = './'+dataset+'/train.txt'
ctes_path = './'+dataset+'/domain2constants.txt'
predicates_path = './'+dataset+'/relations.txt'

constants = set()
predicates = set()
with open(path, 'r') as f: 
    for line in open(path, 'r'):
        predicate = line.split('(')[0]
        consant1 = line.split('(')[1].split(',')[0]
        consant2 = line.split('(')[1].split(',')[1].split(')')[0]
        # print(consant1, consant2)
        constants.add(consant1)
        constants.add(consant2)
        predicates.add(predicate)

# if the constants are numbers, order them by value
# if they are strings, order them alphabetically
print(constants)
constants = sorted(constants, key=lambda x: int(x) if x.isdigit() else x)
predicates = sorted(predicates, key=lambda x: int(x) if x.isdigit() else x)
print(constants)

with open(ctes_path, 'w') as f:
    for cte in constants:
        f.write(cte + '\n')

with open(predicates_path, 'w') as f:
    for pred in predicates:
        f.write(pred + '\n')
