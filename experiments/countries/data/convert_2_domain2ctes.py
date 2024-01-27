# take all the constants from train.txt for every line, with format predicate(constant1,constant2).
# write the unique constants to domain2constants.txt
import os
path = './train.txt'
ctes_path = './domain2constants.txt'
paths = ['./train.txt'] #,'./test.txt','./valid.txt']

constants = set()
for path in paths: 
    # if the file exists: 
    if os.path.exists(path):
        with open(path, 'r') as f: 
            for line in open(path, 'r'):
                consant1 = line.split('(')[1].split(',')[0]
                consant2 = line.split('(')[1].split(',')[1].split(')')[0]
                # print(consant1, consant2)
                constants.add(consant1)
                constants.add(consant2)

# if the constants are numbers, order them by value
# if they are strings, order them alphabetically
print(constants)
constants = sorted(constants, key=lambda x: int(x) if x.isdigit() else x)
print(constants)

with open(ctes_path, 'w') as f:
    for cte in constants:
        f.write(cte + ' ')
