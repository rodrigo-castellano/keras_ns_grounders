# take all the constants from train.txt for every line, with format predicate(constant1,constant2).
# write the unique constants to domain2constants.txt
path = './train.txt'
ctes_path = './domain2constants.txt'

constants = set()
with open(path, 'r') as f: 
    for line in open(path, 'r'):
        if line == '\n':
            continue
        constant1 = line.split('(')[1].split(',')[0]
        constant2 = line.split('(')[1].split(',')[1].split(')')[0]
        # remove blanks from constants
        constant1 = constant1.replace(' ', '')
        constant2 = constant2.replace(' ', '')
        print(constant1, constant2)
        # try to add it as a number, if it fails, add it as a string
        try:
            constants.add(int(constant1))
            constants.add(int(constant2))
        except:
            constants.add(constant1)
            constants.add(constant2)
        # constants.add(constant1)
        # constants.add(constant2)

# if the constants are numbers, order them by value, if they are strings, order them alphabetically
try:
    constants = sorted(list(constants), key=int)
except:
    constants = sorted(list(constants))

with open(ctes_path, 'w') as f:
    for cte in constants:
        f.write(str(cte) + ' ')
