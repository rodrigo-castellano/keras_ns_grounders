import re
import sys
 
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
paths = [current_dir+'/train_umls']#, './train_nations','train_kinship_family']

for path in paths:
    atoms = []
    with open(path+'.txt', 'r') as f:
        for line in f:
            line = line.split('(')
            predicate = line[0]
            constants = line[1].split(',') 
            constant1 = constants[0]
            constant2 = constants[1].split(')')[0]
            print(predicate, constant1, constant2)
            atoms.append((predicate, constant1, constant2))

    with open(path+'.tsv', 'w') as f:
        for atom in atoms:
            predicate = atom[0]
            constant1 = atom[1]
            constant2 = atom[2]
            # insert a tab between the predicate and the constants
            f.write(str(constant1)+ '\t' + str(predicate) + '\t' + str(constant2) + '\n')
        