import re
import sys
 

paths = ['./pharmkg_supersmall/train','./pharmkg_supersmall/valid','./pharmkg_supersmall/test',
          './kinship_family/train','./kinship_family/valid','./kinship_family/test'] # './nations/train',

for path in paths:
    print('path',path)
    save_path = path+'_triples'
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

    with open(save_path+'.txt', 'w') as f:
        print('writing to ', save_path)
        for atom in atoms:
            predicate = atom[0]
            constant1 = atom[1]
            constant2 = atom[2]
            # insert a tab between the predicate and the constants
            f.write(str(constant1)+ '\t' + str(predicate) + '\t' + str(constant2) + '\n')
        