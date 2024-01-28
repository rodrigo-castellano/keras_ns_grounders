# open the file nations_10_4.txt. It contains the rules extracted by NCRL.
# In every line there is one rule, with the structure:
# a (b)	Head <-- Predicate1, Predicate2, ..., PredicateN
# The first number a is the confidence, the second number b is the support.
# Filter by confidence and support, and save the rules in a list.

import re
import sys

confidence_threshold = 0.1

# dataset = 'nations'
# dataset = 'kinship_family'
dataset = 'pharmkg_supersmall'
paths = ['./rules/'+dataset+'_5_2','./rules/'+dataset+'_5_3']
save_path = './rules/'+dataset

rules = []
for path in paths:
    with open(path+'.txt', 'r') as f:
        for line in f:
            # split by tab
            line = line.split('\t')
            scores = line[0].split(' ')
            confidence = float(scores[0])
            rule = line[1]
            head = rule.split('<--')[0].strip()
            body = rule.split('<--')[1].strip()
            # split by comma
            predicates = [ p.strip() for p in body.split(',') ]

            # add the vars to the head and body of the rule. To the head, append the vars (a,b)
            head = head + '(a,b)'
            # For the body, the first var of the first predicate is a, the last var of the last predicate is b, and the rest are c, d, e, ...
            # For example, for length 2, the body is: Predicate1(a,c), Predicate2(c,b)
            # For length 3, the body is: Predicate1(a,c), Predicate2(c,d), Predicate3(d,b)
            # For length 4, the body is: Predicate1(a,c), Predicate2(c,d), Predicate3(d,e), Predicate4(e,b)
            if len (predicates) == 1:
                predicates[0] = predicates[0] + '(a,b)'
            elif len(predicates) == 2:
                predicates[0] = predicates[0] + '(a,c)'
                predicates[1] = predicates[1] + '(c,b)'
            elif len(predicates) == 3:
                predicates[0] = predicates[0] + '(a,c)'
                predicates[1] = predicates[1] + '(c,d)'
                predicates[2] = predicates[2] + '(d,b)'
            elif len(predicates) == 4:
                predicates[0] = predicates[0] + '(a,c)'
                predicates[1] = predicates[1] + '(c,d)'
                predicates[2] = predicates[2] + '(d,e)'
                predicates[3] = predicates[3] + '(e,b)'

            # print all the info
            print('confidence: ', confidence, 'head: ', head, 'body: ', predicates)

            # filter by confidence
            if confidence >= confidence_threshold:
                rules.append((confidence, head, predicates))
        
# order the rules by confidence
rules.sort(key=lambda tup: tup[0], reverse=True)
# filter them by confidence 0.1
rules = [rule for rule in rules if rule[0] >= confidence_threshold]
 
with open(save_path+'.txt', 'w') as f:
    for i,rule in enumerate(rules):
        # predicates is the second element of the tuple, which is a list of strings. Join them with comma
        predicates = ', '.join(rule[2])
        string = 'r' + str(i) + ':' + str(rule[0]) + ':' + str(predicates) + ' -> ' + str(rule[1])+'\n'
        f.write(string)
        print(string)