# open the file nations_rules.txt and read the rules
# the format of the rules is the following:
# str(Predicate) => str(Rules) float(std_Confidence) float (Support) string(not needed) 

import re
import sys

std_confidence_threshold = 0.0
max_rules = 100

path = './pharmkg_supersmall'
save_path = path+'_filtered'

rules = []

with open(path+'.txt', 'r') as f:
    for line in f:
        # split by '=>'
        line = line.split('=>')
        body = line[0].strip()
        # In line[1], all the elements that are separated by more than one space, reduce it to one space
        line[1] = re.sub(' +', ' ', line[1])
        # substitute \t by space
        line[1] = re.sub('\t', ' ', line[1])
        # the head is the first 3 elements of line[1].split(' ')
        head_ = line[1].split(' ')[1:4]
 
        head_coverage = float(line[1].split(' ')[4]) 
        std_confidence = float(line[1].split(' ')[5])
        pca_confidence = float(line[1].split(' ')[6])

        
        head = [head_[0][1],head_[1],head_[2][1]]
        head = str(head[1])+'('+str(head[0])+','+str(head[2])+')'

        # remove the elements that are spaces
        body = [x for x in body.split(' ') if x != '']

        num_preds = int(len(body)/3) 
        predicates = []

        for i in range(num_preds):
            lista = [body[3*i+1],body[3*i][1],body[3*i+2][1]]
            predicates.append( str(lista[0])+'('+str(lista[1])+','+str(lista[2])+')' )    

        # print all the info
        # print('std_confidence: ', std_confidence, 'body: ', predicates ,'head: ', head, )

        # filter by std_confidence
        if std_confidence >= std_confidence_threshold :
            rules.append(( head, predicates,std_confidence,pca_confidence,head_coverage))

# order the rules by std_confidence and then by pca_confidence and then by head coverage
rules = sorted( rules, key=lambda x: (x[2], x[3], x[4]), reverse=True)
#filter the number of rules
rules = rules[:max_rules]

for i,rule in enumerate(rules):
    if i<5000:
        print(rule,'\n')

# save the rules
with open(save_path+'.txt', 'w') as f:
    for i,rule in enumerate(rules):
        # predicates is the second element of the tuple, which is a list of strings. Join them with comma
        head = rule[0]
        predicates = ', '.join(rule[1])
        std_confidence = rule[2]
        pca_confidence = rule[3]
        string = 'r' + str(i) + ':' + str(std_confidence) + ':' + str(predicates) + ' -> ' + str(head)+'\n'
        f.write(string)
        # print(rule)
        # print(string)