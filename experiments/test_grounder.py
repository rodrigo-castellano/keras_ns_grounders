import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir, '..', 'ns_lib'))

import random
from os.path import join
from typing import Dict, List, Tuple, Union, Iterable
import numpy as np
import tensorflow as tf
import ns_lib as ns
from ns_lib.grounding import *
from model_utils import * 
from ns_lib.logic.commons import Atom, FOL,RuleGroundings
from dataset import KGCDataHandler
import argparse
from ns_lib.grounding.grounder_factory import BuildGrounder

parser = argparse.ArgumentParser(description='Description of your script')  
args = parser.parse_args()

args.grounder = 'backward_1_2'
# args.grounder = 'backwardnoprune_2_1'

# args.dataset_name = 'wn18rr'
# args.dataset_name = 'kinship_family'
args.dataset_name = 'FB15k237'
# args.dataset_name = 'dummy'
args.data_path = "experiments/data"
args.facts_file = 'facts.txt'
args.train_file = 'train.txt'  
args.valid_file = 'valid.txt'
args.test_file = 'test.txt'
args.domain_file = 'domain2constants.txt'
args.rules_file = 'rules_amie.txt' if (args.dataset_name == 'kinship_family' or args.dataset_name == 'FB15k237' ) else 'rules.txt'

args.num_negatives = 0
args.test_negatives = 0

args.format = "functional"

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Data Loading
data_handler = KGCDataHandler(
    dataset_name=args.dataset_name,
    base_path=args.data_path,
    format=args.format,
    domain_file= args.domain_file,
    train_file= args.train_file,
    valid_file=args.valid_file,
    test_file= args.test_file,
    fact_file= args.facts_file)


dataset_train = data_handler.get_dataset(split="train",number_negatives=args.num_negatives)
dataset_test = data_handler.get_dataset(split="test", corrupt_mode='TAIL',number_negatives=args.test_negatives)

fol = data_handler.fol 
facts = fol.facts
rules = ns.utils.read_rules(join(args.data_path, args.dataset_name, args.rules_file),args)
queries, labels = dataset_test[0:len(dataset_test)]

# type = args.grounder
# backward_width = None
# if type.count('_') == 2:
#     backward_width = int(type[type.index('_')+1]) # take the first character after the first '_'
#     backward_depth = int(type[-1])
#     type = 'ApproximateBackwardChainingGrounder'
# else:
#     backward_depth = int(type[-1])
#     type = 'BackwardChainingGrounder'

# prune_incomplete_proofs = True #if (backward_width is None or backward_width == 0) else False
# print('Grounder: ',args.grounder,'backward_depth:', backward_depth, 'Prune:', prune_incomplete_proofs, 'backward_width:', backward_width)

# if type == 'BackwardChainingGrounder':
#     engine = BackwardChainingGrounder(
#                 rules, facts=facts, domains={d.name:d for d in fol.domains},
#                 domain2adaptive_constants=None,
#                 pure_adaptive=False,
#                 num_steps=backward_depth)
# elif type == 'ApproximateBackwardChainingGrounder':
#     engine = ApproximateBackwardChainingGrounder(
#                 rules, facts=facts, domains={d.name:d for d in fol.domains},
#                 domain2adaptive_constants=None,
#                 pure_adaptive=False,
#                 num_steps=backward_depth,
#                 max_unknown_fact_count=backward_width,
#                 max_unknown_fact_count_last_step=backward_width,
#                 prune_incomplete_proofs=prune_incomplete_proofs)


engine = BuildGrounder(args, rules, facts=facts, fol=fol, domain2adaptive_constants=None)


ground_formulas = engine.ground(tuple(facts),tuple(ns.utils.to_flat(queries)),deterministic=True)
# print('ground_formulas:',ground_formulas)
queries = queries[:1]
# queries = [[('locatedInCR','luxembourg','europe')]]
print('number of queries:',len(queries))

len_groundings = []
n_queries_with_groundings = 0
for query in queries:
    # print('\nquery:',query)
    ground_formulas = engine.ground(tuple(facts),tuple(ns.utils.to_flat(query)),deterministic=True)

    print('num groundings:',len([grounding for rule in ground_formulas for grounding in ground_formulas[rule]]))
    len_groundings.append(len([grounding for rule in ground_formulas for grounding in ground_formulas[rule]]))
    n_queries_with_groundings += 1 if len_groundings[-1] > 0 else 0
    # print('ground_formulas:',ground_formulas)

    # print('\n\nground_formulas:')
    # for rule in ground_formulas:
    #     print('Rule:',rule)#,ground_formulas[rule])
    #     for grounding in ground_formulas[rule]:
    #         print(grounding[0][0],'       ',grounding[1][0], grounding[1][1])

print('avg number of grounding:',round(np.mean(len_groundings),3), 'std:',round(np.std(len_groundings),3))
print('converage:',round(n_queries_with_groundings/len(queries),3))

# ground_formulas = engine.ground(tuple(facts),tuple(ns.utils.to_flat(queries)),deterministic=True)
# print('num groundings:',len([grounding for rule in ground_formulas for grounding in ground_formulas[rule]]))
