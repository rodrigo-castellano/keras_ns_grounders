import copy
import time
from typing import List, Set, Tuple, Dict, Union
from ns_lib.logic.commons import Atom, Domain, Rule
from ns_lib.grounding.engine import Engine
from itertools import chain, product
import shutil
from typing import Dict, List, Tuple, Union, Iterable

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from typing import Dict, List
from ns_lib.logic.commons import Atom, FOL,RuleGroundings
import time
import argparse
import ns_lib as ns
from itertools import product
import numpy as np
from os.path import join
import random
import pickle

from dataset import KGCDataHandler, build_domains
from model import CollectiveModel
from keras.callbacks import CSVLogger
from ns_lib.logic.commons import Atom, Domain, Rule, RuleLoader
from ns_lib.nn.kge import KGEFactory
from ns_lib.utils import MMapModelCheckpoint, KgeLossFactory, read_file_as_lines
from ns_lib.grounding import  AtomIndex, get_atoms, get_atoms_on_groundings, backward_chaining_grounding_one_rule
from ns_lib.utils import get_arg

def PruneIncompleteProofs(rule2groundings: Dict[str, Set[Tuple[Tuple, Tuple]]],
                          rule2proofs:Dict[str, List[Tuple[Tuple, List[Tuple]]]],
                          fact_index: AtomIndex,
                          num_steps: int) ->  Dict[str, Set[Tuple[Tuple, Tuple]]]:
    #for rn,g in rule2groundings.items():
    #    print('RIN', rn, len(g))
    atom2proved: Dist[Tuple[str, str, str], bool] = {}

    # This loop iteratively finds the atoms that are already proved.
    for i in range(num_steps):
        for rule_name,proofs in rule2proofs.items():
            for query_and_proof in proofs:
                query, proof = query_and_proof[0], query_and_proof[1]
                if query not in atom2proved or not atom2proved[query]:
                    atom2proved[query] = all(
                        [atom2proved.get(a, False)
                         # This next check is useless as atoms added in the proofs
                         # are by definition not proved already in the data.
                         # or fact_index._index.get(a, None) is not None)
                         for a in proof])

    # Now atom2proved has all proved atoms. Scan the groundings and keep only
    # the ones that have been proved within num_steps:
    pruned_rule2groundings = {}
    for rule_name,groundings in rule2groundings.items():
        pruned_groundings = []
        for g in groundings:
            head_atoms = g[0]
            # WE CHECK IF ALL THE ATOMS IN THE HEAD ARE PROVED
            # all elements in the grounding are either in the training data
            # or they are provable using the rules,
            if all([(atom2proved.get(a, False) or
                     fact_index._index.get(a, None) is not None)
                    for a in head_atoms]):
                pruned_groundings.append(g)
        pruned_rule2groundings[rule_name] = set(pruned_groundings)
    #for rn,g in pruned_rule2groundings.items():
    #    print('ROUT', rn, len(g))
    return pruned_rule2groundings

def Prune_groundings_per_level(groundings_per_level,
                          rule2proofs:Dict[str, List[Tuple[Tuple, List[Tuple]]]],
                          fact_index: AtomIndex,
                          num_steps: int) ->  Dict[str, Set[Tuple[Tuple, Tuple]]]:
    #for rn,g in groundings_per_level.items():
    #    print('RIN', rn, g)
    # Here we will keep all the proves that are complete, i.e. all the atoms
    atom2proved: Dist[Tuple[str, str, str], bool] = {}
    # go through every atom to prove from all queries
    for i in range(num_steps):
        for rule_name,proofs in rule2proofs.items():
            for query_and_proof in proofs:
                query, proof = query_and_proof[0], query_and_proof[1]
                if query not in atom2proved or not atom2proved[query]:
                    atom2proved[query] = all(
                        [atom2proved.get(a, False) for a in proof])

    # Now atom2proved has all proved atoms. Scan the groundings and keep only
    # the ones that have been proved within num_steps:
    pruned_groundings_per_level = {}
    pruned_groundings = []
    for g in groundings_per_level:
        head_atoms = g[0]
        if all([(atom2proved.get(a, False) or
                    fact_index._index.get(a, None) is not None)
                for a in head_atoms]):                
            pruned_groundings.append(g)
    pruned_groundings_per_level = set(pruned_groundings)
    return pruned_groundings_per_level

def approximate_backward_chaining_grounding_one_rule(
    groundings_per_level,
    atoms_proof_last_level,
    step, 
    n_steps,
    domains: Dict[str, Domain],
    domain2adaptive_constants: Dict[str, List[str]],
    pure_adaptive: bool,
    rule: Rule,
    queries: List[Tuple],
    fact_index: AtomIndex,
    # Max number of unknown facts to expand:
    # 0 implements the known_body_grounder
    # -1 or body_size implements pure unrestricted backward_chaining (
    max_unknown_fact_count: int,
    res: Set[Tuple[Tuple, Tuple]]=None,
    # proof is a dictionary:
    # atom -> list of atoms needed to be proved in the application of the rule.
    proofs: Dict[Tuple[Tuple, Tuple], List[Tuple[Tuple, Tuple]]]=None) -> Union[
        None, Set[Tuple[Tuple, Tuple]]]:
    #start = time.time()
    # We have a rule like A(x,y) B(y,z) => C(x,z)
    assert len(rule.head) == 1, (
        'Rule is not a Horn clause %s' % str(rule))
    head = rule.head[0]
    build_proofs: bool = (proofs is not None)

    new_ground_atoms = set()
    start = time.time()
    
    # lim = 1
    # cont = 0
    for q in queries:
    #   groundings_per_query = 0
    #   cont += 1 
    #   print('\n\n***************q', q,'********************') if cont< lim else None
      if q[0] != head[0]:  # predicates must match.
        continue

      # Get the variable assignments from the head.
      head_ground_vars = {v: a for v, a in zip(head[1:], q[1:])}

      for i in range(len(rule.body)):
        # Ground atom by replacing variables with constants.
        # The result is the partially ground atom A('Antonio',None)
        # with None indicating unground variables.
        body_atom = rule.body[i]
        ground_body_atom = (body_atom[0], ) + tuple(
            [head_ground_vars.get(body_atom[j+1], None)
             for j in range(len(body_atom)-1)])
        # print('\n- i', i,'. ground_body_atom:', ground_body_atom, '. Substitution (by None) of the vars not present in head.') if cont< lim else None
        if all(ground_body_atom[1:]):
            groundings = (ground_body_atom,)
        else:
            # Tuple of atoms matching A(Antonio,None) in the facts.
            # This is the list of ground atoms for the i-th atom in the body.
            # groundings = fact_index.get_matching_atoms(ground_body_atom)
            groundings = fact_index._index.get(ground_body_atom, [])
            # print(' GROUNDINGS', groundings) if cont< lim else None

        if len(rule.body) == 1:
            # Shortcut, we are done, the clause has no free variables.
            # Return the groundings.
            # print('groundings already done, #all vars are subtituted', groundings) if cont< lim else None
            # print('ADDED', q, '->', (groundings,)) if cont< lim else None
            new_ground_atoms.add(((q,), groundings))
            continue

        for ground_atom in groundings:
            # print('     -GROUND ATOM', ground_atom) if cont< lim else None
            # This loop is only needed to ground at least one atom in the body
            # of the formula. Otherwise it would be enough to start with the
            # loop for ground_vars in product(...) but it would often expand
            # into many nodes. The current solution does not allow to fully
            # recover a classical backward_chaining, though.
            head_body_ground_vars = copy.copy(head_ground_vars)
            head_body_ground_vars.update(
                {v: a for v, a in zip(body_atom[1:], ground_atom[1:])})

            free_var2domain = [(v,d) for v,d in rule.vars2domain.items()
                               if v not in head_body_ground_vars]
            free_vars = [vd[0] for vd in free_var2domain]
            if pure_adaptive:
                ground_var_groups = [domain2adaptive_constants.get(vd[1], [])
                                     for vd in free_var2domain]
            elif domain2adaptive_constants is not None:
                ground_var_groups = [domains[vd[1]].constants +
                                     domain2adaptive_constants.get(vd[1], [])
                                     for vd in free_var2domain]
            else:
                ground_var_groups = [domains[vd[1]].constants
                                     for vd in free_var2domain]

            # Iterate over the groundings of the free vars.
            # If no free vars are available, product returns a single empty
            # tuple, meaning that we still correctly enter in the following
            # for loop for a single round.
            # print('     FREE VARS_SPAN', list(product(*[domains[vd[1]].constants for vd in free_var2domain])))
            for ground_vars in product(*ground_var_groups):
                var2ground = dict(zip(free_vars, ground_vars))
                full_ground_vars = {**head_body_ground_vars, **var2ground}
                # print('     FULL_VARS', full_ground_vars) if cont< lim else None

                accepted: bool = True
                body_grounding = []
                if build_proofs:
                    body_grounding_to_prove = []
                unknown_fact_count: int = 0
                for j in range(len(rule.body)):
                    if i == j:
                        # print('         -j=i') if cont< lim else None
                        new_ground_atom = ground_atom
                        # by definition as it is coming from the groundings.
                        is_known_fact = True
                    else:
                        body_atom2 = rule.body[j]
                        new_ground_atom = (body_atom2[0], ) + tuple(
                            [full_ground_vars.get(body_atom2[k+1], None)
                             for k in range(len(body_atom2)-1)])
                        if new_ground_atom == q:
                            # print('         -j=',j,'NEW GROUND ATOM', new_ground_atom, ' Same atom as query, discard') if cont< lim else None
                            accepted = False
                            break
                        is_known_fact = (fact_index._index.get(
                            new_ground_atom, None) is not None)

                    assert all(new_ground_atom), (
                        'Unexpected free variables in %s' %
                        str(new_ground_atom))
                    if not is_known_fact and (
                            max_unknown_fact_count < 0 or
                            unknown_fact_count < max_unknown_fact_count):
                        body_grounding.append(new_ground_atom)
                        if build_proofs:
                            body_grounding_to_prove.append(new_ground_atom)
                        unknown_fact_count += 1
                        # print('         -j=',j,'NEW GROUND ATOM', new_ground_atom, '. Is known_fact:',is_known_fact,'. Accepted. We have to prove it') if cont< lim else None
                    elif is_known_fact:
                        body_grounding.append(new_ground_atom)
                        # print('         -j=',j,'NEW GROUND ATOM', new_ground_atom, '. Is known_fact:',is_known_fact,'. Accepted') if cont< lim else None
                    else:
                        # print('         -j=',j,'NEW GROUND ATOM', new_ground_atom, '. Is known_fact:',is_known_fact,'. Discard',unknown_fact_count,'/', max_unknown_fact_count) if cont< lim else None
                        accepted = False
                        break

                if accepted:
                    # print('     ADDED', q, '->', tuple(body_grounding)) if cont< lim else None
                    # print('ADDED', q, '->', tuple(body_grounding), 'TO_PROVE',          str(body_grounding_to_prove) if build_proofs else '') if len(body_grounding_to_prove)>0 else None
                    new_ground_atoms.add(((q,), tuple(body_grounding)))
                    # groundings_per_query +=1
                    if build_proofs:
                        proofs.append((q, body_grounding_to_prove))
                        if step == n_steps-1:
                            print('ATOMS TO PROVE', body_grounding_to_prove) 
                            atoms_proof_last_level.update(body_grounding_to_prove) if len(body_grounding_to_prove) > 0 else None
                            print('atoms_proof_last_level',atoms_proof_last_level)

    #   print('NUM_GROUNDINGS for the query',q, groundings_per_query) #, 'TIME', end - start)
    #   groundings_numbers.append(groundings_per_query)
    #   print('       AVG_NUM_GROUNDINGS', sum(groundings_numbers)/len(groundings_numbers))
    # print('AVG_NUM_GROUNDINGS', sum(groundings_numbers)/len(groundings_numbers))

    end = time.time()
    # print('NUM GROUNDINGS', len(new_ground_atoms),'. TIME', end - start)
    # print('NUM ATOMS TO PROVE', len(atoms_proof_last_level))

    # print('NEW GROUND ATOMS', new_ground_atoms) if cont< lim else None

    if step not in groundings_per_level:
        groundings_per_level[step] = set()
    if new_ground_atoms is not None:
        for g in new_ground_atoms:
            groundings_per_level[step].add(g)

    # print('Unique groundings per level, level',step,':',len(groundings_per_level[step])) if step in groundings_per_level else None

    if res is None:
        return new_ground_atoms
    else:
        res.update(new_ground_atoms)



class ApproximateBackwardChainingGrounder(Engine):

    def __init__(self, rules: List[Rule], facts: List[Union[Atom, str, Tuple]],
                 domains: Dict[str, Domain],
                 domain2adaptive_constants: Dict[str, List[str]]=None,
                 pure_adaptive: bool=False,
                 max_unknown_fact_count: int=1,
                 max_unknown_fact_count_last_step: int=1,
                 num_steps: int=1,
                 max_groundings_per_rule: int=-1,  # to speedup the computation.
                 # Whether the groundings should be accumulated across calls.
                 accumulate_groundings: bool=False,
                 prune_incomplete_proofs: bool=True):
        self.max_unknown_fact_count = max_unknown_fact_count
        self.max_unknown_fact_count_last_step = max_unknown_fact_count_last_step
        self.num_steps = num_steps
        self.accumulate_groundings = accumulate_groundings
        self.prune_incomplete_proofs = prune_incomplete_proofs
        self.max_groundings_per_rule = max_groundings_per_rule
        self.rules = rules
        self.domains = domains
        self.domain2adaptive_constants = domain2adaptive_constants
        self.pure_adaptive = pure_adaptive
        self.facts = [a if isinstance(a,Tuple) else a.toTuple()
                      if isinstance(a,Atom) else Atom(s=a).toTuple()
                      for a in facts]
        # self.facts = facts
        for rule in self.rules:
            assert len(rule.head) == 1, (
                '%s is not a Horn clause' % str(rule))
        self._fact_index = AtomIndex(self.facts)
        self.relation2queries = {}
        self.rule2groundings = {}
        self.rule2proofs = {}

    def _init_internals(self, queries: List[Tuple], clean: bool):
        # this tell us the queries for each relation to analyse.
        self.relation2queries = {}  # reset
        for q in queries:
            if q[0] not in self.relation2queries:
                self.relation2queries[q[0]] = set()
            self.relation2queries[q[0]].add(q)

        # If clean=False, groundings are incrementally added.
        for rule in self.rules:
            if clean or rule.name not in self.rule2groundings:
                self.rule2groundings[rule.name] = set()
            # if clean or rule.name not in self.rule2proofs:
                self.rule2proofs[rule.name] = []

    # Ground a batch of queries, the result is cached for speed.
    #@profile
    def ground(self,
               facts: List[Tuple],
               queries: List[Tuple],
               **kwargs) -> Dict[str, RuleGroundings]:
        print('\n------------------------------------------------------------------------------------------------\n')
        if self.rules is None or len(self.rules) == 0:
            return []
        self.groundings_per_level = {}
        self.atoms_proof_last_level = set()
        # When accumulating groundings, we keep a single large set of
        # groundings that are reused over all batches.
        self._init_internals(queries, clean=(not self.accumulate_groundings))
        # order also the relation2queries
        # for k,v in self.relation2queries.items():
        #     self.relation2queries[k] = sorted(list(v), key=lambda x: (x[0], x[1:])) if len(v) < 50 else v
        # print('\nAtoms to process per query. self.relation2queries\n',self.relation2queries)
        # Keeps track of the queris already processed for this rule.
        self._rule2processed_queries = {rule.name: set() for rule in self.rules}
        # groundings_numbers = []
        for step in range(self.num_steps):
            # print('\n\n\nSTEP NUMBER ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', step,'^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^','step ',step,'/', self.num_steps, 'known body',step == self.num_steps - 1, )
            for j,rule in enumerate(self.rules):
                # print('\nrule ', rule, ' """"""""""""""""""""""""""""""""""""""""" """""""""""""""""""""""""" ')
                # Here we assume to have a Horn clause, fix it.
                queries_per_rule = list(
                    self.relation2queries.get(rule.head[0][0], set()))
                # print('\nqueries_per_rule\n',len(queries_per_rule), queries_per_rule)
                if not queries_per_rule:
                    continue
                approximate_backward_chaining_grounding_one_rule(
                    self.groundings_per_level,
                    self.atoms_proof_last_level,
                    step,
                    self.num_steps,
                    self.domains,
                    self.domain2adaptive_constants,
                    self.pure_adaptive,
                    rule, queries_per_rule, self._fact_index,
                    # max_unknown_fact_count
                    (self.max_unknown_fact_count if step < self.num_steps - 1
                     else self.max_unknown_fact_count_last_step),
                    # Output added here.
                    res=self.rule2groundings[rule.name],
                    # Proofs added here.
                    proofs=(self.rule2proofs[rule.name]
                            if self.prune_incomplete_proofs else None),
                    # groundings_numbers=groundings_numbers
                    )
                # Update the list of processed rules.
                # print('\nTotal groundings in res after rule',j,'/',len(self.rules),', step',step,':',sum([len(v) for k, v in self.rule2groundings.items()])) # IS IS MORE THAN THE GROUNDINGS_per_level BECAUSE THERE ARE DUPLICATES
                self._rule2processed_queries[rule.name].update(queries_per_rule)
                # print('\nqueries processed:, _rule2processed_queries\n', len(self._rule2processed_queries[rule.name]),self._rule2processed_queries[rule.name])
                # print()
                # for k,v in self.rule2groundings.items():
                #     print('rule2groundings', k, len(v))#,v)
 
            if step == self.num_steps - 1:
                break

            # Get the queries for the next iteration.
            new_queries = set()
            for rule in self.rules:
                groundings = self.rule2groundings[rule.name]
                # Get the new queries left to prove, these facts that are not
                # been processed already and that are not known facts.
                new_queries.update(
                    [a for a in get_atoms_on_groundings(groundings)
                     if a not in self._rule2processed_queries[rule.name] and
                     self._fact_index._index.get(a, None) is None])
            # print('\nNEW Q',len(new_queries),'\n', list(new_queries), ' FROM groundings', len(groundings))
            # Here we update the queries to process in the next iteration, we only keep the new ones.
            self._init_internals(list(new_queries), clean=False)

        # print('Num groundings',sum([len(v) for k, v in self.rule2groundings.items()]))
        if self.prune_incomplete_proofs:
            # check all the groundings with at least 1 atom missing, to see if they are proved (all atoms present in the facts)
            # print('\nstarting PruneIncompleteProofs')
            self.rule2groundings = PruneIncompleteProofs(self.rule2groundings,
                                                         self.rule2proofs,
                                                         self._fact_index,
                                                         self.num_steps)
            print('\nNum groundings after pruning',sum([len(v) for k, v in self.rule2groundings.items()]))
        # print('\nFinal groundings\n')
        # This should be done after sorting the groundings to ensure the output
        # to be deterministic.
        if self.max_groundings_per_rule > 0:
            self.rule2groundings = {rule_name:set(list(groundings)[:self.max_groundings_per_rule])
                                    for rule_name,groundings in self.rule2groundings.items()}
                    


        for level in range(self.num_steps):
            # if the level is in the keys of groundings_per_level, prune the groundings
            if level in self.groundings_per_level:
                # print the keys of the self.groundings_per_level
                print('\nNum groundings in level',level,',',len(self.groundings_per_level[level]))
                if self.prune_incomplete_proofs:
                    self.groundings_per_level[level] = Prune_groundings_per_level(self.groundings_per_level[level],
                                                                self.rule2proofs,
                                                                self._fact_index,
                                                                self.num_steps)
                    print('Num groundings in level',level,', after pruning,',len(self.groundings_per_level[level]))



        #print('R', self.rule2groundings)
        if 'deterministic' in kwargs and kwargs['deterministic']:
            ret = {rule_name: RuleGroundings(
                rule_name, sorted(list(groundings), key=lambda x : x.__repr__()))
                   for rule_name,groundings in self.rule2groundings.items()}
        else:
            ret = {rule_name: RuleGroundings(rule_name, list(groundings))
                   for rule_name,groundings in self.rule2groundings.items()}
        print('\n------------------------------------------------------------------------------------------------\n')
        return ret, self.groundings_per_level, self.rule2groundings, self.atoms_proof_last_level
 

def find_atom_in_groundings(atom, groundings): 
    # this, in case of pruning, checks if the atom is still present in the groundings
    # Only on the body
    for grounding in groundings:
        for body_atom in grounding[1]:
            if body_atom == atom:
                return True
    return False

def find_cutoff(atoms_remove,data_handler,threshold=0.4):
    # from atoms_remove, count how many times each predicate appears, and divide it by the total number of atoms in the facts with that predicate
    pred_counts_to_remove = {}
    for  atom in atoms_remove:
        pred = atom[0]
        if pred not in pred_counts_to_remove:
            pred_counts_to_remove[pred] = 0
        pred_counts_to_remove[pred] += 1

    # calculate the number of times that predicate appears in train_known_facts_set
    pred_counts_facts = {}
    for atom in data_handler.train_known_facts_set:
        pred = atom[0]
        if pred not in pred_counts_facts:
            pred_counts_facts[pred] = 0
        pred_counts_facts[pred] += 1
    
    # calculate the % of atoms to remove out of the total number of atoms in the facts with that predicate
    pred_cutoff = {}
    max_atoms_allowed = {}
    print('Threshold per predicate:')
    for pred in pred_counts_to_remove:
        max_atoms_allowed[pred] = int(threshold*pred_counts_facts[pred])
        pred_cutoff[pred] = pred_counts_to_remove[pred] / pred_counts_facts[pred]
        print('     ',pred, 'percentage of atoms with that predicate in facts: ', np.round(pred_cutoff[pred],3),'. ', pred_counts_to_remove[pred], '/', pred_counts_facts[pred])
    return max_atoms_allowed,pred_cutoff,pred_counts_to_remove,pred_counts_facts

def write_dataset(base_path, dest_path, args, data_handler, new_facts, new_train, ctes_facts, num_steps, prune_backward):
    if not os.path.exists(dest_path): os.mkdir(dest_path)
    facts_file = join(dest_path,'facts.txt')
    train_file = join(dest_path,'train.txt')
    test_file = join(dest_path,'test.txt')
    valid_file = join(dest_path,'valid.txt')
    domain_file = join(dest_path,'domain2constants.txt')

    with open(train_file, 'w') as f:
        for train_fact in new_train:
            f.write(str(train_fact[0])+'('+str(train_fact[1])+','+str(train_fact[2])+').\n')
    if len(new_facts) > 0:
        with open(facts_file, 'w') as f:
            for fact in new_facts:
                f.write(str(fact[0])+'('+str(fact[1])+','+str(fact[2])+').\n')

    # write a new test and valid files, skip the atoms with constants not present in ctes_facts
    with open(test_file, 'w') as f:
        for test_fact in data_handler.test_facts:
            if test_fact[1] in ctes_facts and test_fact[2] in ctes_facts:
                f.write(str(test_fact[0])+'('+str(test_fact[1])+','+str(test_fact[2])+').\n')
    with open(valid_file, 'w') as f:
        for valid_fact in data_handler.valid_facts:
            if valid_fact[1] in ctes_facts and valid_fact[2] in ctes_facts:
                f.write(str(valid_fact[0])+'('+str(valid_fact[1])+','+str(valid_fact[2])+').\n')
    # Write the domain file 
    domain_ctes = data_handler.domain2constants
    # Now I have to selec from domain_ctes only the constants that are present in the new_facts
    domain_ctes_write = {}
    for k,ctes in domain_ctes.items():
        domain_ctes_write[k] = [cte for cte in ctes if cte in ctes_facts]    

    with open(domain_file, 'w') as f:
        for k,ctes in domain_ctes_write.items():
            if k!='default':
                f.write(k+' ')
                [f.write(cte+' ') for cte in ctes] 
                f.write('\n')
    
        # copy the rules file
        rules_file = join(base_path, args.dataset_name, args.rules_file)
        dest_rules_file = join(dest_path,args.rules_file)   
        shutil.copyfile(rules_file, dest_rules_file)



def main(base_path, output_filename, kge_output_filename, log_filename, args):

    print('\nARGS', args,'\n')

    seed = get_arg(args, 'seed_run_i', 0)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
  
    # Data Loading
    data_handler = KGCDataHandler(
        dataset_name=args.dataset_name,
        base_path=base_path,
        format=get_arg(args, 'format', None, True),
        domain_file= args.domain_file,
        train_file= args.train_file,
        valid_file=args.valid_file,
        test_file= args.test_file,
        fact_file= args.facts_file)
    
    dataset_train = data_handler.get_dataset(split="train",number_negatives=args.num_negatives)
    dataset_test = data_handler.get_dataset(split="test", corrupt_mode='TAIL',number_negatives=args.test_negatives)
    
    fol = data_handler.fol 

    enable_rules = (args.reasoner_depth > 0 and args.num_rules > 0)
    if enable_rules: 
        rules = ns.utils.read_rules(join(base_path, args.dataset_name, args.rules_file),args)

        if 'backward' in args.grounder:
            num_steps = int(args.grounder.split('_')[-1])
            prune_backward = True  
            print('Using backward chaining with %d steps' % num_steps)
            engine = ApproximateBackwardChainingGrounder(
                rules, facts=fol.facts, domains={d.name:d for d in fol.domains},
                domain2adaptive_constants=None,
                pure_adaptive=False,
                num_steps=num_steps,
                max_unknown_fact_count=1,
                max_unknown_fact_count_last_step=0,
                max_groundings_per_rule=-1,
                prune_incomplete_proofs=True)

    else:
        rules = []
        engine = None
 
    queries, labels = dataset_test[0:len(dataset_test)]
    facts = fol.facts
    rules = engine.rules


    # # For every rule, get the predicate in the head. Count the number of times each predicate appears
    # pred_head_facts_counts = {}
    # for rule in rules:
    #     pred = rule.head[0][0]
    #     if pred not in pred_head_facts_counts:
    #         pred_head_facts_counts[pred] = 0
    #     pred_head_facts_counts[pred] += 1

    # # print the number of times each predicate appears in the head of the rules
    # print('pred_head_facts_counts', pred_head_facts_counts)
    # pred_counts = sorted(pred_head_facts_counts, key=pred_head_facts_counts.get, reverse=True)

    groundings,groundings_per_level, all_groundings,atoms_proof_last_level = engine.ground(tuple(facts), tuple(ns.utils.to_flat(queries)), deterministic=True)
    rules = engine.rules
    print('Total groundings: ', sum( [len(v) for k,v in all_groundings.items()]))
    print('Total groundings as sum by level', sum([len(v) for k,v in groundings_per_level.items()]))

    print('Groundings per level')
    print( [(k,len(v)) for k,v in groundings_per_level.items()])

    # Remove atoms present in grondings of every level, except for the last (we need them)
    atoms_remove = set()
    for level in range(num_steps-1): # except for the last step
        # get all the atoms from the body
        for grounding in groundings_per_level[level]:
            for body_atom in grounding[1]:
                atoms_remove.add(body_atom)
    
    # Dont remove atoms if they are in the proof of the last level
    # with approximate, I need to adapt this so that if in the last level max 1 atom is missing, still append it
    atoms_last_level = set()
    if num_steps-1 in groundings_per_level and len(groundings_per_level[num_steps-1])!=0 :
        for grounding in groundings_per_level[num_steps-1]:
            for body_atom in grounding[1]:
                if body_atom in atoms_remove:
                    atoms_last_level.add(body_atom)

    # Dont remove atoms from the test set
    atoms_in_test = set()
    for fact in data_handler.test_facts_set:
        if fact in atoms_remove:
            atoms_in_test.add(fact)
    
    # If I have too many atoms to remove, I can exclude those that are the head of a rule, because those will lead to more proofs


    # # Dont remove atoms that imply that, in the levels except for the last one, there is more than 1 atom missing
    # # ERROR!!! AQUI NO TENGO QUE MIRAR GROUNDINGS PER LEVEL, SI NO QUE TENGO QUE MIRAR LA PRUEBA
    # atoms_proof_inter_levels = set()
    # for level in range(num_steps-1): 
    #     for grounding in groundings_per_level[level]:
    #         sum_atoms_removed = 0
    #         atoms_problematic = set()
    #         for body_atom in grounding[1]:
    #             sum_atoms_removed += 1 if body_atom in atoms_remove else 0
    #             atoms_problematic.add(body_atom)
    #         if sum_atoms_removed > 1:
    #             atoms_proof_inter_levels.update(atoms_problematic)

    # UNA COSA ES QUITAR ATOMOS DE GROUNDINGS DE NIVELES INTERMEDIOS, Y OTRA A PARTE ES CONSERVAR ATOMS EN LAS PRUEBAS DEL ULTIMO NIVEL

    final_atoms_remove = atoms_remove - atoms_last_level - atoms_in_test - atoms_proof_last_level
    print('\nTotal atoms to remove:', len(atoms_remove))
    print('Total atoms to exclude that are present in the last level:', len(atoms_last_level))
    print('Total atoms to exclude that are present in the test set:', len(atoms_in_test))
    print('Number of atoms used in the proofs for the last level:', len(atoms_proof_last_level))
    # print('Total atoms to exclude from the proof in the intermediate level:', len(atoms_proof_inter_levels))
    print('\nFinal total atoms to remove: ', len(final_atoms_remove))

    print('Initial number of facts:', len(facts), '. New number of facts:', len(set(facts)-final_atoms_remove))
    # Do the grounding again to compare the original version with the new version: 
    engine = ApproximateBackwardChainingGrounder(
        rules, facts=list(set(facts)-final_atoms_remove), domains={d.name:d for d in fol.domains},
        domain2adaptive_constants=None,
        pure_adaptive=False,
        num_steps=num_steps,
        max_unknown_fact_count=1,
        max_unknown_fact_count_last_step=0,
        max_groundings_per_rule=-1,
        prune_incomplete_proofs=True)
    # Facts is a list, final_atoms_remove is a set. Convert the substraction to a tuple
    _,groundings_per_level, all_groundings,_ = engine.ground(tuple(set(facts)-final_atoms_remove), tuple(ns.utils.to_flat(queries)), deterministic=True)

    print('\nTotal groundings with new set of facts', sum( [len(v) for k,v in all_groundings.items()]))
    print('Groundings per level:', [(k,len(v)) for k,v in groundings_per_level.items()])