import copy
import time
from typing import List, Set, Tuple, Dict, Union
from keras_ns.logic.commons import Atom, Domain, Rule
from keras_ns.grounding.engine import Engine
from itertools import chain, product

from typing import Dict, List, Tuple, Union, Iterable

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from typing import Dict, List
from keras_ns.logic.commons import Atom, FOL,RuleGroundings
import time
import argparse
import keras_ns as ns
from itertools import product
import numpy as np
from os.path import join
import random
import pickle

from dataset import KGCDataHandler, build_domains
from model import CollectiveModel
from keras.callbacks import CSVLogger
from keras_ns.logic.commons import Atom, Domain, Rule, RuleLoader
from keras_ns.nn.kge import KGEFactory
from keras_ns.utils import MMapModelCheckpoint, KgeLossFactory, read_file_as_lines
from keras_ns.grounding import  AtomIndex, get_atoms, get_atoms_on_groundings, backward_chaining_grounding_one_rule
 

def PruneIncompleteProofs(rule2groundings: Dict[str, Set[Tuple[Tuple, Tuple]]],
                          rule2proofs:Dict[str, List[Tuple[Tuple, List[Tuple]]]],
                          fact_index: AtomIndex,
                          num_steps: int) ->  Dict[str, Set[Tuple[Tuple, Tuple]]]:
    #for rn,g in rule2groundings.items():
    #    print('RIN', rn, g)
    # Here we will keep all the proves that are complete, i.e. all the atoms
    atom2proved: Dist[Tuple[str, str, str], bool] = {}
    # go through every atom to prove from all queries
    for i in range(num_steps ):
        for rule_name,proofs in rule2proofs.items():
            for query_and_proof in proofs:
                query, proof = query_and_proof[0], query_and_proof[1]
                if query not in atom2proved or not atom2proved[query]:
                    atom2proved[query] = all(
                        [atom2proved.get(a, False) for a in proof])

    # Now atom2proved has all proved atoms. Scan the groundings and keep only
    # the ones that have been proved within num_steps:
    pruned_rule2groundings = {}
    for rule_name,groundings in rule2groundings.items():
        pruned_groundings = []
        for g in groundings:
            head_atoms = g[0]
            # WE CHECK IF ALL THE ATOMS IN THE HEAD ARE PROVED
            if all([(atom2proved.get(a, False) or
                     fact_index._index.get(a, None) is not None)
                    for a in head_atoms]):                pruned_groundings.append(g)
        pruned_rule2groundings[rule_name] = set(pruned_groundings)
    #for rn,g in pruned_rule2groundings.items():
    #    print('ROUT', rn, g)
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
    for i in range(num_steps ):
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
                for a in head_atoms]):                pruned_groundings.append(g)
    pruned_groundings_per_level = set(pruned_groundings)
    #for rn,g in pruned_groundings_per_level.items():
    #    print('ROUT', rn, g)
    return pruned_groundings_per_level


# res is a Set of (Tuple_head_groundings, Tuple_body_groundings)
def backward_chaining_grounding_one_rule_with_domains(
    res_full,
    pred_counts,
    atoms_remove,
    groundings_per_level,
    domains: Dict[str, Domain],
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
    proofs: Dict[Tuple[Tuple, Tuple], List[Tuple[Tuple, Tuple]]]=None,step=-1, n_steps=-1) -> Union[
        None, Set[Tuple[Tuple, Tuple]]]:
    
    start = time.time()
    # We have a rule like A(x,y) B(y,z) => C(x,z)
    assert len(rule.head) == 1, (
        'Rule is not a Horn clause %s' % str(rule))
    head = rule.head[0]
    build_proofs: bool = (proofs is not None)

    new_ground_atoms = set()
    pred_counts_ordered = sorted(pred_counts, key=pred_counts.get, reverse=True)
    count_groundings = 0
    cont = 0
    lim=10000
    for q in queries:
      cont += 1 
    #   print('\n\n***************q', q,'********************') if cont< lim else None
      if q[0] != head[0]:  # predicates must match.
        continue
      # Get the variable assignments from the head.
      head_ground_vars = {v: a for v, a in zip(head[1:], q[1:])}
      for i in range(len(rule.body)):
        body_atom = rule.body[i]
        ground_body_atom = (body_atom[0], ) + tuple(
            [head_ground_vars.get(body_atom[j+1], None)
             for j in range(len(body_atom)-1)])
        # print('\n- i', i,'. ground_body_atom:', ground_body_atom, '. Substitution (by None) of the vars not present in head.') if cont< lim else None
        if all(ground_body_atom[1:]):
            groundings = (ground_body_atom,)
        else:
            groundings = fact_index._index.get(ground_body_atom, [])
            # print(' GROUNDINGS', groundings) if cont< lim else None
        if len(rule.body) == 1:
            # Shortcut, we are done, the clause has no free variables.
            new_ground_atoms.add(((q,), groundings))
            continue
        for ground_atom in groundings:
            # print('     -GROUND ATOM', ground_atom)
            head_body_ground_vars = copy.copy(head_ground_vars)
            head_body_ground_vars.update(
                {v: a for v, a in zip(body_atom[1:], ground_atom[1:])})

            free_var2domain = [(v,d) for v,d in rule.vars2domain.items()
                               if v not in head_body_ground_vars]
            free_vars = [vd[0] for vd in free_var2domain]
            # print('     FREE VARS_SPAN', list(product(*[domains[vd[1]].constants for vd in free_var2domain])))
            for ground_vars in product(
                *[domains[vd[1]].constants for vd in free_var2domain]):
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
                        # print('         -j=i')
                        new_ground_atom = ground_atom
                        # by definition as it is coming from the groundings.
                        is_known_fact = True
                    else:
                        body_atom2 = rule.body[j]
                        new_ground_atom = (body_atom2[0], ) + tuple(
                            [full_ground_vars.get(body_atom2[k+1], None)
                             for k in range(len(body_atom2)-1)])
                        if new_ground_atom == q:
                            # print('         -j=',j,'NEW GROUND ATOM', new_ground_atom, ' Same atom as query, discard')
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
                        # print('         -j=',j,'NEW GROUND ATOM', new_ground_atom, '. Is known_fact:',is_known_fact,'. Accepted. We have to prove it')
                    elif is_known_fact:
                        body_grounding.append(new_ground_atom)
                        # print('         -j=',j,'NEW GROUND ATOM', new_ground_atom, '. Is known_fact:',is_known_fact,'. Accepted')
                    else:
                        # print('         -j=',j,'NEW GROUND ATOM', new_ground_atom, '. Is known_fact:',is_known_fact,'. Discard',unknown_fact_count,'/', max_unknown_fact_count)
                        accepted = False
                        break
                if accepted:
                    count_groundings += 1
                    # print('ADDED', q, '->', tuple(body_grounding), 'TO_PROVE',          str(body_grounding_to_prove) if build_proofs else '')
                    # print('     ADDED', q, '->', tuple(body_grounding)) if cont< lim else None
                    new_ground_atoms.add(((q,), tuple(body_grounding)))
                    if build_proofs:
                        proofs.append((q, body_grounding_to_prove))
                    # CHECK IF ALL THE ATOMS ARE KNOWN. IF YES, CHOOSE ONE OF THEM TO REMOVE. CHOOSE ONLY THE ONES THAT ARE HEAD OF RULES
                    if unknown_fact_count == 0:
                        # print('\nALL ATOMS KNOWN', body_grounding) if cont< lim else None
                        possible_atoms = [atom for atom in body_grounding if atom[0] in pred_counts]
                        # print('POSSIBLE ATOMS', possible_atoms) if cont< lim else None
                        if len(possible_atoms) == 0:
                            break
                        # for each predicate, go through all the atoms.if the predicate is the same as the most frequent predicate, remove it
                        for pred in pred_counts_ordered:
                            atom_found = False
                            for atom in possible_atoms: 
                                # print('     Predicate', pred, 'Atom', atom) if cont< lim else None
                                # if the predicate is the same as the most frequent predicate, remove it
                                if atom[0] == pred:
                                    # print('     (fully found. ADDED', q, '->', tuple(body_grounding)) if cont< lim else None
                                    # print('     Predicate', pred, 'Atom', atom,'. Remove') if cont< lim else None
                                    if step not in atoms_remove:
                                        atoms_remove[step] = set()
                                    atoms_remove[step].add(atom)
                                    atom_found = True
                                    break
                            if atom_found:
                                break
    # print('Number of groundings gone through', count_groundings) if cont< lim else None
    end = time.time()
    res.update(new_ground_atoms) if res is not None else None       

    if step not in groundings_per_level:
        groundings_per_level[step] = set()
    if new_ground_atoms is not None:
        for g in new_ground_atoms:
            groundings_per_level[step].add(g)
    # print('New groundings: ', len(new_ground_atoms)) if len(new_ground_atoms) >0 else None
    # print('Atoms to remove so far:',len(atoms_remove[step])) if step in atoms_remove else None
    # print('Unique groundings per level, level',step,':',len(groundings_per_level[step])) 
    # print('Groundings in res for this rule', len(res)) if res is not None else None
    # Order the groundings in groundings_levels
    # ordered = sorted(groundings_levels[step], key=lambda x: (x[0], x[1][0],x[1][1])) if len(groundings_levels[step]) > 1 else groundings_levels[step]
            
    # THIS IS JUST TO SEE THE UNIQUE GROUNDINGS IN RES
    # unique_groundings_in_res = set()
    # if res_full is not None:
    #     for rule in res_full:
    #         for i,g in enumerate(res_full[rule]):
    #             # print('     Grounding',i, grounding) if cont< lim else None
    #             unique_groundings_in_res.add(g)
    # # I also need to take into acc the ones from this level
    # for g in new_ground_atoms:
    #     unique_groundings_in_res.add(g)
    # # print('Unique groundings in res', len(unique_groundings_in_res)) if res is not None else None

    # AFTER I UPDATE RES, THE NUMBER OF GROUNDINGS IN self.rule2groundings IS GREATER THAN THE NUMBER OF GROUNDINGS IN RES. I NEED TO REMOVE THE DUPLICATES
    if res is None:
        return new_ground_atoms,atoms_remove,groundings_per_level
    else:
        res,atoms_remove,groundings_per_level


class BackwardChainingGrounder(Engine):

    def __init__(self, rules: List[Rule], facts: List[Union[Atom, str, Tuple]],
                 domains: Dict[str, Domain],
                 max_unknown_fact_count: int=1,
                 max_unknown_fact_count_last_step: int=0,
                 num_steps: int=1,
                 prune_incomplete_proofs: bool=True):
        self.max_unknown_fact_count = max_unknown_fact_count
        self.max_unknown_fact_count_last_step = max_unknown_fact_count_last_step
        self.num_steps = num_steps
        self.prune_incomplete_proofs = prune_incomplete_proofs
        self.rules = rules
        self.domains = domains
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

    def _init_internals(self, queries: List[Tuple]):
        # this tell us the queries for each relation to analyse.
        self.relation2queries = {}  # reset
        for q in queries:
            if q[0] not in self.relation2queries:
                self.relation2queries[q[0]] = set()
            self.relation2queries[q[0]].add(q)

        # the other are not reset as they are incrementally added.
        for rule in self.rules:
            if rule.name not in self.rule2groundings:
                self.rule2groundings[rule.name] = set()
            if rule.name not in self.rule2proofs:
                self.rule2proofs[rule.name] = []

    # Ground a batch of queries, the result is cached for speed.
    def ground(self,
               pred_counts,
               facts: List[Tuple],
               queries: List[Tuple],
               **kwargs) -> Dict[str, RuleGroundings]:

        if self.rules is None or len(self.rules) == 0:
            return []
        atoms_remove = {}
        groundings_per_level = {}
        self._init_internals(queries)
        self._rule2processed_queries = {rule.name: set() for rule in self.rules}
        for step in range(self.num_steps):
            # print('STEP NUMBER ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', step,'^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^','step ',step,'/', self.num_steps, 'known body',step == self.num_steps - 1, )
            for j,rule in enumerate(self.rules):
                # print('\nrule ', rule, ' """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" ')
                # Here we assume to have a Horn clause, fix it.
                queries_per_rule = list(
                    self.relation2queries.get(rule.head[0][0], set()))
                # print('\nqueries_per_rule\n',len(queries_per_rule), queries_per_rule)
                if not queries_per_rule:
                    continue
                backward_chaining_grounding_one_rule_with_domains(
                    self.rule2groundings,
                    pred_counts,
                    atoms_remove,
                    groundings_per_level,
                    self.domains, rule, queries_per_rule, self._fact_index,
                    # max_unknown_fact_count
                    (self.max_unknown_fact_count if step < self.num_steps - 1
                     else self.max_unknown_fact_count_last_step),
                    # Output added here.
                    res=self.rule2groundings[rule.name],
                    # Proofs added here.
                    proofs=(self.rule2proofs[rule.name]
                            if self.prune_incomplete_proofs else None),step=step, n_steps =self.num_steps)
                print('Total  groundings in res after rule',j,'/',len(self.rules),', step',step,sum([len(v) for k, v in self.rule2groundings.items()])) # IS IS MORE THAN THE GROUNDINGS_per_level BECAUSE THERE ARE DUPLICATES
                # Update the list of processed rules.
                self._rule2processed_queries[rule.name].update(queries_per_rule)
            if step == self.num_steps - 1:
                break
            # FROM THE ATOMS TO REMOVE, I SHOULD ONLY KEEP THE ONES THAT ARE PROVED, AND AT MOST 1 ATOM IS MISSING
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
            self._init_internals(list(new_queries))

        print('\nNum groundings',sum([len(v) for k, v in self.rule2groundings.items()]))
        if self.prune_incomplete_proofs:
            self.rule2groundings = PruneIncompleteProofs(self.rule2groundings,
                                                         self.rule2proofs,
                                                         self._fact_index,
                                                         self.num_steps)
            print('Num groundings after pruning',sum([len(v) for k, v in self.rule2groundings.items()]))
        if self.prune_incomplete_proofs:
            for level in range(self.num_steps):
                # if the level is in the keys of groundings_per_level, prune the groundings
                if level in groundings_per_level:
                    # print the keys of the groundings_per_level
                    print('\nNum groundings in level',level,',',len(groundings_per_level[level]))
                    groundings_per_level[level] = Prune_groundings_per_level(groundings_per_level[level],
                                                                self.rule2proofs,
                                                                self._fact_index,
                                                                self.num_steps)
                    print('Num groundings in level',level,', after pruning,',len(groundings_per_level[level]))

        if 'deterministic' in kwargs and kwargs['deterministic']:
            ret = {rule_name: RuleGroundings(
                rule_name, sorted(list(groundings), key=lambda x : x.__repr__()))
                   for rule_name,groundings in self.rule2groundings.items()}
        else:
            ret = {rule_name: RuleGroundings(rule_name, list(groundings))
                   for rule_name,groundings in self.rule2groundings.items()}
        return ret,atoms_remove,groundings_per_level


 



def get_arg(args, name: str, default=None, assert_defined=False):
    value = getattr(args, name) if hasattr(args, name) else default
    if assert_defined:
        assert value is not None, 'Arg %s is not defined: %s' % (name, str(args))
    return value

def read_rules(path,args):
    print('Reading rules')
    rules = []
    with open(path, 'r') as f:
        for line in f:
            # if len(rules) < 11:
            # split by :
            line = line.split(':')
            # first element is the name of the rule
            rule_name = line[0]
            # second element is the weight of the rule
            rule_weight = float(line[1].replace(',', '.'))
            # third element is the rule itself. Split by ->
            rule = line[2].split('->')
            # second element is the head of the rule
            rule_head = rule[1]
            # remove the \n from the head and the space
            rule_head = [rule_head[1:-1]]
            # first element is the body of the rule
            rule_body = rule[0]
            # split the body by ,
            rule_body = rule_body.split(', ')
            # for every body element, if the last character is a " ", remove it
            for i in range(len(rule_body)):
                if rule_body[i][-1] == " ":
                    rule_body[i] = rule_body[i][:-1]
            # Take the vars of the body and head and put them in a dictionary
            all_vars = rule_body + rule_head
            var_names = {}
            for i in range(len(all_vars)):
                # split the element of the body by (
                open_parenthesis = all_vars[i].split('(')
                # Split the second element by )
                variables = open_parenthesis[1].split(')')
                # divide the variables by ,
                variables = variables[0].split(',')
                # Create a dictionary with the variables as keys and the value "countries" as values
                if 'nations' in args.dataset_name:
                    for var in variables:
                        var_names[var] = "countries"
                elif ('countries' in args.dataset_name) or ('test_dataset' in args.dataset_name):
                        var_names = {"X": "countries", "W": "subregions", "Z": "regions", "Y": "countries", "K": "countries"}
                elif 'kinship' in args.dataset_name:
                    # var_names = {"x": "people", "y": "people", "z": "people","a": "people", "b": "people","c": "people","d": "people"}
                    for var in variables:
                        var_names[var] = "people"      
                elif 'pharmkg' in args.dataset_name:
                    # var_names = {"a": "cte", "b": "cte","c": "cte","d": "cte", "h": "cte", "g": "cte"}
                    for var in variables:
                        var_names[var] = "cte" 
            rules.append(Rule(name=rule_name,var2domain=var_names,body=rule_body,head=rule_head))
    print('number of rules: ', len(rules))
    return rules

def main(base_path, output_filename, kge_output_filename, log_filename, args):

    csv_logger = CSVLogger(log_filename, append=True, separator=';')
    print('\nARGS', args,'\n')

    seed = get_arg(args, 'seed_run_i', 0)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Params
    ragged = get_arg(args, 'ragged', None, True)
    valid_frequency = get_arg(args, 'valid_frequency', None, True)

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
    
    dataset_train = data_handler.get_dataset(
        split="train",
        number_negatives=args.num_negatives)

    dataset_test = data_handler.get_dataset(split="test", corrupt_mode='TAIL',number_negatives=args.test_negatives)
    
    fol = data_handler.fol
    domain2adaptive_constants: Dict[str, List[str]] = None
    num_adaptive_constants = get_arg(args, 'engine_num_adaptive_constants', 0)

    enable_rules = (args.reasoner_depth > 0 and args.num_rules > 0)
    if enable_rules: 
        rules = read_rules(join(base_path, args.dataset_name, args.rules_file),args)
        # For KGEs with no domains.
        # domains = {Rule.default_domain(): fol.domains[0]}

        domain2adaptive_constants = {
            d.name : ['__adaptive_%s_%d' % (d.name, i)
                    for i in range(num_adaptive_constants)]
            for d in fol.domains
            }
        
        if 'backward' in args.grounder:
            num_steps = int(args.grounder.split('_')[-1])
            prune_backward = True if ( ('backward' in args.grounder) and ('prune'in args.grounder) ) else False
            print('Using backward chaining with %d steps' % num_steps)
            engine = BackwardChainingGrounder(rules, facts=list(data_handler.train_known_facts_set),
                                                        domains={d.name:d for d in fol.domains},
                                                        num_steps=num_steps,prune_incomplete_proofs=prune_backward)

    else:
        rules = []
        engine = None

    serializer = ns.serializer.LogicSerializerFast(
        predicates=fol.predicates, domains=fol.domains,
        constant2domain_name=fol.constant2domain_name,
        domain2adaptive_constants=domain2adaptive_constants)

  
    queries, labels = dataset_test[0:len(dataset_test)]
    facts = fol.facts
    rules = engine.rules
    # for every rule, get the predicate in the head. Count the number of times each predicate appears
    pred_head_facts_counts = {}
    for rule in rules:
        pred = rule.head[0][0]
        if pred not in pred_head_facts_counts:
            pred_head_facts_counts[pred] = 0
        pred_head_facts_counts[pred] += 1
    # print the number of times each predicate appears
    print('pred_head_facts_counts', pred_head_facts_counts)
    pred_counts = sorted(pred_head_facts_counts, key=pred_head_facts_counts.get, reverse=True)
    print('PRED COUNTS ORDERED', pred_counts) 

    groundings,atoms_remove_per_level,groundings_per_level= engine.ground(pred_head_facts_counts,tuple(facts),tuple(ns.utils.to_flat(queries)),deterministic=True)

    print('\n')
    if num_steps == 2:
        print('take into account atoms from level 0 ') 
        remove_levels =[0]
    elif num_steps == 3:
        print('take into account atoms from level 0 and 1') 
        remove_levels =[0,1]
    else:  
        print('ERROR: num_steps should be 2 or 3 minimum')
    print()

    def find_atom_in_groundings(atom, groundings): 
        # this, in case of pruning, checks if the atom is still present in the groundings
        # Only on the body
        for grounding in groundings:
            for body_atom in grounding[1]:
                if body_atom == atom:
                    return True
        return False

    # Take only the levels we are interested in and append the atoms into final atoms. Filter the atoms that are not in the proved groundings.  
    atoms_remove = set()
    print ('ATOMS TO REMOVE:')
    for k,v in atoms_remove_per_level.items():
        if k in remove_levels:
            print('level',k,', number of atoms to remove',len(v))
            count = 0
            for atom in set(v):
                # check if the atom is in the pruned groundings of that level, because otherwise that body atom is not proved, and we have to remove the atom 
                atom_found = find_atom_in_groundings(atom, groundings_per_level[k])
                print('  Atom', atom,  '     in body of groundings, level',k,'       ',atom_found)
                if atom_found:
                    count += 1
                    atoms_remove.add(atom)
            # print('level',k,', number of atoms to remove taking into account prunning:',count)
            print()

    # THE MEASURE TO DO THE CUTOFF IS, FOR EACH PREDICATE, THE % OF ATOMS OUT OF THE NUMBER OF ATOMS IN FACTS WITH THAT PREDICATE
    # I COULD ALSO TAKE INTO ACCOUNT HOW MANY TIMES THEY ARE IN THE BODY OF A GROUNDING IN THE LAST LEVEL, i.e. if I do backward 3, level 3
    # Ex: If in a grounding at level 2 I remove  LocIn(Italy,Europe), then it is a problem if that is later in another grounding at level 3, because if it is not in the body at level 3, it is not proved
        
    def find_cutoff(atoms_remove,data_handler):
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
            max_atoms_allowed[pred] = int(0.2*pred_counts_facts[pred])
            pred_cutoff[pred] = pred_counts_to_remove[pred] / pred_counts_facts[pred]
            print('     ',pred, 'percentage of atoms with that predicate in facts: ', np.round(pred_cutoff[pred],3),'. ', pred_counts_to_remove[pred], '/', pred_counts_facts[pred])
        return max_atoms_allowed,pred_cutoff,pred_counts_to_remove,pred_counts_facts

    max_atoms_allowed,pred_cutoff,pred_counts_to_remove,pred_counts_facts = find_cutoff(atoms_remove,data_handler)

    # from atoms_remove, check how many atoms are in the body of the grounding of the last level 
    atoms_problem = set()
    if num_steps-1 in groundings_per_level and len(groundings_per_level[num_steps-1])!=0 :
        n_groundings_last_level = len(groundings_per_level[num_steps-1])  
        print('n_groundings_last_level',n_groundings_last_level)
        groundings_last_level = groundings_per_level[num_steps-1]
        for grounding in groundings_last_level:
            for body_atom in grounding[1]:
                if body_atom in atoms_remove:
                    print('atom', atom, 'is in the body of a grounding in the last level. Added to problematic atoms')
                    atoms_problem.add(atom)
    else:
        print('No groundings in the last level')


    # after that, I should check how many groundings for all the levels except the last one, have the atoms to remove in the body
    # If it is more than 2, we are not going to the next level because we are not admitting the subsitution
    # Therefore, for every level, for every final_atom, check how many groundings have that more than one final_atom in their body  
    
    for level,groundings_level_i in groundings_per_level.items():
        if level != num_steps-1:
            # print('level', level)
            for grounding in groundings_level_i:
                cuenta = 0 
                atoms_problematic = set()
                # print('body', grounding[1])
                for body_atom in grounding[1]:
                    # print('     body_atom', body_atom)
                    for atom in atoms_remove:
                        # print('             atom', atom)
                        if atom == body_atom:
                            # print('atom', atom, 'is in the body ', body_atom,'------------------------------------------------------------------------')
                            cuenta += 1
                            atoms_problematic.add(atom)
                if cuenta > 1:
                    atoms_problem.add(atoms_problematic)
                    print('grounding has more than 1 atom to removed in the body because of', atoms_problematic,'. Added to problematic atoms')




    print('Number of atoms to remove:', len(atoms_remove))
    for predicate,cutoff in max_atoms_allowed.items():
        # first remove the atoms that are not problematic
        for atom in atoms_remove:
            if atom[0] == predicate:
                if len(atoms_remove) > cutoff:
                    if atom not in atoms_problem:
                        atoms_remove.remove(atom)
    print('Number of atoms to remove after selected cutoff:', len(atoms_remove)) 

    for predicate,cutoff in max_atoms_allowed.items():
        # if there are still atoms to remove, do it randomly
        if len(atoms_remove) > cutoff:
            if atom[0] == predicate:
                atoms_remove = random.sample(list(atoms_remove), cutoff)
    print('Number of atoms to remove after random cutoff:', len(atoms_remove)) 


    # Last thing to do. Write a new (train,facts) file without the final atoms removed
    # Write the facts file if the number of facts and atoms to remove is bigger than 0
    # if  
    facts_file = join(base_path, args.dataset_name,'facts_reason_'+str(num_steps)+'.txt')
    train_file = join(base_path, args.dataset_name,'train_reason_'+str(num_steps)+'.txt')

    if len(atoms_remove) > 0:
        print('Writing the train file', train_file)
        with open(train_file, 'w') as f:
            for train_fact in data_handler.train_facts:
                if train_fact not in atoms_remove:
                    f.write('('+str(train_fact[0])+','+str(train_fact[1])+','+str(train_fact[2])+').\n')
        if len(facts) > 0:
            print('Writing the facts file', facts_file)
            with open(facts_file, 'w') as f:
                for fact in data_handler.known_facts:
                    if fact not in atoms_remove:
                        f.write('('+str(fact[0])+','+str(fact[1])+','+str(fact[2])+').\n')

    # Print all the info in a .txt file
    output_file = join(base_path, args.dataset_name,'info_'+str(num_steps)+'.txt')
    with open(output_file, 'w') as f:
        f.write('pred_head_facts_counts: {}\n'.format(pred_head_facts_counts))
        for pred in pred_counts_to_remove:
            f.write('     ',pred, 'percentage of atoms with that predicate in facts: ', np.round(pred_cutoff[pred],3),'. ', pred_counts_to_remove[pred], '/', pred_counts_facts[pred])
        f.write('Number of atoms to remove: {}\n'.format(len(atoms_remove)))
        f.write('Number of atoms to remove after selected cutoff: {}\n'.format(len(atoms_remove)))
        f.write('Number of atoms to remove after random cutoff: {}\n'.format(len(atoms_remove)))
        f.write('problematic atoms:',str(atoms_problematic))
