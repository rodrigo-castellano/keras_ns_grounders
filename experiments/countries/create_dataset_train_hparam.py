import copy
import time
from typing import List, Set, Tuple, Dict, Union
from keras_ns.logic.commons import Atom, Domain, Rule, RuleGroundings
from keras_ns.grounding.engine import Engine
from itertools import chain, product


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from typing import Dict, List
from keras_ns.logic.commons import Atom, FOL, RuleGroundings
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


class AtomIndex():
    def __init__(self, facts: List[Tuple[str, str, str]]):
        _index = {}
        for f in facts:
            r_idx, s_idx, o_idx = f
            key = (r_idx, None, None)
            if key not in _index:
                _index[key] = set()
            _index[key].add(f)
            key = (r_idx, s_idx, None)
            if key not in _index:
                _index[key] = set()
            _index[key].add(f)
            key = (r_idx,  None, o_idx)
            if key not in _index:
                _index[key] = set()
            _index[key].add(f)
            key = f
            _index[key] = set([f])

        # Store tuple instead of sets.
        self._index = {k: tuple(v) for k, v in _index.items()}

    def get_matching_atoms(self,
                           atom: Tuple[str, str, str]) -> Tuple[Tuple[str, str, str]]:
        return self._index.get(atom, [])

def get_atoms(r2g:Dict[str, Set[Tuple[Tuple, Tuple]]]) -> Set[Tuple]:
    atoms = set()
    for r, g in r2g.items():
        for rule_atoms in g:
            for atom in rule_atoms[0]:
                atoms.add(atom)
            for atom in rule_atoms[1]:
                atoms.add(atom)
    return atoms

def get_atoms_on_groundings(groundings:Set[Tuple[Tuple, Tuple]]) -> Set[Tuple]:
    atoms = set()
    for rule_atoms in groundings:
        for atom in rule_atoms[0]:  # head
            atoms.add(atom)
        for atom in rule_atoms[1]:  # tail
            atoms.add(atom)
    return atoms

# res is a Set of (Tuple_head_groundings, Tuple_body_groundings)
def backward_chaining_grounding_one_rule(
        rule: Rule, queries: List[Tuple],
        fact_index: AtomIndex,
        known_body_only: bool=False,
        res: Set[Tuple[Tuple, Tuple]]=None) -> Union[None,
                                                     Set[Tuple[Tuple, Tuple]]]:
    # We have a rule like A(x,y) B(y,z) => C(x,z)
    assert len(rule.head) == 1, (
        'Rule is not a Horn clause %s' % str(rule))
    head = rule.head[0]

    assert len(rule.body) <= 2, (
        'Horn clauses with more than 2 elements in the body are '
        'not supported yet')

    new_ground_atoms = set()

    for q in queries:
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
        if all(ground_body_atom[1:]):
            groundings = (ground_body_atom,)
        else:
            # Tuple of atoms matching A(Antonio,None) in the facts.
            # This is the list of ground atoms for the i-th atom in the body.
            # groundings = fact_index.get_matching_atoms(ground_body_atom)
            groundings = fact_index._index.get(ground_body_atom, [])

        if len(rule.body) == 1:
            # Shortcut, we are done, the clause has no free variables.
            # Return the groundings.
            new_ground_atoms.add(((q,), groundings))
            continue

        # Select the other atom in the body and ground it with the
        # assignments with the head and the other body ground atom fixed.
        body_atom2 = rule.body[(i + 1) % 2]
        for ground_atom in groundings:
            head_body_ground_vars = copy.copy(head_ground_vars)
            head_body_ground_vars.update(
                {v: a for v, a in zip(body_atom[1:], ground_atom[1:])})
            new_ground_atom = (body_atom2[0], ) + tuple(
                [head_body_ground_vars.get(body_atom2[j+1], None)
                 for j in range(len(body_atom2)-1)])

            if all(new_ground_atom) and (
                not known_body_only or
                fact_index._index.get(new_ground_atom, [])):
                body_grounding = (ground_atom if i == 0 else new_ground_atom,
                                  new_ground_atom if i == 0 else ground_atom)
                new_ground_atoms.add(((q,), body_grounding))

    if res is None:
        return new_ground_atoms
    else:
        res.update(new_ground_atoms)

# res is a Set of (Tuple_head_groundings, Tuple_body_groundings)
def backward_chaining_grounding_one_rule_with_domains(
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
    proofs: Dict[Tuple[Tuple, Tuple], List[Tuple[Tuple, Tuple]]]=None,
    atoms_to_remove=set(), step=-1, n_steps=-1) -> Union[
        None, Set[Tuple[Tuple, Tuple]]]:
    
    start = time.time()
    # We have a rule like A(x,y) B(y,z) => C(x,z)
    assert len(rule.head) == 1, (
        'Rule is not a Horn clause %s' % str(rule))
    head = rule.head[0]
    build_proofs: bool = (proofs is not None)

    new_ground_atoms = set()
    cont = 0
    lim=10
    for q in queries:
      cont += 1 
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
            # print('     -GROUND ATOM', ground_atom)
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
            # Iterate over the groundings of the free vars.
            # If no free vars are available, product returns a single empty
            # tuple, meaning that we still correctly enter in the following
            # for loop for a single round.
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
                    # print('     ADDED', q, '->', tuple(body_grounding)) if cont< lim else None
                    # print('ADDED', q, '->', tuple(body_grounding), 'TO_PROVE',          str(body_grounding_to_prove) if build_proofs else '')
                    new_ground_atoms.add(((q,), tuple(body_grounding)))
                    if build_proofs:
                        proofs.append((q, body_grounding_to_prove))
    #   print('updated new_ground_atoms', new_ground_atoms) if cont< lim else None

    end = time.time()
    # print('NUM_GROUNDINGS', len(new_ground_atoms), 'TIME', end - start)
    # print('NEW GROUND ATOMS', new_ground_atoms) if cont< lim else None
    print('step number',step,'/',n_steps)
    if step != (n_steps-1) and step!= 0 :
        print('step',step,', adding new_ground_atoms')
        # update atoms_to_remove with new_ground_atoms
        atoms_to_remove.update(new_ground_atoms) 
        print('atoms_to_remove',atoms_to_remove)
    if res is None:
        return new_ground_atoms,atoms_to_remove
    else:
        res.update(new_ground_atoms),atoms_to_remove


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
               facts: List[Tuple],
               queries: List[Tuple],
               **kwargs) -> Dict[str, RuleGroundings]:
        atoms_to_remove = set()
        if self.rules is None or len(self.rules) == 0:
            return []
        # To debug: order the queries by the head, and then by body.if the len of the queries is less than 100
        queries = sorted(queries, key=lambda x: (x[0], x[1:])) if len(queries) < 50 else queries
        # print('\nQUERIES\n', queries, '\n')
        self._init_internals(queries)
        # order also the relation2queries
        for k,v in self.relation2queries.items():
            self.relation2queries[k] = sorted(list(v), key=lambda x: (x[0], x[1:])) if len(v) < 50 else v
        # print('\nAtoms to process per query. self.relation2queries\n',self.relation2queries)
        # Keeps track of the queris already processed for this rule.
        self._rule2processed_queries = {rule.name: set() for rule in self.rules}
        for step in range(self.num_steps):
            print('STEP NUMBER ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', step,'^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^','step ',step,'/', self.num_steps, 'known body',step == self.num_steps - 1, )
            for rule in self.rules:
                print('\nrule ', rule, ' """"""""""""""""""""""""""""""""""""""""" """""""""""""""""""""""""" ')
                # Here we assume to have a Horn clause, fix it.
                queries_per_rule = list(
                    self.relation2queries.get(rule.head[0][0], set()))
                # print('\nqueries_per_rule\n',len(queries_per_rule), queries_per_rule)
                if not queries_per_rule:
                    continue
                backward_chaining_grounding_one_rule_with_domains(
                    self.domains, rule, queries_per_rule, self._fact_index,
                    # max_unknown_fact_count
                    (self.max_unknown_fact_count if step < self.num_steps - 1
                     else self.max_unknown_fact_count_last_step),
                    # Output added here.
                    res=self.rule2groundings[rule.name],
                    # Proofs added here.
                    proofs=(self.rule2proofs[rule.name]
                            if self.prune_incomplete_proofs else None),
                    atoms_to_remove = atoms_to_remove, step=step, n_steps =self.num_steps)
                # Update the list of processed rules.
                self._rule2processed_queries[rule.name].update(queries_per_rule)
                # print('\nqueries processed:, _rule2processed_queries\n', len(self._rule2processed_queries[rule.name]),self._rule2processed_queries[rule.name])
                # print()
                # for k,v in self.rule2groundings.items():
                #     print('rule2groundings', k, len(v),v)

            if step == self.num_steps - 1:
                break
            
            # FROM THE ATOMS TO REMOVE, I SHOULD ONLY KEEP THE ONES THAT ARE PROVED, AND AT MOST 1 ATOM IS MISSING
            # I SHOULD DO THE PROCESS FOR THE TEST SET

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

        # print('\nFinal groundings\n')
        # for k,v in self.rule2groundings.items():
            # print('rule2groundings', k, len(v),v)
        if 'deterministic' in kwargs and kwargs['deterministic']:
            ret = {rule_name: RuleGroundings(
                rule_name, sorted(list(groundings), key=lambda x : x.__repr__()))
                   for rule_name,groundings in self.rule2groundings.items()}
        else:
            ret = {rule_name: RuleGroundings(rule_name, list(groundings))
                   for rule_name,groundings in self.rule2groundings.items()}

        return ret











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
            # print all the info
            # if len(rules) < 1001:
            #     print('rule name: ', rule_name, 'rule weight: ', rule_weight, 'rule head: ', rule_head, 
            #         'rule body: ', rule_body, 'var_names: ', var_names)
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
    print('ENABLE RULES',enable_rules, 'REASONER DEPTH', args.reasoner_depth, 'NUM RULES', args.num_rules, 'GROUNDER', args.grounder)
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
            num_steps = int(args.grounder.split('_')[1])
            print('Using backward chaining with %d steps' % num_steps)
            print('CHOOSING BACKWARD AS GROUNDER')
            engine = BackwardChainingGrounder(rules, facts=list(data_handler.train_known_facts_set),
                                                        domains={d.name:d for d in fol.domains},
                                                        num_steps=num_steps)

    else:
        rules = []
        engine = None

    serializer = ns.serializer.LogicSerializerFast(
        predicates=fol.predicates, domains=fol.domains,
        constant2domain_name=fol.constant2domain_name,
        domain2adaptive_constants=domain2adaptive_constants)

  
    print('starting the gruonding')

    queries, labels = dataset_train[0:len(dataset_train)]
    facts = fol.facts
    ground_formulas = engine.ground(tuple(facts),tuple(ns.utils.to_flat(queries)),deterministic=True)
    rules = engine.rules

    # print('Generating train data')
    # start = time.time()
    # data_gen_train = ns.dataset.DataGenerator(
    #     dataset_train, fol, serializer, engine,
    #     batch_size=args.batch_size, ragged=ragged)
    # end = time.time()
    # print("Time to create data generator train: ", np.round(end - start,2))
 

    # start = time.time()
    # data_gen_test = ns.dataset.DataGenerator(
    #     dataset_test, fol, serializer, engine,
    #     batch_size=args.eval_batch_size, ragged=ragged)
    # end = time.time()
    # print("Time to create data generator test: ",  np.round(end - start,2))