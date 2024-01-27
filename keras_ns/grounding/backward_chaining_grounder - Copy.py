#! /bin/python3
import copy
import time
from typing import List, Set, Tuple, Dict, Union
from keras_ns.logic.commons import Atom, Domain, Rule, RuleGroundings
from keras_ns.grounding.engine import Engine
from itertools import product

kNThreads = 1


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
    rule: Rule, queries: List[Tuple],
    fact_index: AtomIndex,
    known_body_only: bool=False,
    res: Set[Tuple[Tuple, Tuple]]=None) -> Union[None,
                                                 Set[Tuple[Tuple, Tuple]]]:
    start = time.time()
    # We have a rule like A(x,y) B(y,z) => C(x,z)
    assert len(rule.head) == 1, (
        'Rule is not a Horn clause %s' % str(rule))
    head = rule.head[0]

    new_ground_atoms = set()
 
    cont = 0
    lim=300000
    for q in queries:
      cont += 1 
      print('\n\n***************q', q,'********************') if cont< lim else None
      if q[0] != head[0]:  # predicates must match.
        continue

      # Get the variable assignments from the head.
      head_ground_vars = {v: a for v, a in zip(head[1:], q[1:])}
      print('for every atom i in the body') if cont< lim else None
      for i in range(len(rule.body)):
        # Ground atom by replacing variables with constants.
        # The result is the partially ground atom A('Antonio',None)
        # with None indicating unground variables.
        body_atom = rule.body[i]
        ground_body_atom = (body_atom[0], ) + tuple(
            [head_ground_vars.get(body_atom[j+1], None)
             for j in range(len(body_atom)-1)])
        print('\n- i', i,'. ground_body_atom:', ground_body_atom, '. Substitution (by None) of the vars not present in head.') if cont< lim else None
        if all(ground_body_atom[1:]):
            groundings = (ground_body_atom,)
            print('groundings already done, #all vars are subtituted', groundings) if cont< lim else None
        else:
            # Tuple of atoms matching A(Antonio,None) in the facts.
            # This is the list of ground atoms for the i-th atom in the body.
            # groundings = fact_index.get_matching_atoms(ground_body_atom)
            groundings = fact_index._index.get(ground_body_atom, [])
            print('groundings found in facts', groundings) if cont< lim else None

        if len(rule.body) == 1:
            print('length one in the body, one predicate') if cont< lim else None
            # Shortcut, we are done, the clause has no free variables.
            # Return the groundings.
            if len(groundings) != 0:
                print('there are not more body atoms to ground') if cont< lim else None
                for grounding_ in groundings:
                    print('ADDED', q, '->', (grounding_,)) if cont< lim else None
                    new_ground_atoms.add(((q,), (grounding_,)))
            continue
        print('\nfor every grounding of the body atom') if cont< lim else None
        for ground_atom in groundings:
            print('--grounded_atom', ground_atom, ' The other vars (not present in head) are left as free') if cont< lim else None
            head_body_ground_vars = copy.copy(head_ground_vars)
            head_body_ground_vars.update(
                {v: a for v, a in zip(body_atom[1:], ground_atom[1:])})
    
            free_var2domain = [(v,d) for v,d in rule.vars2domain.items()
                               if v not in head_body_ground_vars]
            free_vars = [vd[0] for vd in free_var2domain]
            for ground_vars in product(
                *[domains[vd[1]].constants for vd in free_var2domain]):
                var2ground = dict(zip(free_vars, ground_vars))
                full_ground_vars = {**head_body_ground_vars,
                                    **var2ground}
                print('for every possible grounding of the free vars',ground_vars) if cont< lim else None

                accepted = True
                body_grounding = []
                for j in range(len(rule.body)):
                    if i == j:
                        if known_body_only and not (fact_index._index.get(ground_atom, [])):
                            print('     ',ground_atom, 'not in facts, and known body is true ')
                            accepted = False
                            break
                        body_grounding.append(ground_atom)
                        print('     ---j', j,', body_grounding,i=j', ground_atom, '. Not correct that (known body and not in facts). Add the ground atom BY DEFINITION IT IS ALWAYS IN FACTS') if cont< lim else None
                        continue
                    body_atom2 = rule.body[j]
                    new_ground_atom = (body_atom2[0], ) + tuple(
                        [full_ground_vars.get(body_atom2[k+1], None)
                         for k in range(len(body_atom2)-1)])
                    # print('     condition: (fact_index._index.get(new_ground_atom, []))', (fact_index._index.get(new_ground_atom, [])) ) if cont< lim else None
                    # print('     condition ', (all(new_ground_atom) and (fact_index._index.get(new_ground_atom, [])))  ) if cont< lim else None
                    if all(new_ground_atom) and (not known_body_only or fact_index._index.get(new_ground_atom, [])):
                        body_grounding.append(new_ground_atom)
                        print('     ---j', j,', body_grounding,i=!j, accepted ', new_ground_atom,'. No free vars:',all(new_ground_atom),', is in index:',fact_index._index.get(new_ground_atom, []) ) if cont< lim else None
                    else:
                        accepted = False
                        print('     ---j', j,', body_grounding,i=!j, not accepted', new_ground_atom,'. No free vars:',all(new_ground_atom),', is in index:',fact_index._index.get(new_ground_atom, []) ) if cont< lim else None
                        break
                if accepted:
                    print('ADDED', q, '->', tuple(body_grounding)) if cont< lim else None
                    new_ground_atoms.add(((q,), tuple(body_grounding)))
      print('----------------new body_grounding', new_ground_atoms) if cont< lim else None

    end = time.time()
    # print('NUM_GROUNDINGS', len(new_ground_atoms), end - start)
    if res is None:
        return new_ground_atoms
    else:
        res.update(new_ground_atoms)


def new_backward_chaining_grounding_one_rule_with_domains(
    domains: Dict[str, Domain],
    rule: Rule, queries: List[Tuple],
    fact_index: AtomIndex,
    known_body_only: bool=False,
    res: Set[Tuple[Tuple, Tuple]]=None,
    queries_to_prove: Dict[int, Tuple[Tuple, Tuple, Tuple]]=None,
    level =  None ) -> Union[None, Set[Tuple[Tuple, Tuple]]]:
    # initialize Queries_to_prove at the current level
    queries_to_prove[level] = set()
    start = time.time()
    # We have a rule like A(x,y) B(y,z) => C(x,z)
    assert len(rule.head) == 1, (
        'Rule is not a Horn clause %s' % str(rule))
    head = rule.head[0]

    new_ground_atoms = set()
 
    cont = 0
    lim=300000
    for q in queries:
      cont += 1 
      print('\n\n***************q', q,'********************') if cont< lim else None
      if q[0] != head[0]:  # predicates must match.
        continue

      # Get the variable assignments from the head.
      head_ground_vars = {v: a for v, a in zip(head[1:], q[1:])}
      print('for every atom i in the body') if cont< lim else None
      for i in range(len(rule.body)):
        # Ground atom by replacing variables with constants.
        # The result is the partially ground atom A('Antonio',None)
        # with None indicating unground variables.
        body_atom = rule.body[i]
        ground_body_atom = (body_atom[0], ) + tuple(
            [head_ground_vars.get(body_atom[j+1], None)
             for j in range(len(body_atom)-1)])
        print('\n- i', i,'. ground_body_atom:', ground_body_atom, '. Substitution (by None) of the vars not present in head.') if cont< lim else None
        if all(ground_body_atom[1:]):
            groundings = (ground_body_atom,)
            print('groundings already done, #all vars are subtituted', groundings) if cont< lim else None
        else:
            # Tuple of atoms matching A(Antonio,None) in the facts.
            # This is the list of ground atoms for the i-th atom in the body.
            # groundings = fact_index.get_matching_atoms(ground_body_atom)
            groundings = fact_index._index.get(ground_body_atom, [])
            print('groundings found in facts', groundings) if cont< lim else None

        if len(rule.body) == 1:
            print('length one in the body, one predicate') if cont< lim else None
            # Shortcut, we are done, the clause has no free variables.
            # Return the groundings.
            if len(groundings) != 0:
                print('there are not more body atoms to ground') if cont< lim else None
                for grounding_ in groundings:
                    print('ADDED', q, '->', (grounding_,)) if cont< lim else None
                    new_ground_atoms.add(((q,), (grounding_,)))
            continue
        print('\nfor every grounding of the body atom') if cont< lim else None
        for ground_atom in groundings:
            print('--grounded_atom', ground_atom, ' The other vars (not present in head) are left as free') if cont< lim else None
            head_body_ground_vars = copy.copy(head_ground_vars)
            head_body_ground_vars.update(
                {v: a for v, a in zip(body_atom[1:], ground_atom[1:])})
    
            free_var2domain = [(v,d) for v,d in rule.vars2domain.items()
                               if v not in head_body_ground_vars]
            free_vars = [vd[0] for vd in free_var2domain]
            for ground_vars in product(
                *[domains[vd[1]].constants for vd in free_var2domain]):
                var2ground = dict(zip(free_vars, ground_vars))
                full_ground_vars = {**head_body_ground_vars,
                                    **var2ground}
                print('for every possible grounding of the free vars',ground_vars) if cont< lim else None

                # accepted = True
                # create body_grounding as a dict : (tuple:Bool)
                body_grounding =  {}
                for j in range(len(rule.body)):
                    if i == j:
                        if known_body_only and not (fact_index._index.get(ground_atom, [])):
                            print('     ',ground_atom, 'not in facts, and known body is true ')
                            # accepted = False
                            break
                        # body_grounding.append(ground_atom)
                        if fact_index._index.get(ground_atom, []):
                            body_grounding[ground_atom] = True
                        else: 
                            body_grounding[ground_atom] = False

                        print('     ---j', j,', body_grounding,i=j', ground_atom, '. Not correct that (known body and not in facts). Add the ground atom BY DEFINITION IT IS ALWAYS IN FACTS') if cont< lim else None
                        continue
                    body_atom2 = rule.body[j]
                    new_ground_atom = (body_atom2[0], ) + tuple(
                        [full_ground_vars.get(body_atom2[k+1], None)
                         for k in range(len(body_atom2)-1)])
                    # print('     condition: (fact_index._index.get(new_ground_atom, []))', (fact_index._index.get(new_ground_atom, [])) ) if cont< lim else None
                    # print('     condition ', (all(new_ground_atom) and (fact_index._index.get(new_ground_atom, [])))  ) if cont< lim else None
                    # if all(new_ground_atom) and (not known_body_only or fact_index._index.get(new_ground_atom, [])):
                    #     body_grounding.append(new_ground_atom)
                    #     print('     ---j', j,', body_grounding,i=!j, accepted ', new_ground_atom,'. No free vars:',all(new_ground_atom),', is in index:',fact_index._index.get(new_ground_atom, []) ) if cont< lim else None
                    # else:
                    #     accepted = False
                    #     print('     ---j', j,', body_grounding,i=!j, not accepted', new_ground_atom,'. No free vars:',all(new_ground_atom),', is in index:',fact_index._index.get(new_ground_atom, []) ) if cont< lim else None
                    #     break
                    if all(new_ground_atom):
                        if known_body_only and not (fact_index._index.get(ground_atom, [])):
                            print('     ',ground_atom, 'not in facts, and known body is true ')
                            # accepted = False
                            break
                        if not known_body_only and (fact_index._index.get(ground_atom, [])):
                            body_grounding[ground_atom] = True
                        if not known_body_only and not (fact_index._index.get(ground_atom, [])):
                            body_grounding[ground_atom] = False 
                    else: 
                        # accepted = False
                        print('     ---j', j,', body_grounding,i=!j, not accepted', new_ground_atom,'. No free vars:',all(new_ground_atom),', is in index:',fact_index._index.get(new_ground_atom, []) ) if cont< lim else None
                        break
                # if accepted:
                #     print('ADDED', q, '->', tuple(body_grounding)) if cont< lim else None
                #     new_ground_atoms.add(((q,), tuple(body_grounding)))
                
                # go through all the body_grounding and check if all atoms are in facts
                if all(body_grounding.values()):
                    print('ADDED', q, '->', tuple(body_grounding)) if cont< lim else None
                    new_ground_atoms.add(((q,), tuple(body_grounding)))
                
      print('----------------new body_grounding', new_ground_atoms) if cont< lim else None

    end = time.time()
    # print('NUM_GROUNDINGS', len(new_ground_atoms), end - start)
    if res is None:
        return new_ground_atoms
    else:
        res.update(new_ground_atoms)


class BackwardChainingGrounder(Engine):

    def __init__(self, rules: List[Rule], facts: List[Union[Atom, str, Tuple]],
                 domains: Dict[str, Domain]=None,
                 num_steps: int=1,
                 # Unused.
                 n_threads: int=kNThreads):
        self.num_steps = num_steps
        self.rules = rules
        self.domains = domains
        self.facts = [a if isinstance(a,Tuple) else a.toTuple() if isinstance(a,Atom) else Atom(s=a).toTuple() for a in facts]
        # self.facts = facts
        for rule in self.rules:
            assert len(rule.head) == 1, (
                '%s is not a Horn clause' % str(rule))
        self._fact_index = AtomIndex(self.facts)


    def _init_internals(self, queries: List[Tuple]):
        self.relation2queries = {}
        for q in queries:
            if q[0] not in self.relation2queries:
                self.relation2queries[q[0]] = []
            self.relation2queries[q[0]].append(q)

        self.rule2groundings = {}
        for rule in self.rules:
            if rule.name not in self.rule2groundings:
                self.rule2groundings[rule.name] = set()

    # Ground a batch of queries, the result is cached for speed.
    def ground(self,
               facts: List[Tuple],
               queries: List[Tuple],
               **kwargs) -> Dict[str, RuleGroundings]:
        if self.rules is None or len(self.rules) == 0:
            return []
        print('queries', len(queries), queries)
        self._init_internals(queries)
        # create a dict with  one key for each level 
        queries_to_prove = {}
        for step in range(self.num_steps):
            print('STEP NUMBER ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', step,'^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^','step ',step,'/', self.num_steps, 'known body',step == self.num_steps - 1, )
            if step == self.num_steps - 1:
                known_body_only = True
            else:
                known_body_only = False
            for rule in self.rules:
                print('\nrule ', rule, ' """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" ')
                queries = self.relation2queries.get(rule.head[0][0], [])
                if not queries:
                    continue
                if self.domains is not None:
                    new_backward_chaining_grounding_one_rule_with_domains(
                        self.domains, rule, queries, self._fact_index,
                        known_body_only,
                        # Output added here.
                        self.rule2groundings[rule.name],
                        queries_to_prove,
                        level=step) 
                    # in self.rule2groundings print every item and its values in a different row
                else:
                    backward_chaining_grounding_one_rule(
                        rule, queries, self._fact_index,
                        known_body_only,
                        # Output added here.
                        self.rule2groundings[rule.name])

                if self.num_steps > 1:  # expand the queries and iterate.
                    # removes all the repeated queries (it doesnt apply for the first step, I should apply it as well)
                    queries = list(set(queries).union(
                        get_atoms(self.rule2groundings)))
                    self._init_internals(queries)


        if 'deterministic' in kwargs and kwargs['deterministic']:
            ret = {rule_name:
                   RuleGroundings(rule_name,
                                  sorted(list(groundings), key=lambda x : x.__repr__()))
                   for rule_name,groundings in self.rule2groundings.items()}
        else:
            ret = {rule_name:
                   RuleGroundings(rule_name, list(groundings))
                   for rule_name,groundings in self.rule2groundings.items()} 

        return ret
