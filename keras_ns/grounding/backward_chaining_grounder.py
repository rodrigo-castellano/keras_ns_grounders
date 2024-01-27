#! /bin/python3
import copy
import time
from typing import List, Set, Tuple, Dict, Union
from keras_ns.logic.commons import Atom, Domain, Rule, RuleGroundings
from keras_ns.grounding.engine import Engine
from itertools import chain, product

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
    # -1 or body_size implements pure unrestricted backward_chaining
    max_unknown_fact_count: int,
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
      #print('head_ground_vars', head_ground_vars) if cont< lim else None  

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
            print('groundings already done, #all vars are subtituted', groundings) if cont< lim else None
            groundings = (ground_body_atom,)
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
            new_ground_atoms.add(((q,), groundings))
            print('ADDED', q, '->', (groundings,)) if cont< lim else None
            continue
        # print('\nfor every grounding of the body atom') if cont< lim else None
        for ground_atom in groundings:
            # print('--grounded_atom', ground_atom, ' The other vars (not present in head) are left as free') if cont< lim else None
            # This loop is only needed to ground at least one atom in the body of the formula.
            # Otherwise it would be enough to start with the loop for ground_vars in product(...)
            # but it would often expand into many nodes.
            # The current solution does not allow to fully recover a classical backward_chaining, though.
            head_body_ground_vars = copy.copy(head_ground_vars)
            head_body_ground_vars.update(
                {v: a for v, a in zip(body_atom[1:], ground_atom[1:])})

            free_var2domain = [(v,d) for v,d in rule.vars2domain.items()
                               if v not in head_body_ground_vars]
            free_vars = [vd[0] for vd in free_var2domain] 
            # Iterate over the groundings of the free vars.
            # If no free vars are available, itertools.product returns a single empty tuple, meaning
            # that we still correctly enter in the following for loop for a single round.
            ################## A SHORCUT: if the body atom is already in the facts, we can skip the rest of the loop.
            ################### if len(rule.body) == max_unknown_fact_count +1, and free vars is empty, then add it to new groundings
            ################### if len(rule.body) == 2, and max unknown fact count is 1, AND THE LEFT VARS ARE FILLED, then acceptadd it to new groundings
            for ground_vars in product(
                *[domains[vd[1]].constants for vd in free_var2domain]):
                var2ground = dict(zip(free_vars, ground_vars))
                full_ground_vars = {**head_body_ground_vars,
                                    **var2ground}
                # print('for every possible grounding of the free vars',ground_vars) if cont< lim else None
                accepted = True
                body_grounding = []
                unknown_fact_count = 0
                for j in range(len(rule.body)):
                    # print('     ---j', j) if cont< lim else None
                    if i == j:
                        # print('      body_grounding,i=j', ground_atom, 'it is already in facts') if cont< lim else None
                        new_ground_atom = ground_atom
                        is_known_fact = True  # by definition as it is coming from the groundings.
                    else:
                        body_atom2 = rule.body[j]
                        new_ground_atom = (body_atom2[0], ) + tuple(
                            [full_ground_vars.get(body_atom2[k+1], None)
                             for k in range(len(body_atom2)-1)])
                        is_known_fact = (len(fact_index._index.get(new_ground_atom, [])) > 0)

                    assert all(new_ground_atom), 'Some unexpected free variables in %s' % str(new_ground_atom)

                    if not is_known_fact and (
                            max_unknown_fact_count < 0 or unknown_fact_count < max_unknown_fact_count):
                        body_grounding.append(ground_atom)
                        unknown_fact_count += 1
                        # print('      ', new_ground_atom, 'it is not in facts. Unknown count=',unknown_fact_count,'/',max_unknown_fact_count ) if cont< lim else None
                    elif is_known_fact:
                        body_grounding.append(new_ground_atom)
                        # print('      ', new_ground_atom, 'it is in facts') if cont< lim else None
                    else:
                        # print('      ', new_ground_atom, 'it is not in facts. Unknown count=',unknown_fact_count,'/',max_unknown_fact_count,'.Not accepted') if cont< lim else None
                        accepted = False
                        break

                if accepted:
                    print('ADDED', q, '->', tuple(body_grounding),'. Number of missing atoms:', unknown_fact_count) if cont< lim else None
                    new_ground_atoms.add(((q,), tuple(body_grounding)))
                    # print('------UPDATED NEW GROUNDINGS', new_ground_atoms) if cont< lim else None

    end = time.time()
    print('NUM_GROUNDINGS', len(new_ground_atoms), end - start)
    print('NEW GROUNDINGS: ', new_ground_atoms)  
    if res is None:
        return new_ground_atoms
    else:
        res.update(new_ground_atoms)


class BackwardChainingGrounder(Engine):

    def __init__(self, rules: List[Rule], facts: List[Union[Atom, str, Tuple]],
                 domains: Dict[str, Domain],
                 max_unknown_fact_count: int=1,
                 num_steps: int=1,
                 # Unused.
                 n_threads: int=kNThreads):
        self.max_unknown_fact_count = max_unknown_fact_count
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
        print('initiating internals')
        print('queries', len(queries),queries)
        # Reset relation2queries and rule2groundings
        # Relation to queries is: {predicate: queries for that predicate}
        self.relation2queries = {}
        for q in queries:
            if q[0] not in self.relation2queries:
                self.relation2queries[q[0]] = []
            self.relation2queries[q[0]].append(q)
        # rule2groundings is: {rule: groundings for that rule}
        self.rule2groundings = {}
        for rule in self.rules:
            if rule.name not in self.rule2groundings:
                self.rule2groundings[rule.name] = set()
        print('relation2queries\n', self.relation2queries)
        print('\nrule2groundings\n', self.rule2groundings)

    # Ground a batch of queries, the result is cached for speed.
    def ground(self,
               facts: List[Tuple],
               queries: List[Tuple],
               **kwargs) -> Dict[str, RuleGroundings]:

        if self.rules is None or len(self.rules) == 0:
            return []
        print('queries', len(queries), queries)
        self._init_internals(queries)
        # Keeps track of the queris already processed for this rul.
        self._rule2processed_queries = {rule.name: set() for rule in self.rules}
        for step in range(self.num_steps):
            print('STEP NUMBER ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', step,'^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^','step ',step+1,'/', self.num_steps, 'known body',step == self.num_steps - 1, )
            for rule in self.rules:
                print('\nrule ', rule, ' """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" ')
                queries_per_rule = self.relation2queries.get(rule.head[0][0], [])
                print('queries_per_rule\n', len(queries_per_rule),queries_per_rule)
                print('rule2groundings before\n', self.rule2groundings)
                if not queries_per_rule:
                    continue
                backward_chaining_grounding_one_rule_with_domains(
                    self.domains, rule, queries_per_rule, self._fact_index,
                    # max_unknown_fact_count
                    (self.max_unknown_fact_count if step < self.num_steps - 1 else 0),
                    # Output added here.
                    self.rule2groundings[rule.name])
                print('\nrule2groundings after\n', self.rule2groundings)

            if self.num_steps > 1:  # expand the queries and iterate.
                rule2new_queries = {}
                for rule in self.rules:
                    # These are the just processed queries.
                    queries_per_rule = self.relation2queries.get(rule.head[0][0], [])
                    # Update the list of processed rules by appending the new queries.
                    self._rule2processed_queries[rule.name] = ( self._rule2processed_queries[rule.name] | set(queries_per_rule) )
                    print('\nFinished. self._rule2processed_queries for: ',rule,'\n', self._rule2processed_queries[rule.name])
                    print('\nAtoms in the groundings for rule ', rule.name, '\n', get_atoms_on_groundings(self.rule2groundings[rule.name]))
                    # Calculate the atoms left to prove as the difference between the atoms in the groundings and the processed queries.
                    ################
                    # The new queries that are left to prove.
                    # We get the new groundings if they are not already in the processed queries.
                    rule2new_queries[rule.name] = [a for a in get_atoms_on_groundings(self.rule2groundings[rule.name])
                                                   if a not in self._rule2processed_queries[rule.name]]
                    print('\nrule2new_queries[rule.name]\n', rule2new_queries[rule.name])
                # Merge all new queries in a single list.
                new_queries = list(set(chain.from_iterable(rule2new_queries.values())))
                self._init_internals(new_queries)

        if 'deterministic' in kwargs and kwargs['deterministic']:
            ret = {rule.name:
                   RuleGroundings(rule_name,
                                  sorted(list(groundings), key=lambda x : x.__repr__()))
                   for rule_name,groundings in self.rule2groundings.items()}
        else:
            ret = {rule.name:
                   RuleGroundings(rule_name, list(groundings))
                   for rule_name,groundings in self.rule2groundings.items()}

        return ret