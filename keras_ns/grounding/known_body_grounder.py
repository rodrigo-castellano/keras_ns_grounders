#! /bin/python3
import copy
from typing import Dict, List, Set, Tuple, Union
from keras_ns.logic.commons import Atom, Rule, RuleGroundings
from keras_ns.grounding.engine import Engine
from keras_ns.grounding.backward_chaining_grounder import AtomIndex
from itertools import product

class KnownBodyGrounder(Engine):

    def __init__(self, rules: List[Rule], facts: List[Union[Atom, str, Tuple]]):
        self._name2rule = {rule.name: rule for rule in rules}
        self.rules = rules
        self.facts = [a if isinstance(a,Tuple) else
                      (a.toTuple() if isinstance(a,Atom) else
                      Atom(s=a).toTuple()) for a in facts]
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

    def ground(self,
               facts: List[Tuple],
               queries: List[Tuple],
               **kwargs) -> Dict[str, RuleGroundings]:

        if self.rules is None or len(self.rules) == 0:
            return []
        # print('queries', len(queries), queries)
        self._init_internals(queries)
        print('queries', len(queries),queries[:50])
        # print('relation2queries')
        # for k,v in self.relation2queries.items():
        #     print(k, len(v))
        # print('rule2groundings', self.rule2groundings)
        
        for rule in self.rules:
            print('\nrule ', rule, ' """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" ')
            # print('rule', rule)
            # print('rule.body', rule.body)
            # print('rule.head[0][0]', rule.head[0][0])
            rel_queries = self.relation2queries.get(rule.head[0][0], [])
            # print('rel_queries',rel_queries)
            if rel_queries:
                if len(rule.body) <= 2:
                    self.ground_one_rule_body_len2(rule, rel_queries)
                else:
                    self.ground_one_rule(rule, rel_queries)
        if 'deterministic' in kwargs and kwargs['deterministic']:
            ret = {rule.name:
                   RuleGroundings(rule_name, sorted(list(groundings),
                                                    key=lambda x : x.__repr__()))
                   for rule_name,groundings in self.rule2groundings.items()}
        else:
            ret = {rule.name:
                   RuleGroundings(rule_name, list(groundings))
                   for rule_name,groundings in self.rule2groundings.items()}
        return ret

    # Special case for rules with body len < 2.
    def ground_one_rule_body_len2(
        self, rule: Rule, queries: List[Tuple]):
      # We have a rule like A(x,y) B(y,z) => C(x,z)
      assert len(rule.head) == 1, (
          'Rule is not a Horn clause %s' % str(rule))
      head = rule.head[0]

      assert len(rule.body) <= 2, (
          'Horn clauses with more than 2 elements in the body are '
          'not supported yet')

      new_groundings = set()
      cont = 0 
      lim=300000
      for q in queries:
        # print('\n\n***************q', q,'********************') if cont< lim else None
        cont +=1

        if q[0] != head[0]:  # predicates must match.
          continue

        # Get the variable assignments from the head.
        head_var_assignments = {v: a for v, a in zip(head[1:], q[1:])}

        # Ground first body atom by replacing variables with constants.
        # The result is the partially ground atom A('Antonio',None)
        # with None indicating unground variables.
        body_atom = rule.body[0]
        ground_body_atom = (body_atom[0], ) + tuple(
            [head_var_assignments.get(body_atom[j+1], None)
             for j in range(len(body_atom)-1)])
        # print('\nground_body_atom:', ground_body_atom, '. Substitution (by None) of the vars not present in head.') if cont< lim else None
        if all(ground_body_atom[1:]):
          # Variables all match, so we have already the wanted grounding.
          # Rule was in the form A(x,y) ^ ... -> B(x,y)
        #   print('groundings already done, #all vars are subtituted', groundings) if cont< lim else None
          groundings = (ground_body_atom,)
        else:
          # One varibale match, rule was in the form A(x,z) ^ ... -> B(x,y)
          # Tuple of atoms matching A(Antonio,None) in the facts.
          # This is the list of ground atoms for the i-th atom in the body.
          # groundings = self._fact_index.get_matching_atoms(ground_body_atom)
          groundings = self._fact_index._index.get(ground_body_atom, [])  # optimization to avoid one extra function call
        #   print('groundings found in facts', groundings) if cont< lim else None
        if len(rule.body) == 1:
        #   print('length one in the body, one predicate') if cont< lim else None  
          # Shortcut, we are done, the clause has no free variables.
          # Return the groundings.
          new_groundings.add(((q,), groundings))
        #   print('ADDED', q, '->', (groundings,)) if cont< lim else None
          continue

        # Select the other atom in the body and ground it with the
        # assignments with the head and the other body ground atom fixed.
        body_atom2 = rule.body[1]
        
        # print('\nfor every grounding of the body atom') if cont< lim else None
        for atom in groundings:
        #   print('--grounded_atom', atom, ' The other vars (not present in head) are left as free') if cont< lim else None
          head_body_var_assignments = copy.copy(head_var_assignments)
          head_body_var_assignments.update(
              {v: a for v, a in zip(body_atom[1:], atom[1:])})
          new_grounding = (body_atom2[0], ) + tuple(
              [head_body_var_assignments.get(body_atom2[j+1], None)
               for j in range(len(body_atom2)-1)])
          if all(new_grounding) and self._fact_index._index.get(
              new_grounding, []):
              # (body_atom1, body_atom2)
              body_grounding = (atom, new_grounding)
              new_groundings.add(((q,), body_grounding))
            #   print('ADDED', q, '->', tuple(body_grounding))
            #   print('------UPDATED NEW GROUNDINGS', new_groundings) if cont< lim else None

      self.rule2groundings[rule.name].update(new_groundings)
    #   print('NUM_GROUNDINGS', len(new_groundings))
    #   print('NEW GROUNDINGS', new_groundings)

    def ground_one_rule(self, rule: Rule, queries: List[Tuple]):
      # We have a rule like A(x,y) B(y,z) => C(x,z)
      assert len(rule.head) == 1, (
          'Rule is not a Horn clause %s' % str(rule))
      head = rule.head[0]

      new_groundings = set()
      cont = 0
      lim=300000
      for q in queries:
        cont += 1 
        print('\n\n***************q', q,'********************') if cont< lim else None
        if q[0] != head[0]:  # predicates must match.
          continue

        # Get the variable assignments from the head.
        head_ground_vars = {v: a for v, a in zip(head[1:], q[1:])}
        var2constants = {var:[c] for var,c in head_ground_vars.items()}
        for i,body_atom in enumerate(rule.body):
            # Ground first body atom by replacing variables with constants.
            # The result is the partially ground atom A('Antonio',None)
            # with None indicating unground variables.
            ground_body_atom = (body_atom[0], ) + tuple(
                [head_ground_vars.get(body_atom[j], None)
                 for j in range(1, len(body_atom))])
            print('\n- i', i,'. ground_body_atom:', ground_body_atom, '. Substitution (by None) of the vars not present in head.') if cont< lim else None
            # optimization to avoid one extra function call.
            atom_candidates = self._fact_index._index.get(ground_body_atom, [])
            print('groundings found in facts', atom_candidates) if cont< lim else None
            for j in range(1, len(body_atom)):
                if ground_body_atom[j] is not None:
                    print('     -j= ',j,'. var', body_atom[j], 'is already grounded') if cont< lim else None
                    continue
                constant_candidates = [a[j] for a in atom_candidates]
                var = body_atom[j]
                if var in var2constants:
                    var2constants[var] = list(set(var2constants[var]) &
                                              set(constant_candidates))
                else:
                    var2constants[var] = list(set(constant_candidates))
                print('     -j= ',j,'. var', var, 'is not grounded yet. Candidates:', var2constants[var]) if cont< lim else None
                if len(var2constants[var]) == 0:
                    break

        vars = var2constants.keys()
        for ground_vars in product(*[c for c in var2constants.values()]):
            print('for every possible grounding of the free vars',ground_vars) if cont< lim else None
            full_ground_vars = dict(zip(vars, ground_vars))
            ground_body_atoms = []
            for body_atom in rule.body:
                ground_body_atom = (body_atom[0], ) + tuple(
                    [full_ground_vars.get(body_atom[j+1], None)
                     for j in range(len(body_atom)-1)])
                print('     -ground_body_atom:', ground_body_atom, '. Substitution (by None) of the vars not present in head.') if cont< lim else None
                ground_body_atoms.append(ground_body_atom)
            new_groundings.add(((q,), tuple(ground_body_atoms)))
            print('ADDED', q, '->', tuple(ground_body_atoms)) if cont< lim else None
            print('------UPDATED NEW GROUNDINGS', new_groundings) if cont< lim else None
      self.rule2groundings[rule.name].update(new_groundings)
      print('NUM_GROUNDINGS', len(new_groundings))
      print('NEW GROUNDINGS', new_groundings)
