from typing import Dict, List, Set, Tuple, Union
from itertools import product
from keras_ns.logic.commons import Atom, Domain, Rule, RuleGroundings
from keras_ns.grounding.engine import Engine

#def parse_query(query: str) -> Tuple[str, str, str]:
#    query = query.replace(' ', '').replace('(', ',').replace(')', '')
#    return tuple(query.split(','))
#def query_from_tuple(parsed_query: Tuple[str, str, str]) -> str:
#    return parsed_query[0] + '(' + ','.join(parsed_query[1:]) + ')'

class PlaceholderGeneratorFullGrounder(Engine):

    def __init__(self, domains: Dict[str, Domain],
                 rules: List[Rule],
                 domain2adaptive_constants: Dict[str, List[str]]=None,
                 exclude_symmetric: bool=False,
                 exclude_query: bool=False,
                 limit: int=0):  # 0 means no limit.

        self.domains = domains
        self.rules = rules
        self.domain2num_constants = {
            k:len(d.constants) for k,d in self.domains.items()}
        self.domain2adaptive_constants = domain2adaptive_constants
        self.limit = limit
        self.exclude_symmetric = exclude_symmetric
        self.exclude_query = exclude_query

    def _init_internals(self, queries: List[Tuple]):
        self.rule2groundings = {}
        for rule in self.rules:
            #if rule.name not in self.rule2groundings:
            self.rule2groundings[rule.name] = set()

    #@lru_cache
    def ground(self,
               facts: List[Tuple],
               queries: List[Tuple],
               **kwargs) -> Dict[str, RuleGroundings]:

        self._init_internals(queries)

        #for query in queries:
        #    res[query] = {}
        #    parsed_query = parse_query(query)  # ('path', '0', '1')
        for clause in self.rules:
            added: int = 0
            groundings = []
            # substitutions = []
            tuples_per_rule = product(
                *[self.domains[name].constants +
                  self.domain2adaptive_constants.get(name, [])
                  for _,name in clause.vars2domain.items()])

            for ground_vars in tuples_per_rule:

                var_assignments = {k:v for k,v in zip(clause.vars2domain.keys(), ground_vars)}

                # We use a lexicographical order of the variables
                constant_tuples = [v for k,v in sorted(var_assignments.items(),
                                                       key=lambda x: x[0])]

                is_good: bool = True
                body_atoms = []
                for atom in clause.body:
                    ground_atom = (atom[0], ) + tuple(
                        [var_assignments.get(atom[j+1], None) for j in range(len(atom)-1)])
                    assert all(ground_atom), 'Unresolved %s' % str(ground_atom)
                    if (self.exclude_symmetric and
                        ground_atom[1] == ground_atom[2]):
                        is_good = False
                        break
                    if self.exclude_query and ground_atom in queries:
                        is_good = False
                        break
                    body_atoms.append(ground_atom)

                head_atoms = []
                for atom in clause.head:
                    ground_atom = (atom[0], ) + tuple(
                        [var_assignments.get(atom[j+1], None) for j in range(len(atom)-1)])
                    assert all(ground_atom), 'Unresolved %s' % str(ground_atom)
                    if (self.exclude_symmetric and
                        ground_atom[1] == ground_atom[2]):
                        is_good = False
                        break
                    if self.exclude_query and ground_atom in queries:
                        is_good = False
                        break
                    head_atoms.append(ground_atom)
                if is_good:
                    groundings.append((tuple(head_atoms), tuple(body_atoms)))
                    # substitutions.append(constant_tuples)
                    added += 1
                    if self.limit > 0 and self.limit >= added:
                        break

            self.rule2groundings[clause.name].update(groundings)
            #res[query][clause.name] = (groundings, substitutions)

        res = {rule_name:
               RuleGroundings(rule_name, list(groundings))
               for rule_name,groundings in self.rule2groundings.items()}
        return res
