import abc
from typing import Dict, List, Tuple
from keras_ns.logic.commons import Rule, RuleGroundings
from keras_ns.grounding.engine import Engine

#############################################
class SubstitutionGrounder(Engine):

    def __init__(self, rules: List[Rule], depth: int = 0):
        self.rules = rules
        self.depth = depth

    @abc.abstractmethod
    def queries_to_substitutions(self, queries, rule):
        pass

    def ground(self,
               facts: List[Tuple],
               queries: List[Tuple],
               **kwargs) -> Dict[str, RuleGroundings]:
        res = {}
        for rule in self.rules:
            groundings = []
            # for query in queries:  # avoid 1 call per query.
            for substitution in self.queries_to_substitutions(queries, rule):
                body_atoms = []
                for atom in rule.body:
                    ground_atom = (atom[0], ) + tuple(
                        [substitution.get(atom[j+1], None)
                         for j in range(len(atom)-1)])
                    assert all(ground_atom), 'Unresolved %s' % str(ground_atom)
                    body_atoms.append(ground_atom)

                head_atoms = []
                for atom in rule.head:
                    ground_atom = (atom[0], ) + tuple(
                        [substitution.get(atom[j+1], None)
                         for j in range(len(atom)-1)])
                    assert all(ground_atom), 'Unresolved %s' % str(ground_atom)
                    head_atoms.append(ground_atom)
                groundings.append((tuple(head_atoms), tuple(body_atoms)))

            res[rule.name] = RuleGroundings(rule.name, groundings=groundings)
        return res

#############################################
class LocalFlatGrounder(SubstitutionGrounder):

    def __init__(self, rules:list[Rule], depth: int = 0):
        super().__init__(rules, depth)

    def queries_to_substitutions(self, queries, rule):
        substitutions = []

        for query in queries:
            assert len(rule.vars) == len(query[1:])
            for a in rule.body + rule.head:
                if query[0] == a[0]:
                    # a=p(X,Y) (non ground) of rule(head+body); query is p(a,b)
                    substitutions.append(
                        {variable:constant
                         for variable,constant in zip(a[1:],query[1:])})
        return substitutions
