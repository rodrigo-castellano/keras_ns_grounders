import unittest
from keras_ns.grounding.known_body_forward_grounder import KnownBodyForwardGrounder
from keras_ns.logic.commons import Atom, Rule, RuleGroundings


def count(r2g):
    return sum([sum([len(rule.head) + len(rule.body) for rule in gs])
                     for r,gs in r2g.items()])

class KnownBodyForwardGrounderTest(unittest.TestCase):

    # Test 1 simple 1-step bridge
    def test_one_step_bridge(self):
        queries = ['friend(Carlo,Pluto).']
        facts = ['friend(Paolo,Carlo).']
        queries = [Atom(s=a, format='functional').toTuple() for a in queries]
        facts = [Atom(s=a, format='functional').toTuple() for a in facts]
        rules = ['trans1 friend(x,y) friend(y,z) friend(x,z)']
        rules = [Rule(s=r, format='expressgnn') for r in rules]
        bcg = KnownBodyForwardGrounder(rules, facts)
        res = bcg.ground(queries=queries, facts=[])
        print('RET', res[0])
        #res_rule = [RuleGroundings('trans1', [BackwardChainingRule(
        #    head_atoms=[('friend', 'Carlo', 'Paolo')],
        #    body_atoms=[('friend', 'Carlo', 'Pluto'),
        #                ('friend', 'Pluto', 'Paolo')])])]
        res_rule = [RuleGroundings('trans1',
                                   [ # list of groundings
                                   ( # 2 dim-tuple
                                   ( # tuple of head atoms
                                   ('friend', 'Paolo', 'Pluto'),),
                                   ( # tuple of body atoms
                                   ('friend', 'Carlo', 'Pluto'),
                                   ('friend', 'Paolo', 'Carlo'))
                                   )])]
        print('TAR', res_rule[0])
        self.assertEqual(res[0], res_rule[0])

    # ####################################
    # # Test 4 symmetric data.
    def test_symmetric_rule(self):
        queries = ['friend(a,b).']
        facts = ['friend(a,b).', 'friend(b,a).']
        queries = [Atom(s=a, format='functional').toTuple() for a in queries]
        facts = [Atom(s=a, format='functional').toTuple() for a in facts]
        rules = ['symm friend(x,y) friend(y,x)']
        rules = [Rule(s=r, format='expressgnn') for r in rules]
        bcg = KnownBodyForwardGrounder(rules, facts)
        res = bcg.ground(queries=queries, facts=[])

        res_rule = [RuleGroundings('symm',
                                   [ # list of groundings
                                   ( # 2 dim-tuple
                                   ( # tuple of head atoms
                                   ('friend', 'b', 'a'),),
                                   ( # tuple of body atoms
                                   ('friend', 'a', 'b'),)
                                   )])]
        self.assertEqual(res[0], res_rule[0])

if __name__ == '__main__':
    unittest.main()
