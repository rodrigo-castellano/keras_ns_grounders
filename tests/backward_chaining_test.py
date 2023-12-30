import unittest
from keras_ns.grounding.backward_chaining_grounder import BackwardChainingGrounder
from keras_ns.logic.commons import Atom, Domain, Rule, RuleGroundings


def count(r2g):
    return sum([sum([len(rule.head) + len(rule.body) for rule in gs])
                     for r,gs in r2g.items()])

class BackwardChainingTest(unittest.TestCase):

    # Test 1 simple 1-step bridge
    def test_one_step_bridge(self):
        queries = ['friend(Carlo,Paolo).']
        facts = ['friend(Carlo,Pluto).']
        queries = [Atom(s=a, format='functional').toTuple() for a in queries]
        facts = [Atom(s=a, format='functional').toTuple() for a in facts]
        rules = ['trans1 friend(x,y) friend(y,z) friend(x,z)']
        rules = [Rule(s=r, format='expressgnn') for r in rules]
        bcg = BackwardChainingGrounder(rules, facts, num_steps=1)
        res = bcg.ground(queries=queries, facts=[], deterministic=True)
        # print('RET', res[0])
        res_rule = [RuleGroundings('trans1',
                                   [ # list of groundings
                                   ( # 2 dim-tuple
                                   ( # tuple of head atoms
                                   ('friend', 'Carlo', 'Paolo'),),
                                   ( # tuple of body atoms
                                   ('friend', 'Carlo', 'Pluto'),
                                   ('friend', 'Pluto', 'Paolo'))
                                   )])]
        self.assertEqual(res[0], res_rule[0])

    def test_one_step_bridge_with_single_domains(self):
        queries = ['friend(Carlo,Paolo).']
        facts = ['friend(Carlo,Pluto).']
        domain = Domain('Person', ['Carlo', 'Paolo', 'Pluto'])
        domains = {'x':domain, 'y':domain}
        queries = [Atom(s=a, format='functional').toTuple() for a in queries]
        facts = [Atom(s=a, format='functional').toTuple() for a in facts]
        rules = ['trans1 friend(x,y) friend(y,z) friend(x,z)']
        rules = [Rule(s=r, format='expressgnn') for r in rules]
        bcg = BackwardChainingGrounder(rules, facts, domains=domains,
                                       num_steps=1)
        res = bcg.ground(queries=queries, facts=[], deterministic=True)
        # print('RET', res[0])
        res_rule = [RuleGroundings('trans1',
                                   [ # list of groundings
                                   ( # 2 dim-tuple
                                   ( # tuple of head atoms
                                   ('friend', 'Carlo', 'Paolo'),),
                                   ( # tuple of body atoms
                                   ('friend', 'Carlo', 'Pluto'),
                                   ('friend', 'Pluto', 'Paolo'))
                                   )])]
        self.assertEqual(res[0], res_rule[0])

    def test_one_step_bridge_with_two_domains(self):
        queries = ['friend(Carlo,Paolo).']
        facts = ['friend(Carlo,Pluto).', 'locin(Carlo,France).']
        domain = Domain('Person', ['Carlo', 'Paolo', 'Pluto'])
        domain1 = Domain('Country', ['Italy', 'France'])
        domains = {'Person':domain, 'Country':domain1}
        queries = [Atom(s=a, format='functional').toTuple() for a in queries]
        facts = [Atom(s=a, format='functional').toTuple() for a in facts]
        rules = ['trans1 locin(x,c) friend(x,y) friend(y,z) friend(x,z)']
        rules = [Rule(s=r, format='expressgnn',
                      var2domain={'x':'Person', 'y':'Person', 'z':'Person',
                                  'c':'Country'}) for r in rules]
        bcg = BackwardChainingGrounder(rules, facts, domains=domains,
                                       num_steps=1)
        res = bcg.ground(queries=queries, facts=[], deterministic=True)
        print('\nRET', res[0])
        res_rule = [RuleGroundings('trans1',
                                   [ # list of groundings
                                   ( # 2 dim-tuple
                                   ( # tuple of head atoms
                                   ('friend', 'Carlo', 'Paolo'),),
                                   ( # tuple of body atoms
                                   ('locin',  'Carlo', 'France'),
                                   ('friend', 'Carlo', 'Carlo'),
                                   ('friend', 'Carlo', 'Paolo'))
                                   ),
                                   ( # 2 dim-tuple
                                   ( # tuple of head atoms
                                   ('friend', 'Carlo', 'Paolo'),),
                                   ( # tuple of body atoms
                                   ('locin',  'Carlo', 'France'),
                                   ('friend', 'Carlo', 'Paolo'),
                                   ('friend', 'Paolo', 'Paolo'))
                                   ),
                                   ( # 2 dim-tuple
                                   ( # tuple of head atoms
                                   ('friend', 'Carlo', 'Paolo'),),
                                   ( # tuple of body atoms
                                   ('locin',  'Carlo', 'France'),
                                   ('friend', 'Carlo', 'Pluto'),
                                   ('friend', 'Pluto', 'Paolo'))
                                   ),
                                   ( # 2 dim-tuple
                                   ( # tuple of head atoms
                                   ('friend', 'Carlo', 'Paolo'),),
                                   ( # tuple of body atoms
                                   ('locin',  'Carlo', 'Italy'),
                                   ('friend', 'Carlo', 'Pluto'),
                                   ('friend', 'Pluto', 'Paolo'))
                                   )])]
        self.assertEqual(res[0], res_rule[0])


    def test_two_multi_step_bridge(self):
        queries = ['friend(Carlo,Paolo).']
        facts = ['friend(Carlo,Pluto).', 'friend(Pluto,Marta).']
        queries = [Atom(s=a, format='functional').toTuple() for a in queries]
        facts = [Atom(s=a, format='functional').toTuple() for a in facts]
        rules = ['trans1 friend(x,y) friend(y,z) friend(x,z)']
        rules = [Rule(s=r, format='expressgnn') for r in rules]
        bcg = BackwardChainingGrounder(rules, facts, num_steps=2)
        res = bcg.ground(queries=queries, facts=[], deterministic=True)

        res_rule = [RuleGroundings('trans1',
                                   [ # list of groundings
                                   ( # 2 dim-tuple
                                   ( # tuple of head atoms
                                   ('friend', 'Carlo', 'Paolo'),),
                                   ( # tuple of body atoms
                                   ('friend', 'Carlo', 'Pluto'),
                                   ('friend', 'Pluto', 'Paolo'))
                                   )
                                   ])
                    ]
        self.assertEqual(res, res_rule)


    def test_random_data(self):
        import random
        queries = ['friend(p%02d,p%02d).' % (random.randint(0, 100),
                                             random.randint(0, 100))
                   for i in range(100)]
        facts = ['friend(p%02d,p%02d).' % (random.randint(0, 100),
                                           random.randint(0, 100))
                 for i in range(100)]
        rules = ['trans1 friend(x,y) friend(y,z) friend(x,z)']
        rules = [Rule(s=r, format='expressgnn') for r in rules]
        queries = [Atom(s=a, format='functional').toTuple() for a in queries]
        facts = [Atom(s=a, format='functional').toTuple() for a in facts]

        for ns in range(10):
            bcg = BackwardChainingGrounder(rules, facts, num_steps=ns)
            # print(ns, count(bcg.ground(queries=queries, facts=[])))
            bcg.ground(queries=queries, facts=[], deterministic=True)


    #
    # ####################################
    # # Test 4 symmetric data.
    def test_symmetric_rule(self):
        queries = ['friend(a,b).']
        facts = ['friend(a,b).', 'friend(b,a).']
        queries = [Atom(s=a, format='functional').toTuple() for a in queries]
        facts = [Atom(s=a, format='functional').toTuple() for a in facts]
        rules = ['symm friend(x,y) friend(y,x)']
        rules = [Rule(s=r, format='expressgnn') for r in rules]
        bcg = BackwardChainingGrounder(rules, facts,  num_steps=1)
        res = bcg.ground(queries=queries, facts=[], deterministic=True)

        res_rule = [RuleGroundings('symm',
                                   [ # list of groundings
                                   ( # 2 dim-tuple
                                   ( # tuple of head atoms
                                   ('friend', 'a', 'b'),),
                                   ( # tuple of body atoms
                                   ('friend', 'b', 'a'),)
                                   )])]
        self.assertEqual(res[0], res_rule[0])

if __name__ == '__main__':
    unittest.main()
