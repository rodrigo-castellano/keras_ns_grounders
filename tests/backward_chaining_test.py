import unittest
from ns_lib.grounding.backward_chaining_grounder import BackwardChainingGrounder
from ns_lib.logic.commons import Atom, Domain, Rule, RuleGroundings, FOL


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
        rule_name = 'trans1'
        rules = ['%s friend(x,y) friend(y,z) friend(x,z)' % rule_name]
        rules = [Rule(s=r, format='expressgnn') for r in rules]
        fol = FOL.Build(facts=facts)
        bcg = BackwardChainingGrounder(
            rules, facts=facts, domains={d.name:d for d in fol.domains},
            num_steps=1, max_unknown_fact_count=1, prune_incomplete_proofs=False)

        res = bcg.ground(queries=queries, facts=[], deterministic=True)
        res_rule = RuleGroundings(rule_name,
                                  [ # list of groundings
                                  ( # 2 dim-tuple
                                  ( # tuple of head atoms
                                  ('friend', 'Carlo', 'Paolo'),),
                                  ( # tuple of body atoms
                                  ('friend', 'Carlo', 'Pluto'),
                                  ('friend', 'Pluto', 'Paolo'))
                                  )])
        self.assertEqual(res[rule_name], res_rule)

    def test_one_step_bridge_with_single_domains(self):
        queries = ['friend(Carlo,Paolo).']
        facts = ['friend(Carlo,Pluto).']
        domain = Domain('Person', ['Carlo', 'Paolo', 'Pluto'])
        domains = {'x':domain, 'y':domain}
        queries = [Atom(s=a, format='functional').toTuple() for a in queries]
        facts = [Atom(s=a, format='functional').toTuple() for a in facts]
        rule_name = 'trans1'
        rules = ['%s friend(x,y) friend(y,z) friend(x,z)' % rule_name]
        rules = [Rule(s=r, format='expressgnn') for r in rules]
        fol = FOL.Build(facts=facts)
        bcg = BackwardChainingGrounder(
            rules, facts, domains=domains,
            num_steps=1, max_unknown_fact_count=1, prune_incomplete_proofs=False)
        res = bcg.ground(queries=queries, facts=[], deterministic=True)
        res_rule = RuleGroundings(rule_name,
                                  [ # list of groundings
                                  ( # 2 dim-tuple
                                  ( # tuple of head atoms
                                  ('friend', 'Carlo', 'Paolo'),),
                                  ( # tuple of body atoms
                                  ('friend', 'Carlo', 'Pluto'),
                                  ('friend', 'Pluto', 'Paolo'))
                                  )])
        self.assertEqual(res[rule_name], res_rule)

    def test_one_step_bridge_with_two_domains(self):
        queries = ['friend(Carlo,Paolo).']
        facts = ['friend(Carlo,Pluto).', 'locin(Carlo,France).']
        domain = Domain('Person', ['Carlo', 'Paolo', 'Pluto'])
        domain1 = Domain('Country', ['Italy', 'France'])
        name2domain = {'Person':domain, 'Country':domain1}
        queries = [Atom(s=a, format='functional').toTuple() for a in queries]
        facts = [Atom(s=a, format='functional').toTuple() for a in facts]
        rule_name = 'trans1'
        rules = ['%s locin(x,c) friend(x,y) friend(y,z) friend(x,z)' % rule_name]
        rules = [Rule(s=r, format='expressgnn',
                      var2domain={'x':'Person', 'y':'Person', 'z':'Person',
                                  'c':'Country'}) for r in rules]
        fol = FOL.Build(
            constants=['Carlo', 'Paolo', 'Pluto', 'France', 'Italy'],
            facts=facts,
            domain2constants={n:d.constants for n,d in name2domain.items()})
        bcg = BackwardChainingGrounder(
            rules, facts=facts, domains=name2domain,
            num_steps=1, max_unknown_fact_count=2, prune_incomplete_proofs=False)
        res = bcg.ground(queries=queries, facts=[], deterministic=True)
        res_rule = RuleGroundings(rule_name,
                                  [ # list of groundings
                                  ( # 2 dim-tuple
                                  ( # tuple of head atoms
                                  ('friend', 'Carlo', 'Paolo'),),
                                  ( # tuple of body atoms
                                  ('locin',  'Carlo', 'France'),
                                  ('friend', 'Carlo', 'Pluto'),
                                  ('friend', 'Pluto', 'Paolo'))
                                  ),
                                  ])
        self.assertEqual(res[rule_name], res_rule)

    def test_one_step_bridge_with_two_domains(self):
        print('\n\ntest_one_step_bridge_with_two_domains')
        queries = ['friend(Carlo,Paolo).']
        facts = ['friend(Carlo,Pluto).', 'locin(Carlo,France).']
        domain = Domain('Person', ['Carlo', 'Paolo', 'Pluto'])
        domain1 = Domain('Country', ['Italy', 'France'])
        name2domain = {'Person':domain, 'Country':domain1}
        queries = [Atom(s=a, format='functional').toTuple() for a in queries]
        facts = [Atom(s=a, format='functional').toTuple() for a in facts]
        rule_name = 'trans1'
        rules = ['%s locin(x,c) friend(x,y) friend(y,z) friend(x,z)' % rule_name]
        rules = [Rule(s=r, format='expressgnn',
                      var2domain={'x':'Person', 'y':'Person', 'z':'Person',
                                  'c':'Country'}) for r in rules]
        fol = FOL.Build(
            constants=['Carlo', 'Paolo', 'Pluto', 'France', 'Italy'],
            facts=facts,
            domain2constants={n:d.constants for n,d in name2domain.items()})
        bcg = BackwardChainingGrounder(
            rules, facts=facts, domains=name2domain,
            num_steps=1, max_unknown_fact_count=2,
            max_unknown_fact_count_last_step=2, prune_incomplete_proofs=False)
        res = bcg.ground(queries=queries, facts=[], deterministic=True)
        res_rule = RuleGroundings(rule_name,
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
                                  )])
        self.assertEqual(res[rule_name], res_rule)

    def test_one_step_bridge_with_two_domains_keep_proofs(self):
        print('\n\ntest_one_step_bridge_with_two_domains_keep_proofs')
        queries = ['friend(Carlo,Paolo).']
        facts = ['friend(Carlo,Pluto).', 'locin(Carlo,France).']
        domain = Domain('Person', ['Carlo', 'Paolo', 'Pluto'])
        domain1 = Domain('Country', ['Italy', 'France'])
        name2domain = {'Person':domain, 'Country':domain1}
        queries = [Atom(s=a, format='functional').toTuple() for a in queries]
        facts = [Atom(s=a, format='functional').toTuple() for a in facts]
        rule_name = 'trans1'
        rules = ['%s locin(x,c) friend(x,y) friend(y,z) friend(x,z)' % rule_name]
        rules = [Rule(s=r, format='expressgnn',
                      var2domain={'x':'Person', 'y':'Person', 'z':'Person',
                                  'c':'Country'}) for r in rules]
        fol = FOL.Build(
            constants=['Carlo', 'Paolo', 'Pluto', 'France', 'Italy'],
            facts=facts,
            domain2constants={n:d.constants for n,d in name2domain.items()})
        bcg = BackwardChainingGrounder(
            rules, facts=facts, domains=name2domain,
            num_steps=1, max_unknown_fact_count=2,
            max_unknown_fact_count_last_step=2, prune_incomplete_proofs=True)
        res = bcg.ground(queries=queries, facts=[], deterministic=True)
        res_rule = RuleGroundings(rule_name, [])
        self.assertEqual(res[rule_name], res_rule)


    def test_two_multi_step_bridge(self):
        print('\n\ntest_two_multi_step_bridge')
        queries = ['friend(Carlo,Paolo).']
        facts = ['friend(Carlo,Pluto).', 'friend(Pluto,Marta).']
        queries = [Atom(s=a, format='functional').toTuple() for a in queries]
        facts = [Atom(s=a, format='functional').toTuple() for a in facts]
        rule_name = 'trans1'
        rules = ['%s friend(x,y) friend(y,z) friend(x,z)' % rule_name]
        rules = [Rule(s=r, format='expressgnn') for r in rules]
        fol = FOL.Build(facts=facts)
        bcg = BackwardChainingGrounder(
            rules, facts=facts, domains={d.name:d for d in fol.domains},
            num_steps=2, max_unknown_fact_count=1, max_unknown_fact_count_last_step=1,
            prune_incomplete_proofs=False)
        res = bcg.ground(queries=queries, facts=[], deterministic=True)

        res_rule = RuleGroundings(rule_name,
                                  [ # list of groundings
                                  ( # 2 dim-tuple
                                  ( # tuple of head atoms
                                  ('friend', 'Carlo', 'Paolo'),),
                                  ( # tuple of body atoms
                                  ('friend', 'Carlo', 'Pluto'),
                                  ('friend', 'Pluto', 'Paolo'))  # To prove
                                  ),
                                  ( # 2 dim-tuple
                                  ( # tuple of head atoms
                                  ('friend', 'Pluto', 'Paolo'),),
                                  ( # tuple of body atoms
                                  ('friend', 'Pluto', 'Marta'),
                                  ('friend', 'Marta', 'Paolo'))
                                  )
                                  ])
        self.assertEqual(res[rule_name], res_rule)


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
        fol = FOL.Build(facts=facts)

        for ns in range(10):
            bcg = BackwardChainingGrounder(
                rules, facts=facts, domains={d.name:d for d in fol.domains},
                num_steps=ns, max_unknown_fact_count=1, prune_incomplete_proofs=False)
            #_ = bcg.ground(queries=queries, facts=[], deterministic=True)


    #
    # ####################################
    # # Test 4 symmetric data.
    def test_symmetric_rule(self):
        queries = ['friend(a,b).']
        facts = ['friend(a,b).', 'friend(b,a).']
        queries = [Atom(s=a, format='functional').toTuple() for a in queries]
        facts = [Atom(s=a, format='functional').toTuple() for a in facts]
        rule_name = 'symm'
        rules = ['%s friend(x,y) friend(y,x)' % rule_name]
        rules = [Rule(s=r, format='expressgnn') for r in rules]
        fol = FOL.Build(facts=facts)
        bcg = BackwardChainingGrounder(
            rules, facts=facts, domains={d.name:d for d in fol.domains},
            num_steps=1, max_unknown_fact_count=1, prune_incomplete_proofs=False)
        res = bcg.ground(queries=queries, facts=[], deterministic=True)

        res_rule = RuleGroundings(rule_name,
                                  [ # list of groundings
                                  ( # 2 dim-tuple
                                  ( # tuple of head atoms
                                  ('friend', 'a', 'b'),),
                                  ( # tuple of body atoms
                                  ('friend', 'b', 'a'),)
                                  )])
        self.assertEqual(res[rule_name], res_rule)

if __name__ == '__main__':
    unittest.main()
