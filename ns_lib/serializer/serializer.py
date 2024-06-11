import sys
sys.path.append('C:\\Users\\rodri\\Downloads\\PhD_code\\Review_grounders\\keras_ns_grounders')
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
import traceback
from typing import Dict, List, Tuple, Union
from ns_lib.logic.commons import Atom, Domain, Predicate, Rule, RuleGroundings
import ns_lib as ns
import timeit
import numpy as np
import tensorflow as tf
DomainName = str
PredicateName = str
ConstantName = str
RuleId = str
PredicateIndex = int


class IndexerBase(ABC):

    @abstractmethod
    def serialize(self, queries:List[List[Tuple]],
                  rule_groundings:List[RuleGroundings]):
        pass

# Factory
def LogicSerializer(
    predicates: List[Predicate],
    domains: List[Domain],
    constant2domain: Dict[str, str],
    debug=False,
    domain2adaptive_constants: Dict[str, List[str]]=None) -> IndexerBase:
    if domain2adaptive_constants is not None:
        assert debug == False, 'Non debug mode is not supported yet.'
    return (LogicSerializerDebug(predicates, domains) if debug else
            LogicSerializerFast(predicates, domains,
                                constant2domain, domain2adaptive_constants))

class LogicSerializerFast(IndexerBase):

    def __init__(self, predicates: List[Predicate], domains: List[Domain],
                 constant2domain_name: Dict[str, str],
                 domain2adaptive_constants: Dict[str, List[str]]=None):
        self.predicates = predicates
        self.domains = domains
        self.constant2domain_name = constant2domain_name

        self.constant_to_global_index = defaultdict(OrderedDict)
        self.adaptive_constant2domain = defaultdict(OrderedDict)
        for domain in domains:
            # Add fixed constants. Global index for each constant in each domain (starting from 0 in each domain)
            for i, constant in enumerate(domain.constants):
                self.constant_to_global_index[domain.name][constant] = i
            # Add adaptive constants.
            if domain2adaptive_constants is not None:
                offset = len(domain.constants)
                adaptive_constants = domain2adaptive_constants.get(
                    domain.name, [])
                for i, constant in enumerate(adaptive_constants):
                    self.constant_to_global_index[domain.name][constant] = (
                        i + offset)
                    self.adaptive_constant2domain[constant] = domain.name



        self.predicate_to_domains = {}
        for predicate in self.predicates:
            self.predicate_to_domains[predicate.name] = [
                domain.name for domain in predicate.domains]
        
        # #####################################

        # Create a constant_to_global_unique_index in which each constant, even if in different domains, has a unique index. This is useful for ultra
        self.constant_to_global_unique_index = defaultdict(OrderedDict)
        counter = 0
        for domain in domains:
            for constant in domain.constants:
                self.constant_to_global_unique_index[domain.name][constant] = counter
                counter += 1

        self.predicate_to_global_index = defaultdict(dict)  # A_predicates with global index
        for i,predicate in enumerate(self.predicates):
            self.predicate_to_global_index[predicate.name] = i


    def serialize(self, queries:List[List[Tuple]],
                  rule_groundings:Dict[str, RuleGroundings]):
        domain_to_global = defaultdict(list)  # X_domains, it's a map, where the i-th element of the list is the local index and its value is the global index
        domain_to_local_constant_index = defaultdict(dict)  # helper, as X_domains but for local indices, created new every batch
        predicate_to_constant_tuples = defaultdict(list)  # A_predicates

        # Set of all atoms in the groundings to index
        all_atoms = ns.utils.to_flat(queries)
        for rg in rule_groundings.values():
            for g in rg.groundings:
                all_atoms += g[0] # head
                all_atoms += g[1] # body
        all_atoms = sorted(list(set(all_atoms)))

        # Bucket them per predicate
        all_atoms_per_predicate = {predicate.name: []
                                   for predicate in self.predicates}
        for atom in all_atoms:
            all_atoms_per_predicate[atom[0]].append(atom)

        atom_to_index = {}
        count = 0
        for predicate in self.predicates: # A loop iterates over each predicate.
            constant_tuples = [] # For each predicate, a list of constant indices is generated for each atom, e.g. LocIn:[[1,34],[2,3],...]
            # get the domains of the predicate
            domains = self.predicate_to_domains[predicate.name]
            for atom in all_atoms_per_predicate[predicate.name]: #For every atom in the bucketed atoms (ordered alphabetically by predicate)
                # print('atom:',atom) #if count_print <100 else None
                atom_to_index[atom] = count  # Assigns an index to the atom, e.g. {LocIn(morocco,spain):0,LocIn(morocco,france):1,...}
                count += 1
                indices_cs = [] # This is for A_predicates, to get the constant indices in a list format, e.g.  (Morocco,Spain)->[1,34] (from the LocIn(morocco,spain) example)

                for c in atom[1:]:
                    domain = (self.constant2domain_name[c]    # get the domain of the constant
                              if c in self.constant2domain_name else
                              self.adaptive_constant2domain[c])
                    # Check that the domain of the constant i is the corresponding domain of the predicate
                    assert domain == domains[atom[1:].index(c)], 'Domain of constant does not match the domain of the predicate'

                    # get the local indices of the ctes built so far for this batch
                    constant_index = domain_to_local_constant_index[domain] # It is a domain_to_global_index but for the local indices, created for every batch
                    if c not in constant_index:  
                        constant_index[c] = len(constant_index) # Add it to domain_to_local_constant_index if not already there
                        domain_to_global[domain].append(
                            self.constant_to_global_index[domain][c]) # Append it to X_domains
                    indices_cs.append(constant_index[c]) # Append the local index of the constant to A_predicates
                    # indices_cs.append(self.constant_to_global_index[domain][c]) # Append the global index of the constant to A_predicates
                    # print('     indices:',indices_cs) if count_print <100 else None
                constant_tuples.append(indices_cs) 
            predicate_to_constant_tuples[predicate.name] = constant_tuples
        
        index_groundings = {}
        for name,rule in rule_groundings.items():
            if len(rule.groundings) > 0:
                G_body = []
                G_head = []
                for g in rule.groundings:
                    G_body.append([atom_to_index[atom] for atom in g[1]])
                    G_head.append([atom_to_index[atom] for atom in g[0]])
                index_groundings[name] = G_body, G_head

        index_queries = [[atom_to_index[q] for q in Q] for Q in queries]

        return (
            # domain->[global_idx] where i-th element is the global index of the
            # i-th local constant. e.g. this maps a local index into a global
            # one.
            domain_to_global,
            # These are theatoms expressd in form of local indices:
            # predicate->[(constant_local_idx)]
            predicate_to_constant_tuples,
            # rule -> [atom_local_idx_for_body, atom_local_idx_for_head]
            index_groundings,
            # [atom_local_indices_for_query]
            index_queries)
    

    def serialize_global_A_predicates(self, fol, queries:List[List[Tuple]],
                  rule_groundings:Dict[str, RuleGroundings]):
        domain_to_global = defaultdict(list)  # X_domains
        # domain_to_local_constant_index = defaultdict(dict)  # helper
        predicate_to_constant_tuples = defaultdict(list)  # A_predicates


        # Set of all atoms in the groundings to index
        all_atoms = ns.utils.to_flat(queries)
        for rg in rule_groundings.values():
            for g in rg.groundings:
                all_atoms += g[0] # head
                all_atoms += g[1] # body
        all_atoms = sorted(list(set(all_atoms)))

        # Bucket them per predicate
        all_atoms_per_predicate = {predicate.name: []
                                   for predicate in self.predicates}
        for atom in all_atoms:
            all_atoms_per_predicate[atom[0]].append(atom)

        # print('all_atoms_per_predicate')
        # for predicate in self.predicates:
        #     print('\nPredicate',predicate,all_atoms_per_predicate[predicate.name])

        # Create the index following the bucketed order:
        # A loop iterates over each predicate.
        # For each atom in the predicate, an index is assigned in atom_to_index.
        # Each constant in the atom is assigned an index relative to its domain.
        # A list of constant indices is generated for each atom.
        atom_to_index = {}
        count = 0
        for predicate in self.predicates:
            # print('predicate:',predicate.name)
            constant_tuples = [] # For each predicate, a list of constant indices is generated for each atom, e.g. LocIn:[[1,34],[2,3],...]
            for atom in all_atoms_per_predicate[predicate.name]: #For every atom in the queries
                atom_to_index[atom] = count  # HERE IT ASSIGNS AN INDEX TO EVERY ATOM (FOR EVERY PREDICATE), e.g. {LocIn(morocco,spain):0,LocIn(morocco,france):1,...}
                count += 1
                indices_cs = [] # This is for A_predicates, to get the constant indices in a list format, e.g.  (Morocco,Spain)->[1,34] (from the LocIn(morocco,spain) example)
                for c in atom[1:]: # for every constant in the atom
                    # check if that constant has a domain (in this case should be ctes)
                    domain = (self.constant2domain_name[c]
                              if c in self.constant2domain_name else
                              self.adaptive_constant2domain[c])
                    # print('     domain:',domain, c)
                    # constant_index = domain_to_local_constant_index[domain] # It is a domain_to_global_index but for the local indices
                    # if c not in constant_index:  # If the constant is not in the  local indices for constants mentioned before
                    assert c in self.constant_to_global_unique_index[domain], 'Constant not indexed'
                    if self.constant_to_global_unique_index[domain][c] not in domain_to_global[domain]:
                        # constant_index[c] = len(constant_index) # Add it to domain_to_local_constant_index
                        domain_to_global[domain].append(
                            self.constant_to_global_unique_index[domain][c]) # Append it to domain_to_global, which is what I return as X_domains, which is the different constants for each domain
                    # indices_cs.append(constant_index[c]) # Append the local index of the constant to the list of constant indices. INSTEAD, I SHOULD APPEND THE GLOBAL INDEX
                    indices_cs.append(self.constant_to_global_unique_index[domain][c]) # Append the global index of the constant to the list of constant indices
                # print('     indices:',indices_cs, atom[1:])
                constant_tuples.append(indices_cs) # Append the list of constant local indices to the list of constant indices for the predicate
            predicate_to_constant_tuples[predicate.name] = constant_tuples

        index_queries = [[atom_to_index[q] for q in Q] for Q in queries]
        index_groundings = {}
        for name,rule in rule_groundings.items():
            if len(rule.groundings) > 0:
                G_body = []
                G_head = []
                for g in rule.groundings:
                    G_body.append([atom_to_index[atom] for atom in g[1]])
                    G_head.append([atom_to_index[atom] for atom in g[0]])
                index_groundings[name] = G_body, G_head

        # Create indices for queries triples, in which, for each query, the index of the h,t,r is stored.
        # I need to do it here because otherwise ultra only gets the local atom index, and I need a global triplet index (which I cannot recover from the local atom index)
        # but also, at the begginging, when I initialize the dataset, I need to compute the triplet indeces for all the queries
        # I need to be careful with the domains. In regions, 1 is Europe, but in countries 1 is Spain. For Ultra, they will be the same
        queries_global = []
        for query in queries:
            triplet_index_query = []
            for atom in query:
                index_query = []
                for c in atom[1:]:
                    domain = (self.constant2domain_name[c]
                                if c in self.constant2domain_name else
                                self.adaptive_constant2domain[c])
                    index_query.append(self.constant_to_global_index[domain][c])
                # Now get the index of the predicate
                predicate_idx = self.predicate_to_global_index[atom[0]]
                index_query.append(predicate_idx)
                # index_query.append(int(predicate_idx))
                triplet_index_query.append(index_query)
            queries_global.append(triplet_index_query)

        # Create indices for A_predicates triples, in which, for each atom, the index of the h,t,r is stored
        A_predicates_global = []
        A_predicates_global_textualized = []
        for predicate in predicate_to_constant_tuples.keys():
            # Now get the index of the predicate
            predicate_idx = self.predicate_to_global_index[predicate]
            domains = self.predicate_to_domains[predicate]
            domain_head = domains[0]
            domain_tail = domains[1]
            true_flag = True
            for atom in predicate_to_constant_tuples[predicate]:
                A_predicates_global.append(atom + [predicate_idx]) 
                head_position = list(self.constant_to_global_unique_index[domain_head].values()).index(atom[0])
                tail_position = list(self.constant_to_global_unique_index[domain_tail].values()).index(atom[1])
                head_text = list(self.constant_to_global_unique_index[domain_head].keys())[head_position]
                tail_text = list(self.constant_to_global_unique_index[domain_tail].keys())[tail_position]
                if true_flag:
                    A_predicates_global_textualized.append(f"{head_text} is a {domain_head} and it is {predicate} of {tail_text}, which is a {domain_tail}")
                    true_flag = False
                    continue
                A_predicates_global_textualized.append(f"{head_text} is a {domain_head} and it is not {predicate} of {tail_text}, which is a {domain_tail}")
        # print('constant_to_global_unique_index')
        # for domain in self.constant_to_global_unique_index.keys():
        #     print(domain,self.constant_to_global_unique_index[domain])   

        # print('\nSERIALIZER X_DOMAIN KEYS: ,',domain_to_global.keys())
        # for domain in domain_to_global.keys(): # order the constants in the domain
        #     domain_to_global[domain] = sorted(domain_to_global[domain])
        #     tf.print('SERIALIZER X_domain:',domain,domain_to_global[domain])
        # print('SERIALIZER Predicates:',self.predicate_to_global_index.keys())
        # for p,constant_idx in predicate_to_constant_tuples.items():
        #     predicate = fol.name2predicate[p]
            # tf.print('SERIALIZER A_predicate',p,'domains:',[domain.name for domain in predicate.domains],'cte_index',constant_idx)
            # for i,domain in enumerate(predicate.domains):
            #     if len(constant_idx) != 0:
            #         tf.print('SERIALIZER A_predicate:',p,'domain:',domain.name,np.array(constant_idx)[:,i])
            #     else:
            #         tf.print('SERIALIZER A_predicate:',p,'domain:',domain.name, 'empty')

        return (domain_to_global,
                predicate_to_constant_tuples,
                index_groundings,
                index_queries,(queries_global,A_predicates_global,A_predicates_global_textualized))


#################################################
class LogicSerializerDebug(IndexerBase):

    def __init__(self, predicates: List[Predicate], domains: List[Domain]):

        self._finalized = False

        self.predicate_to_index = OrderedDict()
        self.index_to_predicate = OrderedDict()
        self.predicate_to_domains = {}
        self._index_predicates(predicates)

        self.constant_to_global_index = defaultdict(OrderedDict)
        self.global_index_to_constant = defaultdict(OrderedDict)
        self._index_domains_globally(domains)

        self.reset_indices()



    """-----------GLOBAL INDEXING CONSTANTS-------------"""

    def _index_constant_globally(self, constant: str, domain: Domain):
        domain = domain.name
        if constant not in self.constant_to_global_index[domain]:
            self.check_finalized('constant', constant)
            index = len(self.constant_to_global_index[domain])
            self.constant_to_global_index[domain][constant] = index
            self.global_index_to_constant[domain][index] = constant

    def _index_domains_globally(self, domains: List[Domain]):

        for domain in domains:
            for c in domain.constants:
                self._index_constant_globally(c, domain)

    # def get_constant_index(self, constant:str, domain:str = 'default_domain'):
    #     assert isinstance(constant,str)
    #     return self.constant_to_index[domain][constant]
    #
    # def get_constant_str(self, index, domain:str = 'default_domain'):
    #     if index not in self.index_to_constant[domain]:
    #         raise ValueError("Index is not mapped to any constant.")
    #     return self.index_to_constant[domain][index]
    #
    # def serialize_all_constants_indices(self):
    #     return {domain: len(constants) for domain, constants in self.constant_to_index.items()}

    """-----------DYNAMIC INDEXING START HERE-------------"""
    """This is used to filter the dataset into subsets of only those constants used in the batch of atoms """

    """ ------------------------_PREDICATES ------------------------------------"""

    def _index_predicate(self, predicate: Predicate):
        predicate,domains = predicate.name, predicate.domains
        if predicate not in self.predicate_to_index:
            self.check_finalized('predicate', predicate)
            index = len(self.predicate_to_index)
            self.predicate_to_index[predicate] = index
            self.index_to_predicate[index] = predicate
            self.predicate_to_domains[predicate] = [domain.name for domain in domains]
            return index
        else:
            raise Exception("Predicate %s already indexed" % predicate)

    def _index_predicates(self, predicates: List[Predicate]):
        res = []
        for predicate in predicates:
            res.append(self._index_predicate(predicate))
        return

    def get_predicate_index(self, predicate: Predicate):
        assert isinstance(predicate, Predicate)
        predicate_id = predicate.name
        return self.predicate_to_index[predicate]

    def get_predicate_domains(self, predicate: Union[PredicateIndex, PredicateName]):
        if not isinstance(predicate, str):
            predicate = self.index_to_predicate[predicate]
        return self.predicate_to_domains[predicate]

    def get_predicate_str(self, index: PredicateIndex):
        if index not in self.index_to_predicate:
            raise ValueError("Index is not mapped to any predicate.")
        return self.index_to_predicate[index]

    def reset_indices(self):

        self.constant_to_local_index = defaultdict(OrderedDict)
        self.local_index_to_constant = defaultdict(OrderedDict)
        self.atom_to_index = OrderedDict()
        self.index_to_atom = OrderedDict()
        self.atom_index_to_tuples = OrderedDict()
        self.seen_atoms = set()

        self.predicate_to_constant_tuples = OrderedDict()
        self.predicate_to_atom_ids = OrderedDict()
        self.predicate_to_atom_string = OrderedDict()
        for predicate in self.predicate_to_index.keys():
            self.predicate_to_atom_string[predicate] = []
            self.predicate_to_constant_tuples[predicate] = []
            self.predicate_to_atom_ids[predicate] = []

        self.local_to_global_per_domain = OrderedDict()
        for domain in self.global_index_to_constant.keys():
            self.local_to_global_per_domain[domain] = []



    def reindex(self, atoms:List[Tuple], formulas:List[RuleGroundings]):
        self._finalized = False
        self.reset_indices()
        self.index_atoms(atoms)
        self.index_formulas(formulas)
        self._finalize()

    """---------------------------- ATOMS ------------------------------"""

    def _index_atom(self, atom: Tuple):
        if atom not in self.seen_atoms:
            self.seen_atoms.add(atom)
            # Global indexing for atoms
            self.check_finalized('atom', atom)

            # Now we link atoms to their signature:
            #      [predicate, local_constant_id_0_in_domain_0, local_constant_id_1_in_domain_1, ...]

            # l = [atom.r] + atom.args
            l = list(atom)
            # l = [atom[0],atom[1],atom[2]]

            indices_cs = []

            if l[0] not in self.predicate_to_index:
                raise Exception("Predicate %s has not been indexed yet for atom %s" % (l[0], atom))
            else:
                domains = self.predicate_to_domains[l[0]]

            assert len(domains) == len(l) - 1, "Atom %s arity does not correspond to predicate " \
                                               "%s arity (%d)" % (atom, l[0], len(domains))
            for i, c in enumerate(l[1:]):
                if c not in self.constant_to_global_index[domains[i]]:
                    raise Exception("Constant %s in atom %s is unknown for domain %s. You should provide all "
                                    "the constants during the creation of the serializer" % (c, atom, domains[i]))
                else:
                    if c not in self.constant_to_local_index[domains[i]]:
                        index = len(self.constant_to_local_index[domains[i]])
                        self.constant_to_local_index[domains[i]][c] = index
                        self.local_index_to_constant[domains[i]][index] = c
                        self.local_to_global_per_domain[domains[i]].append(
                            self.constant_to_global_index[domains[i]][c])
                    indices_cs.append(self.constant_to_local_index[domains[i]][c])

            self.predicate_to_atom_string[l[0]].append(atom)
            self.predicate_to_constant_tuples[l[0]].append(indices_cs)

    def index_atoms(self, atoms: List[Tuple]):
        # start = timeit.default_timer()

        # atoms = list(atoms) # This fixes a bug that added multiple equal atoms.
        # TODO(giuseppe): If we foresee other uses of the index, check if it is better to have a set in the main and check for each atom

        for atom in atoms:
            if atom not in self.seen_atoms:
                self.seen_atoms.add(atom)
                # Global indexing for atoms
                # self.check_finalized('atom', atom)

                # Now we link atoms to their signature:
                #      [predicate, local_constant_id_0_in_domain_0, local_constant_id_1_in_domain_1, ...]

                # l = [atom.r] + atom.args
                l = atom
                # l = [atom[0],atom[1],atom[2]]

                indices_cs = []

                # if l[0] not in self.predicate_to_index:
                #     raise Exception("Predicate %s has not been indexed yet for atom %s" % (l[0], atom))
                # else:
                domains = self.predicate_to_domains[l[0]]

                # assert len(domains) == len(l) - 1, "Atom %s arity does not correspond to predicate " \
                #                                    "%s arity (%d)" % (atom, l[0], len(domains))
                for i, c in enumerate(l[1:]):
                #     if c not in self.constant_to_global_index[domains[i]]:
                #         raise Exception("Constant %s in atom %s is unknown for domain %s. You should provide all "
                #                         "the constants during the creation of the serializer" % (c, atom, domains[i]))
                #     else:
                        if c not in self.constant_to_local_index[domains[i]]:
                            index = len(self.constant_to_local_index[domains[i]])
                            self.constant_to_local_index[domains[i]][c] = index
                            self.local_index_to_constant[domains[i]][index] = c
                            self.local_to_global_per_domain[domains[i]].append(
                                self.constant_to_global_index[domains[i]][c])
                        indices_cs.append(self.constant_to_local_index[domains[i]][c])

                self.predicate_to_atom_string[l[0]].append(atom)
                self.predicate_to_constant_tuples[l[0]].append(indices_cs)



            # self._index_atom(a)

        # stop = timeit.default_timer()
        # print('Time: ', stop - start)

    def get_constant_local_index(self, c, d):
        return self.constant_to_local_index[d][c]

    ### commented after converting everything into tuples
    def get_atom_tuple(self, atom: Tuple):
        # l = [atom.r] + atom.args
        l = list(atom)
        a = [self.get_predicate_index(l[0])]
        cs = [self.get_constant_local_index(c, self.get_predicate_domains(l[0])[i]) for i, c in enumerate(l[1:])]
        return a + cs

    def get_atom_index(self, atom: Tuple):
        return self.atom_to_index[atom]

    def get_atom_str(self, index):
        if index not in self.index_to_atom:
            raise ValueError("Index %d is not mapped to any atom." % index)
        return self.index_to_atom[index]

    def atoms_as_tuples(self):
        return [i for i in self.atom_index_to_tuples.values()]

    def atoms_as_dict_predicate_tuples(self):
        return self.predicate_to_constant_tuples

    def atoms_as_dict_predicate_ids(self):
        return self.predicate_to_atom_ids

    def _index_all_atoms(self):
        index = -1
        for p, v in self.predicate_to_constant_tuples.items():
            id_predicate = self.predicate_to_index[p]
            for j, cs in enumerate(v):
                index = index + 1
                atom = self.predicate_to_atom_string[p][j]
                self.predicate_to_atom_ids[p].append(index)
                self.atom_index_to_tuples[index] = [id_predicate] + cs
                self.atom_to_index[atom] = index
                self.index_to_atom[index] = atom

    def index_groundings(self, groundings, distinct_output=False):
        for g in groundings:
            for atom in g[0]:
                self._index_atom(atom)
            for atom in g[1]:
                self._index_atom(atom)



    def index_formulas(self, formulas: List[RuleGroundings]):
        self.check_finalized('formulas', formulas)
        for formula in formulas:
            self.index_groundings(formula.groundings)

    def serialize_atoms(self, atoms):
        if isinstance(atoms, Tuple):
            return self.get_atom_index(atoms)
        else:
            return [self.serialize_atoms(a) for a in atoms]

    def serialize_atoms_as_tuples(self, atoms):
        return [self.atom_index_to_tuples[self.get_atom_index(a)] for a in atoms]

    def serialize_groundings(self, groundings):
        G_body = [[self.get_atom_index(atom) for atom in g[1]]
                  for g in groundings]
        G_head =  [
                [self.get_atom_index(atom) for atom in g[0]]
                for g in groundings]
        return G_body, G_head

    def serialize_formulas(self, formulas: List[RuleGroundings]):
        return {ruleg.name: self.serialize_groundings(ruleg.groundings) for ruleg in formulas if len(ruleg.groundings) > 0}

    def _finalize(self):
        self._finalized = True
        self._index_all_atoms()

    def check_finalized(self, type_to_create, object_to_create):
        if self._finalized:
            raise Exception(
                "The serializer is finalized. I cannot index the %s: %s. You should reset the indices first." % (
                type_to_create, object_to_create))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            return False  # uncomment to pass exception through
        if not self._finalized:
            self._finalize()
        return True

    def serialize(self, queries:List[List[Tuple]],
                  rule_groundings:Dict[str, RuleGroundings]):

        ground_formulas = list(rule_groundings.values())
        self.reindex(atoms=ns.utils.to_flat(queries),
                           formulas=ground_formulas)


        domain_to_global = self.local_to_global_per_domain
        predicate_tuples = self.atoms_as_dict_predicate_tuples()
        formulas = self.serialize_formulas(ground_formulas)
        queries = [[self.get_atom_index(q) for q in Q] for Q in queries]

        return domain_to_global, predicate_tuples, formulas, queries


if __name__ == '__main__':
    rule = ["p(X,Y) -> p(Y,X)"]

    d = Domain("domain", ["a", "b"])
    p = Predicate("p", [d,d])


    a1 = ("p", "a", "b")
    a2 = ("p", "b", "a")

    queries = [[a1, a2]]


    rule_groundings = {'r1':
                       RuleGroundings('r1', [((a1,), (a2,)), ((a2,), (a1,))])}
    ls1 = LogicSerializerFast(predicates=[p], domains=[d])
    print(ls1.serialize(queries, rule_groundings))