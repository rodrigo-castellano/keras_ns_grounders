from typing import Dict, Iterable, List, Tuple, Union
import ns_lib as ns
from ns_lib.logic.commons import Atom, FOL, RuleGroundings
import tensorflow as tf
import numpy as np
import torch
import sys
import os
import torch

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir, '..', 'experiments'))

import ultra_utils
from ultra_utils import Ultra as Ultra_modified
from ULTRA.ultra.models import Ultra
from ULTRA.ultra import tasks
import itertools
from ULTRA.ultra.tasks import build_relation_graph
from ns_lib.nn.constant_embedding import LMEmbeddings

DomainName = str
ConstantName = str
PredicateName = str

FormulaSignature = str
Label = int
Tensor = Union[np.array, tf.Tensor]
#Queries = Union[List[Atom], List[List[Atom]]]
#Labels = Union[List[Label], List[List[Label]]]


IntId = np.array
AtomId = int
SingleIdFeature = Tensor
FloatFeatures = Tensor
ConstantFeatures = Union[SingleIdFeature, FloatFeatures]
ConstantTuples = Union[List[List[IntId]], Tensor]
AtomIds = Union[List[List[AtomId]], Tensor] # one atom for each tuple above
AtomTuples = Union[List[List[AtomId]], Tensor] # a list of groundings (lists of AtomId) for each formula
# QueriesIds = Union[List[AtomId], List[List[AtomId]], Tensor]


def _from_strings_to_tensors(fol, serializer,
                             queries, labels, engine, ragged,
                             constants_features=None, deterministic=True, global_serialization=False):

    # Symbolic step
    facts_tuple = tuple(fol.facts)
    queries_tuple = tuple(ns.utils.to_flat(queries))
    if engine is not None:
        ground_formulas: Dict[str, RuleGroundings] = engine.ground(
            facts_tuple, queries_tuple, deterministic=deterministic)
        rules = engine.rules
    else:
        ground_formulas = {}
        rules = []

    '''
    - Domain_to_global: set of global indices of all the ctes whithin each domain. Only the ones present in the batch. 
        Not necessarily in order, they are added as the appear in the ordered groundings.

    - Predicate_tuples: local indices of the ctes in the atoms with respect to each predicate. For each predicate, all groundings of that predicate are present
        For every predicate, indices start from 0, as they appear in the ordered groundings (not neccessarily in strict order).
        The local indices are created every batch in 'domain_to_local_constant_index', starting by 0.

    - Groundings      : local indices that  represent the position of atoms within the respective rule groundings. First index is the rule, e.g. ['r0'], the the first element 
      [0] is a set of bodies and the second element [1] is a set of heads. The body has 2 elements, each representing one atom. The head has 1 element, representing one atom.
      the local indices are the ones assigned to the ordered atoms of the batch.

    - Queries         : local indices that represent the position of atoms within each query. len(queries) is the number of queries, each query is a list of atoms. 
      The first index (atom) represents the original query, the rest of the indeces (atoms) represent the corruptions of that query. 
      The local indices are the ones assigned to the ordered atoms of the batch.

    - Queries_global  : global indices of the queries. The first index is the original query, the rest are the corruptions of that query. It i as queries, but with (h_id, t_id, r_id)
    '''
    A_predicates_global = queries_global = A_predicates_global_textualized = None
    if global_serialization:
        (domain_to_global, predicate_tuples, groundings, queries, (queries_global,A_predicates_global,A_predicates_global_textualized)) = (
            serializer.serialize_global_A_predicates(fol,queries=queries,rule_groundings=ground_formulas))
    else:
        (domain_to_global, predicate_tuples, groundings, queries) = (serializer.serialize(queries=queries,
                                    rule_groundings=ground_formulas))   
    


    # Convert constants(domain) indices from list to tf tensor
    input_domains_tf: Dict[DomainName, ConstantFeatures] = {}
    for d in fol.domains:
        if constants_features is not None and d.name in constants_features:
            # If available, global features for constants are gathered based on their global indices within their domains.
            global_features = constants_features[d.name]
            global_indices = domain_to_global[d.name]
            input_features = tf.gather(global_features, global_indices, axis=0)
            input_domains_tf[d.name] = input_features
        else:
            input_domains_tf[d.name] = tf.constant(domain_to_global[d.name],
                                                   dtype=tf.int32)
    # Creating the input dictionaries (atoms as tuples of domains,
    # atoms as dense ids, formulas as tuples of atoms)
    # Convert ctes indices with respect to predicates from list to tf tensor. (num_predicates, number_of_groundings, arity_of_predicate)
    # Dict[predicate_name, List[Tuple[constants_ids]]]
    input_atoms_tuples_tf: Dict[PredicateName, ConstantTuples] = {
        name:tf.constant(tuples, dtype=tf.int32) if len(tuples) > 0 else
             tf.zeros(shape=(0, fol.name2predicate[name].arity), dtype=tf.int32)
        for name,tuples in predicate_tuples.items()}

    # Same here, but for the groundings of the rules. (num_rules, 2, num_atoms (in body/head), arity_of_predicate)
    # Dict[formula_id, List[Tuple[atom_ids]]]
    input_formulas_tf: Dict[FormulaSignature, (AtomTuples, AtomTuples)] = {}
    for rule in rules:
        ai = len(rule.body)
        ao = len(rule.head)
        if rule.name in groundings and len(groundings[rule.name]) > 0:
            # adding batch dimension
            input_formulas_tf[rule.name] = (
                tf.constant(groundings[rule.name][0], dtype=tf.int32),
                tf.constant(groundings[rule.name][1], dtype=tf.int32))
        else:
            # empty tensor
            input_formulas_tf[rule.name] = (
                tf.zeros(shape=[0, ai], dtype=tf.int32),
                tf.zeros(shape=[0, ao], dtype=tf.int32))
            
    # TODO check how to understand if it is a good tensor or need to be ragged.
    if ragged:
        queries = tf.ragged.constant(queries, dtype=tf.int32)
        labels =  tf.ragged.constant(labels, dtype=tf.float32)
    else:
        queries = tf.constant(queries, dtype=tf.int32)
        labels = tf.constant(labels, dtype=tf.float32)
    return (input_domains_tf, input_atoms_tuples_tf,
            input_formulas_tf, queries, (queries_global,A_predicates_global,A_predicates_global_textualized)), labels


class Dataset():

    def __init__(self,
                 queries: List[List[Union[Atom, str, Tuple]]],
                 labels: List[List[Label]],
                 constants_features: Dict[DomainName, Tensor]=None,
                 format: str='functional'):

        self.constants_features = constants_features
        self.queries = [[a if isinstance(a,Tuple) else a.toTuple() if isinstance(a,Atom) else Atom(s=a, format=format).toTuple() for a in q] for q in queries]
        self.labels = labels
        assert len(queries) == len(labels)
        self.format = format


    def __getitem__(self, item):
        queries,labels =  self.queries[item], self.labels[item]
        return queries, labels


    def __len__(self):
        return len(self.queries)


    def _get_batch(self, i, b):

        return self.queries[b*i:b*(i+1)], self.labels[b*i:b*(i+1)], 



class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 dataset: Dataset,
                 fol: FOL,
                 serializer,
                 engine=None,
                 deterministic=True,
                 batch_size=None,
                 ragged: bool=False,
                 name= "None",
                 use_ultra = False,
                 use_ultra_with_kge = False,
                 use_llm = False,
                 global_serialization = False,
                 dataset_ultra = None):
        
        self.global_serialization = global_serialization
        self.dataset = dataset
        self.deterministic = deterministic
        self.fol = fol
        self.engine = engine
        self.serializer = serializer
        self.ragged = ragged
        self._batch_size = (batch_size
                            if batch_size is not None and batch_size > 0
                            else -1)
        self.name = name
        self.use_ultra = use_ultra
        self.use_ultra_with_kge = use_ultra_with_kge
        self.use_llm = use_llm

        if self.use_ultra or self.use_ultra_with_kge:
            assert dataset_ultra is not None, 'You need to provide the aux dataset_ultra'
            self.aux_dataset = dataset_ultra
            self.Ultra = ultra_utils.load_ultra_model(Ultra, original=True)
            self.Ultra_modified = ultra_utils.load_ultra_model(Ultra_modified, original=False)

        if self.use_llm:
            self.llm_embedder = LMEmbeddings(self.fol, "")

        self._num_batches = 1
        if self._batch_size > 0:
            if len(self.dataset) % self._batch_size == 0:
                self._num_batches = len(self.dataset) // self._batch_size
            else:
                self._num_batches = len(self.dataset) // self._batch_size + 1

        if self._num_batches == 1:
            print('Building Full Batch Dataset', self.name)
            self._full_batch = self._get_batch(0, len(self.dataset))


    def __getitem__(self, item):
        if self._num_batches == 1:
            return self._full_batch
        else:
            return self._get_batch(item, self._batch_size)


    def __len__(self):
        return self._num_batches


    def _get_batch(self, i, b):

        queries, labels = self.dataset[b*i:b*(i+1)]
        constants_features = self.dataset.constants_features
        ((X_domains_data, A_predicates_data, A_rules_data, Q, (Q_global,A_predicates_triplets,A_predicates_textualized)), y) = _from_strings_to_tensors(
            fol=self.fol,
            serializer=self.serializer,
            queries=queries,
            labels=labels,
            engine=self.engine,
            ragged=self.ragged,
            constants_features=constants_features,
            deterministic=self.deterministic,
            global_serialization=self.global_serialization) 

        embeddings = None

        if self.use_ultra_with_kge:  
            constant_embeddings, predicate_embeddings = self.Ultra(self.aux_dataset, Q_global,atom_repr=False) # embedds of the ctes,preds
            for key in constant_embeddings: # Convert embeddings to tf
                constant_embeddings[key] = tf.constant(constant_embeddings[key])
            embeddings = (constant_embeddings, tf.constant(predicate_embeddings, dtype=tf.float32))

        elif self.use_ultra: 
            self.Ultra.eval()
            self.Ultra_modified.eval()
            ultra_utils.mimic_ultra(self.aux_dataset, Ultra)
            # ultra_utils.mimic_test_function_ultra(self.aux_dataset,Q_global,self.Ultra)
            # scores,_ = ultra_utils.mimic_test_function_ultra_with_our_corruptions(self.aux_dataset, Q_global, self.Ultra)
            # scores, y = ultra_utils.get_ultra_outputs(self.aux_dataset,Q_global, self.Ultra_modified)
            # scores, atom_embeds = ultra_utils.get_ultra_outputs_nonfiltered_negatives(self.aux_dataset,Q_global, self.Ultra) # For train, instead of 4 negatives, I have all the negatives
            # scores, atom_embeds = self.get_ultra_embeddings(A_predicates_triplets,Q_global) # this is the official one
            # embeddings = (scores, atom_embeds)

        elif self.use_llm:
            constant_embeddings, predicate_embeddings = self.llm_embedder(A_predicates_triplets)
            embeddings = (constant_embeddings, predicate_embeddings)


        return (X_domains_data, A_predicates_data, A_rules_data, Q, embeddings), y



    def get_ultra_embeddings(self,A_pred,queries):
        '''
        Do the preprocessing of the data to give it to ultra. Take a modified ultra to get the embeddings of the atoms of A_pred.
        Return the embeddings of the atoms and the scores of the atoms of A_pred.
        # Option 1: get the embeddings of the atoms in A_predicates_data. But then I would need to calculate the negatives of the atoms in A_predicates_data
        # Option 2: for every atom in Q_global, calculate the embeddings of the atom and the negatives
        '''

        '''Process: use A_pred_global, get negatives for each triplet. Give it as input to ultra, and get the embeddings and the scores of the atoms.'''

        # Convert A_pred to a format that ultra can understand. For every atom in A_pred, create a list of the atom and the negatives
        batch = torch.tensor(A_pred, dtype=torch.int64)
        # Get the negatives of the atoms in A_pred. Take into account to filter atoms from edge_index, edge_type, target_edge_index, target_edge_type. 
        t_batch, h_batch = tasks.all_negative(self.aux_dataset, batch)
        # Get the embeddings of the atoms in A_pred and the negatives
        t_pred_scores, t_pred_embedd = self.Ultra_modified(self.aux_dataset, t_batch)
        h_pred_scores, h_pred_embedd = self.Ultra_modified(self.aux_dataset, h_batch)
        
        # Now, for the embeddings, I can return them in different ways. It is better to obtain the scores from the embeddings in our mode. 

        # # i) Do the average of the embeddings of all the entities. This is the most general way, but it is not the best way to represent the entities
        # t_pred_embedd = torch.mean(t_pred_embedd, dim=1)
        # h_pred_embedd = torch.mean(h_pred_embedd, dim=1)

        # ii) Take the embeddings of the entitiy that is in the positive query.
        # In t corruptions, the original index is the tail of the first element of the list, in h corruptions, the original index is the head of the first element of the list  
        t_index = batch[:,1]
        h_index = batch[:,0]
        # in the embeddings, the representation given by the index, in the first dimension, is the original representation of the entity
        t_embedd = torch.zeros(t_index.shape[0], t_pred_embedd.shape[2])
        h_embedd = torch.zeros(h_index.shape[0], h_pred_embedd.shape[2])
        for i in range(len(t_index)):
            t_embedd[i] = t_pred_embedd[i][t_index[i]]
            h_embedd[i] = h_pred_embedd[i][h_index[i]]

        # convert the scores and embeddings to tf
        t_pred_scores_tf = tf.squeeze(tf.constant(t_pred_scores.detach().numpy(), dtype=tf.float32))
        t_embedd = tf.squeeze(tf.constant(t_embedd.detach().numpy(), dtype=tf.float32))
        h_pred_scores_tf = tf.squeeze(tf.constant(h_pred_scores.detach().numpy(), dtype=tf.float32))
        h_embedd = tf.squeeze(tf.constant(h_embedd.detach().numpy(), dtype=tf.float32))



        # To calculate the metrics, gather the indices of the triplets of A_pred that are in the queries
        queries_positive = [q[0] for q in queries]
        indices = [i for i in range(len(A_pred)) if A_pred[i] in queries_positive]

        t_pred_scores_queries = t_pred_scores.detach().numpy()
        t_pred_scores_queries = t_pred_scores_queries[indices]

        scores = np.squeeze(t_pred_scores.detach().numpy())
        scores = scores[indices]
        labels_new = np.zeros(tuple(scores.shape))
        labels_new[:,0] = 1
        labels_tf = tf.constant(labels_new, dtype=tf.float32)
        # do a copy of the scores, because the function calculated_metrics_batched modifies the scores
        t_pred_scores_copy = tf.squeeze(tf.constant(t_pred_scores_queries))
        ultra_utils.calculated_metrics_batched(t_pred_scores_copy, labels_tf)

        return scores_tf, t_embedd






class DataGeneratorTensor(tf.keras.utils.Sequence):

    def __init__(self,
                 dataset: Dataset,
                 serializer,
                 engine=None,
                 batch_size=None,
                 ragged: bool=False):

        self.dataset = dataset
        self.engine = engine
        self.serializer = serializer
        self.ragged = ragged
        self._batch_size = (batch_size
                            if batch_size is not None and batch_size > 0
                            else -1)

        self._num_batches = 1
        if self._batch_size > 0:
            if len(self.dataset) % self._batch_size == 0:
                self._num_batches = len(self.dataset) // self._batch_size
            else:
                self._num_batches = len(self.dataset) // self._batch_size + 1

        if self._num_batches == 1:
            print('Building Full Batch Dataset', self.name)
            self._full_batch = self._get_batch(0, len(self.dataset))

    def __getitem__(self, item):
        if self._num_batches == 1:
            return self._full_batch
        else:
            return self._get_batch(item, self._batch_size)


    def __len__(self):
        return self._num_batches


    def _get_batch(self, i, b):

        queries, labels = self.dataset[b*i:b*(i+1)]

        lenghts = [len(x) for x in queries]
        flat_queries = tf.constant(ns.utils.to_flat(queries), dtype=tf.int32)
        if self.engine is not None:
            ground_formulas = self.engine.ground(
                tf.constant(self.dataset.facts, dtype=tf.int32),
                flat_queries)
            rules = self.engine.rules
        else:
            ground_formulas = {}
            rules = []

        (input_domains_tf, input_atoms_tuples_tf,
         input_formulas_tf, flat_queries) = self.serializer.serialize(
            queries=flat_queries, rule_groundings=ground_formulas)

        if self.dataset.constants_features is not None:
            for d in self.dataset.domains:
                global_features = self.dataset.constants_features[d.name]
                local_indices = input_domains_tf[d.name]
                global_to_input = tf.gather(global_features, local_indices, axis=0)
                input_domains_tf[d.name] = global_to_input

        if self.ragged:
            queries = tf.RaggedTensor.from_row_lengths(values = flat_queries, row_lengths = lenghts)
            labels = tf.ragged.constant(labels, dtype=tf.float32)
        else:
            queries = tf.constant(queries, dtype=tf.int32)
            labels = tf.constant(labels, dtype=tf.float32)

        return (input_domains_tf, input_atoms_tuples_tf,
                input_formulas_tf, queries), labels


class DataGeneratorTensorFast(tf.keras.utils.Sequence):

    def __init__(self,
                 dataset: Dataset,
                 batch_size=None,
                 ragged: bool=False):

        self.dataset = dataset

        self.ragged = ragged
        self._batch_size = (batch_size
                            if batch_size is not None and batch_size > 0
                            else -1)

        self._num_batches = 1
        if self._batch_size > 0:
            if len(self.dataset) % self._batch_size == 0:
                self._num_batches = len(self.dataset) // self._batch_size
            else:
                self._num_batches = len(self.dataset) // self._batch_size + 1

        if self._num_batches == 1:
            print('Building Full Batch Dataset', self.name)
            self._full_batch = self._get_batch(0, len(self.dataset))

    def __getitem__(self, item):
        if self._num_batches == 1:
            return self._full_batch
        else:
            return self._get_batch(item, self._batch_size)


    def __len__(self):
        return self._num_batches


    def _get_batch(self, i, b):

        q,l =  self.dataset[b*i:b*(i+1)]

        return tf.ragged.constant(q), tf.ragged.constant(l)

 

def obtain_queries(dataset,data_handler,serializer,engine,ragged,deterministic,global_serialization):
    queries, labels = dataset[:]
    constants_features = dataset.constants_features
    fol = data_handler.fol

    ((X_domains_data, A_predicates_data, A_rules_data, Q, (Q_global,A_predicates_triplets,A_predicates_textualized)), y) = _from_strings_to_tensors(
        fol=fol,
        serializer=serializer,
        queries=queries,
        labels=labels,
        engine=engine,
        ragged=ragged,
        constants_features=constants_features,
        deterministic=deterministic,
        global_serialization=global_serialization) 
    Q_global_positive = [q[0] for q in Q_global]
    # print('\nqueries positive', len(queries), [query[0] for query in queries][:20])
    # print('Q_global_positive', len(Q_global_positive), Q_global_positive[:20])
    return X_domains_data, A_predicates_data, Q_global_positive

def get_ultra_datasets(dataset_train, dataset_valid, dataset_test,data_handler,serializer,engine,ragged=True,deterministic=True,global_serialization=False):

    # Get the triplets
    X_domain_train, A_pred_train, train_triplets = obtain_queries(dataset_train,data_handler,serializer,engine,ragged,deterministic,global_serialization)
    X_domain_valid, A_pred_valid, valid_triplets = obtain_queries(dataset_valid,data_handler,serializer,engine,ragged,deterministic,global_serialization)
    X_domain_test, A_pred_test, test_triplets = obtain_queries(dataset_test,data_handler,serializer,engine,ragged,deterministic,global_serialization)

    def unique_ordered(triplets):
        return list(dict.fromkeys(tuple(t) for t in triplets))

    train_triplets = unique_ordered(train_triplets)
    valid_triplets = unique_ordered(valid_triplets)
    test_triplets = unique_ordered(test_triplets)

    # get the number of nodes and relations for the train,val,test set. Do it by getting unique the ones in train, val, test
    train_nodes = [X_domain_train[key].numpy().tolist() for key in X_domain_train]
    valid_nodes = [X_domain_valid[key].numpy().tolist() for key in X_domain_valid]
    test_nodes = [X_domain_test[key].numpy().tolist() for key in X_domain_test]

    # Flatten the lists
    train_nodes = set(itertools.chain(*train_nodes))
    valid_nodes = set(itertools.chain(*valid_nodes))
    test_nodes = set(itertools.chain(*test_nodes))
    num_node = len(train_nodes.union(valid_nodes).union(test_nodes) )

    # do the same for the relations
    train_relations = [key for key in A_pred_train]
    valid_relations = [key for key in A_pred_valid]
    test_relations = [key for key in A_pred_test]
    # take the unique number of relations
    unique_relations = list(set(train_relations+valid_relations+test_relations))
    num_relations_no_inv = torch.tensor(len(unique_relations))
    # num_relations_no_inv = len(data_handler.fol.predicates)

    train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
    train_target_etypes = torch.tensor([t[2] for t in train_triplets])
    train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
    train_etypes = torch.cat([train_target_etypes, train_target_etypes+num_relations_no_inv])

    # valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
    valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
    valid_etypes = torch.tensor([t[2] for t in valid_triplets])

    test_edges = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
    test_etypes = torch.tensor([t[2] for t in test_triplets])

    train_data = ultra_utils.Dataset_Ultra(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                        target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_relations_no_inv*2)
    train_data.num_edges = train_data.edge_index.shape[1]
    valid_data = ultra_utils.Dataset_Ultra(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                        target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations_no_inv*2)
    valid_data.num_edges = valid_data.edge_index.shape[1]
    test_data = ultra_utils.Dataset_Ultra(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                        target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations_no_inv*2)
    test_data.num_edges = test_data.edge_index.shape[1]

    # edge_index is the sum of all target_edge_index
    edge_index = torch.cat([train_target_edges, valid_edges, test_edges], dim=1)
    edge_type = torch.cat([train_target_etypes, valid_etypes, test_etypes])
    # num_nodes is given by the train set
    num_edges = None # is not defined in Ultra for the general dataset
    device = 'cpu'
    dataset = ultra_utils.Dataset_Ultra(edge_index, edge_type, num_relations_no_inv*2, num_node, num_edges, device)
    filtered_data = dataset 

    train_data = build_relation_graph(train_data)
    valid_data = build_relation_graph(valid_data)
    test_data = build_relation_graph(test_data)

    return train_data, valid_data, test_data, filtered_data