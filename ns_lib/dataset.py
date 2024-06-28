from typing import Dict, Iterable, List, Tuple, Union
import ns_lib as ns
from ns_lib.logic.commons import Atom, FOL, RuleGroundings
import tensorflow as tf
from keras.metrics import Metric
from keras.losses import Loss
import abc
import numpy as np
import torch
from utils import MRRMetric, HitsMetric
import sys
import os

import itertools
import torch
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir, '..', 'experiments'))

from ultra_utils import Ultra as Ultra_modified
from ULTRA.ultra.models import Ultra
from ULTRA.ultra import tasks
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


def split_by_corruptions(batch):
    '''
    Given a batch, split it in subbatches with the same number of corruptions in each subbatch. In this way it is easy to convert it to a pytorch version.
    '''
    n_queries = len(batch)
    subbatches = {}
    for i in range(n_queries):
        n_corruptions = len(batch[i])
        if n_corruptions not in subbatches:
            subbatches[n_corruptions] = []
        subbatches[n_corruptions].append([batch[i]])
    return subbatches

def split_positives_negatives(batch):
    '''
    Given elements in a batch, for every element, if it has head and tail corruption, split it in two elements, one for head and one for tail corruption. 
    Reference: in each element, the first query is the positive and the rest are the negatives. 
    batch: list of lists of queries. Each query is a list of triplets.
    A faster way to implement this function is to use if before entity_representations, because there I work with the tensors, and here I work with the lists.
    '''
    new_batch = []
    for i in range(len(batch)):
        sample = batch[i]
        if len(sample) > 1:
            negs = sample[1:]
            pos_head = sample[0][0]
            pos_tail = sample[0][1]
            # take all the heads of the positives and the tails of the negatives
            heads = [neg[0] for neg in negs]
            tails = [neg[1] for neg in negs]
            # if all the heads are the same, it means that the corruptions are only in the tail
            # if all the tails are the same, it means that the corruptions are only in the head
            if all(head == pos_head for head in heads) or all(tail == pos_tail for tail in tails):
                new_batch.append(sample)
            else:
                head_corruptions, tail_corruptions = [], []
                for neg in negs:
                    if neg[1] == pos_tail:
                        head_corruptions.append(neg)
                    else:
                        tail_corruptions.append(neg)
                if head_corruptions:
                    new_batch.append([sample[0]] + head_corruptions)
                if tail_corruptions:
                    new_batch.append([sample[0]] + tail_corruptions)
        else:   
            new_batch.append(sample)
    return new_batch


class Dataset_Ultra():
    
        def __init__(self, edge_index=None, edge_type=None, num_relations=None, num_nodes=None, num_edges=None, device=None, target_edge_index=None, target_edge_type=None):
            self.edge_index = edge_index
            self.edge_type = edge_type
            self.num_relations = num_relations
            self.num_nodes = num_nodes
            self.num_edges = num_edges
            self.device = device
            self.target_edge_index = target_edge_index
            self.target_edge_type = target_edge_type


def load_ultra_model(model, original=True):
    if original:
        rel_model_cfg = {"class": "RelNBFNet","input_dim": 64,"hidden_dims": [64, 64, 64, 64, 64, 64],"message_func": "distmult","aggregate_func": "sum","short_cut": "yes","layer_norm": "yes"} 
        entity_model_cfg= {"class": "EntityNBFNet","input_dim": 64,"hidden_dims": [64, 64, 64, 64, 64, 64],"message_func": "distmult","aggregate_func": "sum","short_cut": "yes","layer_norm": "yes"}
        model = model(rel_model_cfg, entity_model_cfg)
    else:
        model = model()

    # print('\n\nUltra', Ultra)
    # for name, param in Ultra.named_parameters():             
    #     print(name, param.shape)
    #     print(param[0][:5]) if len(param.shape) > 1 else print(param[:5])

    # 'C:\\Users\\rodri\\Downloads\\PhD_code\\Review_grounders\\keras_ns_grounders\\ULTRA\\ckpts\\ultra_4g.pth'
    state = torch.load('C:\\Users\\rodri\\Downloads\\PhD_code\\Review_grounders\\keras_ns_grounders\\ULTRA\\ckpts\\countries_ft_epoch_10.pth', map_location="cpu")
    state = state['model']

    # subsitution of the keys of the state dict. 'mlp.0' by 'mlp_embedd.0' and 'mlp.2' by 'mlp_score.2'
    # state = OrderedDict((key.replace('mlp', 'mlp_embedd'), value) if 'mlp.0' in key else (key.replace('mlp', 'mlp_score'), value) for key, value in state.items())
    # print('\n\nState', state.keys())
    # for key in state.keys():
    #     print(key, state[key].shape)
    #     print(state[key][0][:5]) if len(state[key].shape) > 1 else print(state[key][:5])

    model.load_state_dict(state, strict=False) # Filter out the keys that correspond to the final layer

    # # show the parameters of the model
    # print('\n\nUltra loaded')
    # for name, param in Ultra.named_parameters():             
    #     print(name, param.shape)
    #     print(param[0][:5]) if len(param.shape) > 1 else print(param[:5])

    model = model.to('cpu')
    return model

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
            self.Ultra = load_ultra_model(Ultra, original=True)
            self.Ultra_modified = load_ultra_model(Ultra_modified, original=False)

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
            use_ultra_original = False
            if not use_ultra_original:

                # scores,atom_embeds = self.Ultra(self.aux_dataset, A_predicates_triplets, atom_repr=True) # embedds of the atoms
                # scores = tf.constant(scores, dtype=tf.float32)
                # atom_embeds = tf.constant(atom_embeds, dtype=tf.float32)

                scores, atom_embeds = self.get_ultra_embeddings(A_predicates_triplets)
                embeddings = (scores, atom_embeds)
            else:
                self.Ultra.eval()
                # self.mimic_ultra()
                # self.mimic_test_function_ultra(Q_global_positive)
                # scores,_ = self.mimic_test_function_ultra_with_our_corruptions(Q_global)
                scores, labels = self.get_ultra_outputs(Q_global)
                y = labels
                # I should still substitute the concept embeddings by random matrix, and create random matrices for the reasoner
                embeddings  = (scores, None)

        elif self.use_llm:
            atom_embeds = self.llm_embedder(A_predicates_textualized)
            atom_embeds = tf.constant(atom_embeds, dtype=tf.float32)
            embeddings = (None, atom_embeds)
            # _,atom_embeds = self.Ultra(self.aux_dataset, A_predicates_triplets, atom_repr=True) # embedds of the atoms
            # atom_embeds = tf.constant(atom_embeds, dtype=tf.float32)
            # embeddings = (_, atom_embeds)

        return (X_domains_data, A_predicates_data, A_rules_data, Q, embeddings), y


    def get_ultra_embeddings(self,A_pred):
        '''
        Do the preprocessing of the data to give it to ultra. Take a modified ultra to get the embeddings of the atoms of A_pred.
        Return the embeddings of the atoms and the scores of the atoms of A_pred.
        # Option 1: get the embeddings of the atoms in A_predicates_data. But then I would need to calculate the negatives of the atoms in A_predicates_data
        # Option 2: for every atom in Q_global, calculate the embeddings of the atom and the negatives
        '''

        '''Process: use A_pred_global, get negatives for each triplet. Give it as input to ultra, and get the embeddings and the scores of the atoms.'''
        # Convert A_pred to a format that ultra can understand. For every atom in A_pred, create a list of the atom and the negatives

        # print('len A_pred:', len(A_pred))
        batch = torch.tensor(A_pred, dtype=torch.int64)
        # print('Batch:', batch.shape, batch[:15])
        # Get the negatives of the atoms in A_pred. Take into account to filter atoms from edge_index, edge_type, target_edge_index, target_edge_type. 
        t_batch, h_batch = tasks.all_negative(self.aux_dataset, batch)
        # print('t_batch:', t_batch.shape, 'h_batch:', h_batch.shape)
        # print('t_batch:', t_batch[:5])
        # print('h_batch:', h_batch[:5])
        # Get the embeddings of the atoms in A_pred and the negatives
        t_pred_scores, t_pred_embedd = self.Ultra_modified(self.aux_dataset, t_batch)
        h_pred_scores, h_pred_embedd = self.Ultra_modified(self.aux_dataset, h_batch)
        
        # Now, for the embeddings, I can return them in different ways. It is better to obtain the scores from the embeddings in our mode. 

        # # i) Do the average of the embeddings of all the entities. This is the most general way, but it is not the best way to represent the entities
        # t_pred_embedd = torch.mean(t_pred_embedd, dim=1)
        # h_pred_embedd = torch.mean(h_pred_embedd, dim=1)

        # ii) Take the embeddings of the entitiy that is in the positive query.
        # In t corruptions, the original index is the tail of the first element of the list, in h corruptions, the original index is the head of the first element of the list  
        t_index = batch[:,1]#.unsqueeze(1)
        h_index = batch[:,0]#.unsqueeze(1)
        # in the embeddings, the representation given by the index, in the first dimension, is the original representation of the entity
        t_embedd = torch.zeros(t_index.shape[0], t_pred_embedd.shape[2])
        h_embedd = torch.zeros(h_index.shape[0], h_pred_embedd.shape[2])
        for i in range(len(t_index)):
            t_embedd[i] = t_pred_embedd[i][t_index[i]]
            h_embedd[i] = h_pred_embedd[i][h_index[i]]
        # Squeeze the embeddings to remove the first dimension
        t_embedd = torch.squeeze(t_embedd, dim=1)
        h_embedd = torch.squeeze(h_embedd, dim=1)

        # convert the scores and embeddings to tf
        t_pred_scores = tf.constant(t_pred_scores.detach().numpy(), dtype=tf.float32)
        t_embedd = tf.constant(t_embedd.detach().numpy(), dtype=tf.float32)
        h_pred_scores = tf.constant(h_pred_scores.detach().numpy(), dtype=tf.float32)
        h_embedd = tf.constant(h_embedd.detach().numpy(), dtype=tf.float32)
        return t_pred_scores, t_embedd

    def get_ultra_outputs(self, batch):
        '''
        What I need to do is to get the outputs from ultra, and create labels that mimic those outputs, give those labels as y. I need to create the concept
        embeddings and outputs in the model according to that. 
            -concept (task) output initial, concept embedd initial needs to be shape (len(sum A_pred), 1), (len(sum A_pred), embed size)
            -concept (task) output final, labels needs to be shape (n_queries, num_negatives->triplets), (n_queries, num_negatives->int 1/0)
        '''
        print('Batch:', len(batch)  ,'sets of corruptions:', [len(b) for b in batch])

        batch = split_positives_negatives(batch)
        print('Batch splitted by positives and negatives:', len(batch), 'sets of corruptions:', [len(b) for b in batch])

        batches = split_by_corruptions(batch)
        print('Batch splitted by corruptions {#number of samples:#number of negatives}:', batches.keys(), [len(b) for b in batches.values()])
        for key in batches.keys():
            if key != 1: # avoid only postivies when the corruption is only tail
                batches[key] = np.concatenate(batches[key], axis=0)
                batches[key] = torch.tensor(batches[key], dtype=torch.int64)

        scores = []
        for key,batch in batches.items():
            score = self.Ultra(self.aux_dataset, batch)
            scores.append(score)

        labels_new = np.empty(len(scores), dtype=object)
        for i in range(len(scores)): 
            labels_new[i] = np.zeros(tuple(scores[i].shape)) 
            
        for l  in labels_new:
            l[:,0] = 1

        # convert scores and labels to tf
        scores = [score.detach().numpy() for score in scores]
        # print('All scores:', len(scores), [s.shape for s in scores])
        # scores_tf = tf.ragged.constant(scores, dtype=tf.float32)
        # for scores, create a ragged tensor where all the elements (np arrays) are concatenated in the first dimension
        ragged_scores = [tf.RaggedTensor.from_tensor(score) for score in scores]
        scores_tf = tf.concat(ragged_scores, axis=0)
        # print('Scores tf shape:', scores_tf.shape)
        # labels_tf = tf.ragged.constant(labels_new, dtype=tf.float32)  
        ragged_labels = [tf.RaggedTensor.from_tensor(label) for label in labels_new]
        labels_tf = tf.concat(ragged_labels, axis=0)
        # print('Labels tf shape:', labels_tf.shape)

        metrics = calculated_metrics_batched(scores_tf, labels_tf)

        return scores_tf, labels_tf



    def mimic_test_function_ultra_with_our_corruptions(self, batch):
        # 1) From the batch predict all negatives, call the model and then apply the mask
        # 2) From the batch divide by tail and head and num_negatives before calling the model -> THIS ONE
        # remove the duplicated triplets
        # print('Batch:', len(batch), batch)
        # batch = list(dict.fromkeys(tuple(t) for t in batch))
        # print('Batch:', len(batch)  ,'sets of corruptions:', [len(b) for b in batch])

        batch = split_positives_negatives(batch)
        # print('Batch splitted by positives and negatives:', len(batch), 'sets of corruptions:', [len(b) for b in batch])

        batches = split_by_corruptions(batch)
        # print('Batch splitted by corruptions:', batches.keys(), [len(b) for b in batches.values()])
        for key in batches.keys():
            if key != 1: # avoid only postivies when the corruption is only tail
                batches[key] = np.concatenate(batches[key], axis=0)
                batches[key] = torch.tensor(batches[key], dtype=torch.int64)

        scores = []
        for key,batch in batches.items():
            score = self.Ultra(self.aux_dataset, batch)
            scores.append(score)
        # print('All scores:', len(scores), [s.shape for s in scores])

        # since the order of labels is not preserved, I need to create the labels again (always the first element is 1)            
        labels_new = np.empty(len(scores), dtype=object)
        for i in range(len(scores)): 
            labels_new[i] = np.zeros(tuple(scores[i].shape)) 
            
        for l  in labels_new:
            l[:,0] = 1

        # convert scores and labels to tf
        scores = [score.detach().numpy() for score in scores]
        scores_tf = tf.ragged.constant(scores, dtype=tf.float32)
        labels_tf = tf.ragged.constant(labels_new, dtype=tf.float32)  

        metrics = calculate_metrics(scores,scores_tf, labels_tf)

        return scores_tf, labels_tf






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




def calculate_metrics(scores,scores_tf, labels_tf):

    metrics = {'MRR': [], 'Hits@1': [], 'Hits@3': [], 'Hits@10': []}
    for i in range(len(scores)): # calculate the metrics for s (need to convert it to numpy)
        mrr_metric = MRRMetric()
        mrr_metric.update_state(labels_tf[i], scores_tf[i])
        hits_metric_1 = HitsMetric(1)
        hits_metric_3 = HitsMetric(3)
        hits_metric_10 = HitsMetric(10)
        hits_metric_1.update_state(labels_tf[i], scores_tf[i])
        hits_metric_3.update_state(labels_tf[i], scores_tf[i])
        hits_metric_10.update_state(labels_tf[i], scores_tf[i])
        print('MRR:', mrr_metric.result().numpy(), 'Hits@1:', hits_metric_1.result().numpy(), 
                'Hits@3:', hits_metric_3.result().numpy(), 'Hits@10:', hits_metric_10.result().numpy())
        
        metrics['MRR'].extend([mrr_metric.result().numpy()]*len(scores[i])) 
        metrics['Hits@1'].extend([hits_metric_1.result().numpy()]*(len(scores[i])))
        metrics['Hits@3'].extend([hits_metric_3.result().numpy()]*(len(scores[i])))
        metrics['Hits@10'].extend([hits_metric_10.result().numpy()]*(len(scores[i])))
    # calculate the average of the metrics
    print('Metrics tf:',*[f"{key}: {np.mean(value)}" for key, value in metrics.items()], sep='\n')

def calculated_metrics_batched(scores_tf, labels_tf):

    mrr_metric = MRRMetric()
    hits_metric_1 = HitsMetric(1)
    hits_metric_3 = HitsMetric(3)
    hits_metric_10 = HitsMetric(10)
    mrr_metric.update_state(labels_tf, scores_tf)
    hits_metric_1.update_state(labels_tf, scores_tf)
    hits_metric_3.update_state(labels_tf, scores_tf)
    hits_metric_10.update_state(labels_tf, scores_tf)
    print('MRR:', mrr_metric.result().numpy(), 'Hits@1:', hits_metric_1.result().numpy(),
            'Hits@3:', hits_metric_3.result().numpy(), 'Hits@10:', hits_metric_10.result().numpy())
    

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

    train_data = Dataset_Ultra(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                      target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_relations_no_inv*2)
    train_data.num_edges = train_data.edge_index.shape[1]
    valid_data = Dataset_Ultra(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                      target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations_no_inv*2)
    valid_data.num_edges = valid_data.edge_index.shape[1]
    test_data = Dataset_Ultra(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                      target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations_no_inv*2)
    test_data.num_edges = test_data.edge_index.shape[1]
    
    # edge_index is the sum of all target_edge_index
    edge_index = torch.cat([train_target_edges, valid_edges, test_edges], dim=1)
    edge_type = torch.cat([train_target_etypes, valid_etypes, test_etypes])
    # num_nodes is given by the train set
    num_edges = None # is not defined in Ultra for the general dataset
    device = 'cpu'
    dataset = Dataset_Ultra(edge_index, edge_type, num_relations_no_inv*2, num_node, num_edges, device)
    filtered_data = dataset 

    train_data = build_relation_graph(train_data)
    valid_data = build_relation_graph(valid_data)
    test_data = build_relation_graph(test_data)

    return train_data, valid_data, test_data, filtered_data




def get_ragged_tensor_shape(rt):
    if not isinstance(rt, tf.RaggedTensor):
        return None
    
    row_lengths = rt.nested_row_lengths()
    shapes = []
    current_shape = tf.shape(rt.flat_values)
    for lengths in reversed(row_lengths):
        current_shape = tf.concat([[tf.shape(lengths)[0]], current_shape], axis=0)
        shapes.append(current_shape)
    shapes.reverse()
    
    return [shape.numpy().tolist() for shape in shapes]