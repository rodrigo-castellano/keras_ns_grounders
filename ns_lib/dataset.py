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

import torch
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir, '..', 'experiments'))

# from ultra_utils import Ultra
# from ultra_utils import build_relation_graph
from ULTRA.ultra.models import Ultra
from ULTRA.ultra import tasks

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


class Dataset_Ultra():
    
        def __init__(self, edge_index, edge_type, num_relations, num_nodes, num_edges, device):
            self.edge_index = edge_index
            self.edge_type = edge_type
            self.num_relations = num_relations
            self.num_nodes = num_nodes
            self.num_edges = num_edges
            self.device = device

def load_ultra_model():
    # self.define_ultra_dataset()
    # self.Ultra = Ultra()

    rel_model_cfg = {"class": "RelNBFNet","input_dim": 64,"hidden_dims": [64, 64, 64, 64, 64, 64],"message_func": "distmult","aggregate_func": "sum","short_cut": "yes","layer_norm": "yes"} 
    entity_model_cfg= {"class": "EntityNBFNet","input_dim": 64,"hidden_dims": [64, 64, 64, 64, 64, 64],"message_func": "distmult","aggregate_func": "sum","short_cut": "yes","layer_norm": "yes"}
    Ultra = Ultra(rel_model_cfg, entity_model_cfg)

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

    Ultra.load_state_dict(state, strict=False) # Filter out the keys that correspond to the final layer

    # # show the parameters of the model
    # print('\n\nUltra loaded')
    # for name, param in Ultra.named_parameters():             
    #     print(name, param.shape)
    #     print(param[0][:5]) if len(param.shape) > 1 else print(param[:5])

    Ultra = Ultra.to('cpu')
    return Ultra

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
            self.Ultra = load_ultra_model()

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
            use_ultra_original = True
            if not use_ultra_original:
                scores,atom_embeds = self.Ultra(self.aux_dataset, A_predicates_triplets, atom_repr=True) # embedds of the atoms
                scores = tf.constant(scores, dtype=tf.float32)
                atom_embeds = tf.constant(atom_embeds, dtype=tf.float32)
                embeddings = (scores, atom_embeds)
            else:
                self.Ultra.eval()

                # self.mimic_ultra()
                # self.mimic_test_function_ultra(Q_global_positive)
                scores = self.mimic_test_function_ultra_with_corruptions(Q_global)
                embeddings  = (scores, None)

        elif self.use_llm:
            atom_embeds = self.llm_embedder(A_predicates_textualized)
            atom_embeds = tf.constant(atom_embeds, dtype=tf.float32)
            embeddings = (None, atom_embeds)
            # _,atom_embeds = self.Ultra(self.aux_dataset, A_predicates_triplets, atom_repr=True) # embedds of the atoms
            # atom_embeds = tf.constant(atom_embeds, dtype=tf.float32)
            # embeddings = (_, atom_embeds)

        return (X_domains_data, A_predicates_data, A_rules_data, Q, embeddings), y


    def split_by_corruptions(self, batch):
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
    
    def split_positives_negatives(self, batch):
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
    

    def mimic_test_function_ultra_with_corruptions(self, batch):
        # 1) From the batch predict all negatives, call the model and then apply the mask
        # 2) From the batch divide by tail and head and num_negatives before calling the model -> THIS ONE

        # remove the duplicated triplets
        # print('Batch:', len(batch), batch)
        # batch = list(dict.fromkeys(tuple(t) for t in batch))
        print('Batch:', len(batch)  ,'sets of corruptions:', [len(b) for b in batch])

        batch = self.split_positives_negatives(batch)
        print('Batch splitted by positives and negatives:', len(batch), 'sets of corruptions:', [len(b) for b in batch])

        batches = self.split_by_corruptions(batch)
        print('Batch splitted by corruptions:', batches.keys(), [len(b) for b in batches.values()])
        for key in batches.keys():
            if key != 1: # avoid only postivies when the corruption is only tail
                batches[key] = np.concatenate(batches[key], axis=0)
                batches[key] = torch.tensor(batches[key], dtype=torch.int64)

        scores = []
        for key,batch in batches.items():
            score = self.Ultra(self.aux_dataset, batch)
            scores.append(score)
        print('All scores:', len(scores), [s.shape for s in scores])

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



    # def mimic_test_function_ultra(self, batch):

    #     def lazy_import():
    #         from ultra_test import preprocess_tf_metrics
    #         return preprocess_tf_metrics
    #     preprocess_tf_metrics = lazy_import()
    #     batch = [q[0] for q in batch] # take only the positives
    #     # filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=train_data.num_nodes)
    #     # filtered_data = dataset
    #     filtered_data = None

    #     val_filtered_data = test_filtered_data = filtered_data
    #     # metrics = test(self.Ultra, self.aux_dataset, 'cpu', filtered_data=filtered_data, return_metrics=True) 

    #     # 1) From the batch predict all negatives, call the model and then apply the mask -> THIS ONE
    #     # 2) From the batch divide by tail and head and num_negatives before calling the model

    #     # remove the duplicated triplets
    #     batch = list(dict.fromkeys(tuple(t) for t in batch))
    #     # convert the list of lists to torch tensor
    #     batch = torch.tensor(batch, dtype=torch.int64)
    #     print('Batch:', batch.shape)

    #     t_batch, h_batch = tasks.all_negative(self.aux_dataset, batch)
    #     t_pred = self.Ultra(self.aux_dataset, t_batch)
    #     h_pred = self.Ultra(self.aux_dataset, h_batch)

    #     if filtered_data is None:
    #         t_mask, h_mask = tasks.strict_negative_mask(self.aux_dataset, batch)
    #     else:
    #         t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        
    #     pos_h_index, pos_t_index, pos_r_index = batch.t()
    #     num_t_negative = t_mask.sum(dim=-1) # number of negatives for each query
    #     num_h_negative = h_mask.sum(dim=-1) # number of negatives for each query
 
    #     # GET THE METRICS FROM TF
    #     mrr_head, mrr_tail = preprocess_tf_metrics(h_pred, t_pred, pos_h_index, pos_t_index, num_h_negative, num_t_negative, h_batch, t_batch, h_mask, t_mask)


    # def mimic_ultra(self):

    #     def lazy_import():
    #         from ultra_test import test
    #         return test
    #     test = lazy_import()
        # rel_model_cfg = {"class": "RelNBFNet","input_dim": 64,"hidden_dims": [64, 64, 64, 64, 64, 64],"message_func": "distmult","aggregate_func": "sum","short_cut": "yes","layer_norm": "yes"} 
        # entity_model_cfg= {"class": "EntityNBFNet","input_dim": 64,"hidden_dims": [64, 64, 64, 64, 64, 64],"message_func": "distmult","aggregate_func": "sum","short_cut": "yes","layer_norm": "yes"}
    #     model = Ultra(rel_model_cfg, entity_model_cfg)                
    #     ckp = None
    #     ckp = 'C:\\Users\\rodri\\Downloads\\PhD_code\\Review_grounders\\keras_ns_grounders\\ULTRA\\ckpts\\countries_ft_epoch_10.pth'
    #     if ckp is not None:
    #         state = torch.load(ckp, map_location="cpu")
    #         model.load_state_dict(state["model"])
    #     model = model.to('cpu')

    #     # filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=train_data.num_nodes)
    #     # filtered_data = dataset
    #     filtered_data = None

    #     val_filtered_data = test_filtered_data = filtered_data
    #     metrics = test(model, self.aux_dataset, 'cpu', filtered_data=filtered_data, return_metrics=True) 


    def define_ultra_dataset(self):
        """
        Get all the information of the dataset graph, as well as the relational graph.
        Also, create the triplets for the queries.

        Returns:
            None
        """
        queries, labels = self.dataset[:]
        constants_features = self.dataset.constants_features
        # I select engine=None because I dont want to ground the rules, only the queries    
        ((X_domains_data, A_predicates_data, _, _, (Q_global,_,_)), _) = _from_strings_to_tensors(
            fol=self.fol,
            serializer=self.serializer,
            queries=queries,
            labels=labels,
            engine=None,
            ragged=self.ragged,
            constants_features=constants_features,
            deterministic=self.deterministic,
            global_serialization=self.global_serialization)
        
        self.Q_global = Q_global    
        # Here Im interested in Q_global, to get the general info that I pass to ultra. From Q_global I can get the triplets of the queries only by taking the first element in every query
        # (the rest are the negative representations)
        self.queries_global = np.array([q[0] for q in Q_global])  

        self.edge_index = self.queries_global[:, :2].T # For all the queries, this takes in the first dim the head, and in the second the tail
        self.edge_type = self.queries_global[:, 2] # For all the queries, it takes the relation
        self.num_relations_no_inv = len(A_predicates_data)
        self.num_relations = len(A_predicates_data)*2
        self.num_nodes = sum(len(X_domains_data[key]) for key in X_domains_data)
        self.num_edges = self.edge_index.shape[1] # it is the number of queries
        self.device = 'cpu' 
    
        # convert edge_index and edge_type to torch tensor to feed it to ULTRA
        self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        self.edge_type = torch.tensor(self.edge_type, dtype=torch.long)

        self.aux_dataset = Dataset_Ultra(edge_index=self.edge_index, edge_type=self.edge_type, num_relations=self.num_relations, 
                                    num_nodes=self.num_nodes, num_edges=self.num_edges, device=self.device)
        self.aux_dataset.device = 'cpu'
        self.aux_dataset = tasks.build_relation_graph(self.aux_dataset)
        self.aux_dataset.fol = self.fol

        return None



    def get_negative_and_outputs(self, model, dataset, batch, atom_repr=True):
        batch_positive = torch.tensor([q[0] for q in batch], dtype=torch.int64)
        t_batch, h_batch = tasks.all_negative(self.aux_dataset, batch_positive)
        t_pred = self.Ultra(self.aux_dataset, t_batch)
        h_pred = self.Ultra(self.aux_dataset, h_batch)
        filtered_data = self.aux_dataset
        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()

        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)

        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)
 
        # GET THE METRICS FROM TF
        mrr_head, mrr_tail = preprocess_tf_metrics(h_pred, t_pred, pos_h_index, pos_t_index, num_h_negative, num_t_negative, h_batch, t_batch, h_mask, t_mask)
        
        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

        tail_rankings += [t_ranking]
        num_tail_negs += [num_t_negative]


    def get_ultra_outputs(self, model, dataset, batch, atom_repr=True):
        """
        Get the outputs of the ultra model for the queries.
        Args:
            model: the ultra model
            dataset: the dataset
            queries: the queries
            atom_repr: if True, the output is the atom representation. If False, the output is the scores.

        Returns:
            The outputs of the ultra model for the queries.
        """
        # print('Batch:', len(batch),'corruptions:', [len(b) for b in batch])
        batches = self.split_by_corruptions(batch)
        for key in batches.keys():
            if key != 1: # avoid only postivies when the corruption is only tail
                batches[key] = np.concatenate(batches[key], axis=0)
                batches[key] = torch.tensor(batches[key], dtype=torch.int64)
        # print('Batches:', batches.keys(), [b.shape for b in batches.values()])

        # For each batch, get the relation representations
        all_relation_representations = []
        all_entity_representations = []
        all_scores = []
        for key,batch in batches.items():
            # print('Batch_i:',batch.shape)
            # if the number of dimensions is 2,add a dimension in the middle (it would mean that there are no negatives, only positives). This is thought for the atom repersentation
            if len(batch.shape) == 2:
                batch = batch.unsqueeze(1)

            # call ultra
            scores = self.Ultra(self.aux_dataset, batch)
            # query_rels = batch[:, 0, 2]
            # relation_representations = self.relation_model(dataset.relation_graph, query=query_rels)
            # batch,relation_representations = self.split_head_tail_negatives(batch,relation_representations)
            # entity_representations, scores = self.entity_model(dataset, relation_representations, batch,atom_repr=atom_repr) # [16,1594,64] = [batch_size, num_negatives, embedd_size]
            # all_relation_representations.append(relation_representations)
            # all_entity_representations.append(entity_representations)
            all_scores.append(scores)
        print('All scores:', len(all_scores), [s.shape for s in all_scores])

        return all_scores




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
