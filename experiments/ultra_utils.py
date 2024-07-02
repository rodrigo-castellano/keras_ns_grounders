import torch
import torch.nn as nn
from torch import distributed as dist
from torch.utils import data as torch_data

from ULTRA.ultra import layers
from ULTRA.ultra.models import RelNBFNet,BaseNBFNet
# from ULTRA.ultra.models import EntityNBFNet as original_entity_model
from ULTRA.ultra import tasks,util

import numpy as np
import tensorflow as tf
from collections import defaultdict
import math 
import itertools
from ns_lib.utils import MRRMetric, HitsMetric
from collections import OrderedDict


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

    # print('\n\nUltra', model)
    # for name, param in model.named_parameters():             
    #     print(name, param.shape)
    #     print(param[0][:5]) if len(param.shape) > 1 else print(param[:5])

    # 'C:\\Users\\rodri\\Downloads\\PhD_code\\Review_grounders\\keras_ns_grounders\\ULTRA\\ckpts\\ultra_4g.pth'
    state = torch.load('C:\\Users\\rodri\\Downloads\\PhD_code\\Review_grounders\\keras_ns_grounders\\ULTRA\\ckpts\\countries_ft_epoch_10.pth', map_location="cpu")
    state = state['model']


    # subsitution of the keys of the state dict. 'mlp.0' by 'mlp_embedd.0' and 'mlp.2' by 'mlp_score.2'
    if not original:
        state = OrderedDict((key.replace('mlp', 'mlp_embedd'), value) if 'mlp.0' in key else (key.replace('mlp', 'mlp_score'), value) for key, value in state.items())
    # print('\n\nState', state.keys())
    # for key in state.keys():
    #     print(key, state[key].shape)
    #     print(state[key][0][:5]) if len(state[key].shape) > 1 else print(state[key][:5])

    model.load_state_dict(state, strict=False) # Filter out the keys that correspond to the final layer

    # print('\n\nUltra loaded')
    # for name, param in model.named_parameters():             
    #     print(name, param.shape)
    #     print(param[0][:5]) if len(param.shape) > 1 else print(param[:5])

    model = model.to('cpu')
    return model


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
    




def get_ultra_outputs_nonfiltered_negatives(dataset,queries, Ultra):


    batch = [q[0] for q in queries] # take only the positives
    # filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=train_data.num_nodes)
    # filtered_data = dataset
    filtered_data = None

    val_filtered_data = test_filtered_data = filtered_data
    # metrics = test(self.Ultra, dataset, 'cpu', filtered_data=filtered_data, return_metrics=True) 

    # 1) From the batch predict all negatives, call the model and then apply the mask -> THIS ONE
    # 2) From the batch divide by tail and head and num_negatives before calling the model

    # remove the duplicated triplets
    batch = list(dict.fromkeys(tuple(t) for t in batch))
    # convert the list of lists to torch tensor
    batch = torch.tensor(batch, dtype=torch.int64)
    print('Batch:', batch.shape)

    t_batch, h_batch = tasks.all_negative(dataset, batch)
    t_pred = Ultra(dataset, t_batch)
    h_pred = Ultra(dataset, h_batch)

    if filtered_data is None:
        t_mask, h_mask = tasks.strict_negative_mask(dataset, batch)
    else:
        t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
    
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    num_t_negative = t_mask.sum(dim=-1) # number of negatives for each query
    num_h_negative = h_mask.sum(dim=-1) # number of negatives for each query

    # GET THE METRICS FROM TF
    mrr_head, mrr_tail = preprocess_tf_metrics(h_pred, t_pred, pos_h_index, pos_t_index, num_h_negative, num_t_negative, h_batch, t_batch, h_mask, t_mask)



    # queries_positive = [q[0] for q in queries]
    # batch = torch.tensor(queries_positive, dtype=torch.int64)
    # print('Batch:', batch.shape)
    # # Get all the negatives of the positive atoms 
    # t_batch, h_batch = tasks.all_negative(dataset, batch)

    # # Get the embeddings of the atoms in A_pred and the negatives
    # t_pred_scores = Ultra(dataset, t_batch)
    # h_pred_scores = Ultra(dataset, h_batch)


    # # EMBEDDINGS, I can return them in different ways. It is better to obtain the scores from the embeddings in our mode. 

    # # # i) Do the average of the embeddings of all the entities. This is the most general way, but it is not the best way to represent the entities
    # # t_pred_embedd = torch.mean(t_pred_embedd, dim=1)
    # # h_pred_embedd = torch.mean(h_pred_embedd, dim=1)

    # # ii) Take the embeddings of the entitiy that is in the positive query.
    # # In t corruptions, the original index is the tail of the first element of the list, in h corruptions, the original index is the head of the first element of the list  
    # t_index = batch[:,1]
    # h_index = batch[:,0]
    # # in the embeddings, the representation given by the index, in the first dimension, is the original representation of the entity
    # t_embedd = torch.zeros(t_index.shape[0], t_pred_embedd.shape[2])
    # h_embedd = torch.zeros(h_index.shape[0], h_pred_embedd.shape[2])
    # for i in range(len(t_index)):
    #     t_embedd[i] = t_pred_embedd[i][t_index[i]]
    #     h_embedd[i] = h_pred_embedd[i][h_index[i]]

    

    # # TAIL CORRUPTIONS
    # scores = np.squeeze(t_pred_scores.detach().numpy())
    # scores_tf = tf.RaggedTensor.from_tensor(scores)
    # labels_new = np.zeros(tuple(scores.shape))
    # labels_new[:,0] = 1
    # labels_tf = tf.RaggedTensor.from_tensor(labels_new)
    # print('Tail corruptions:')
    # calculated_metrics_batched(scores_tf, labels_tf)

    # # HEAD CORRUPTIONS
    # scores = np.squeeze(h_pred_scores.detach().numpy())
    # scores_tf = tf.RaggedTensor.from_tensor(scores)
    # labels_new = np.zeros(tuple(scores.shape))
    # labels_new[:,0] = 1
    # labels_tf = tf.RaggedTensor.from_tensor(labels_new)
    # print('Head corruptions:')
    # calculated_metrics_batched(scores_tf, labels_tf)

    # return scores_tf, t_embedd
    return None, None

def get_ultra_outputs(dataset, batch, Ultra):
    '''
    What I need to do is to get the outputs from ultra, and create labels that mimic those outputs, give those labels as y. I need to create the concept
    embeddings and outputs in the model according to that. 
        -concept (task) output initial, concept embedd initial needs to be shape (len(sum A_pred), 1), (len(sum A_pred), embed size)
        -concept (task) output final, labels needs to be shape (n_queries, num_negatives->triplets), (n_queries, num_negatives->int 1/0)
    '''
    # print('Batch:', len(batch)  ,'sets of corruptions:', [len(b) for b in batch])
    batch = split_positives_negatives(batch)
    # print('Batch splitted by positives and negatives:', len(batch), 'sets of corruptions:', [len(b) for b in batch])
    batches = split_by_corruptions(batch)
    # print('Batch splitted by corruptions {#number of samples:#number of negatives}:', batches.keys(), [len(b) for b in batches.values()])
    for key in batches.keys():
        if key != 1: # avoid only postivies when the corruption is only tail
            batches[key] = np.concatenate(batches[key], axis=0)
            batches[key] = torch.tensor(batches[key], dtype=torch.int64)

    scores = []
    for key,batch in batches.items():
        # score = self.Ultra(self.aux_dataset, batch)
        score,_ = Ultra(dataset, batch, all_representations=False)
        scores.append(score)
    labels_new = np.empty(len(scores), dtype=object)
    for i in range(len(scores)): 
        labels_new[i] = np.zeros(tuple(scores[i].shape)) 
    for l  in labels_new:
        l[:,0] = 1

    # convert scores and labels to tf
    scores = [score.detach().numpy() for score in scores]
    # for scores, create a ragged tensor where all the elements (np arrays) are concatenated in the first dimension
    ragged_scores = [tf.RaggedTensor.from_tensor(score) for score in scores]
    scores_tf = tf.concat(ragged_scores, axis=0)
    ragged_labels = [tf.RaggedTensor.from_tensor(label) for label in labels_new]
    labels_tf = tf.concat(ragged_labels, axis=0)
    calculated_metrics_batched(scores_tf, labels_tf)

    return scores_tf, labels_tf

def mimic_test_function_ultra_with_our_corruptions(dataset, batch, Ultra):
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
        score = Ultra(dataset, batch)
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


def mimic_test_function_ultra(dataset, batch, model):

    # def lazy_import():
    #     from other_scripts.ultra_test import preprocess_tf_metrics
    #     return preprocess_tf_metrics
    # preprocess_tf_metrics = lazy_import()
    batch = [q[0] for q in batch] # take only the positives

    # filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=train_data.num_nodes)
    # filtered_data = dataset
    filtered_data = None
    val_filtered_data = test_filtered_data = filtered_data

    # metrics = test(self.model, dataset, 'cpu', filtered_data=filtered_data, return_metrics=True) 

    # 1) From the batch predict all negatives, call the model and then apply the mask -> THIS ONE
    # 2) From the batch divide by tail and head and num_negatives before calling the model

    # remove the duplicated triplets
    batch = list(dict.fromkeys(tuple(t) for t in batch))
    # convert the list of lists to torch tensor
    batch = torch.tensor(batch, dtype=torch.int64)
    print('Batch:', batch.shape)

    t_batch, h_batch = tasks.all_negative(dataset, batch)
    t_pred = model(dataset, t_batch)
    h_pred = model(dataset, h_batch)

    if filtered_data is None:
        t_mask, h_mask = tasks.strict_negative_mask(dataset, batch)
    else:
        t_mask, h_mask = tasks.strict_negative_mask(filtered_data, batch)
    
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    num_t_negative = t_mask.sum(dim=-1) # number of negatives for each query
    num_h_negative = h_mask.sum(dim=-1) # number of negatives for each query

    # GET THE METRICS FROM TF
    mrr_head, mrr_tail = preprocess_tf_metrics(h_pred, t_pred, pos_h_index, pos_t_index, num_h_negative, num_t_negative, h_batch, t_batch, h_mask, t_mask)



def mimic_ultra(dataset, model):

    # def lazy_import():
    #     from other_scripts.ultra_test import test
    #     return test
    # test = lazy_import()
    
    rel_model_cfg = {"class": "RelNBFNet","input_dim": 64,"hidden_dims": [64, 64, 64, 64, 64, 64],"message_func": "distmult","aggregate_func": "sum","short_cut": "yes","layer_norm": "yes"} 
    entity_model_cfg= {"class": "EntityNBFNet","input_dim": 64,"hidden_dims": [64, 64, 64, 64, 64, 64],"message_func": "distmult","aggregate_func": "sum","short_cut": "yes","layer_norm": "yes"}
    model = model(rel_model_cfg, entity_model_cfg)                
    ckp = None
    ckp = 'C:\\Users\\rodri\\Downloads\\PhD_code\\Review_grounders\\keras_ns_grounders\\ULTRA\\ckpts\\countries_ft_epoch_10.pth'
    if ckp is not None:
        state = torch.load(ckp, map_location="cpu")
        model.load_state_dict(state["model"])
    model = model.to('cpu')

    # filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=train_data.num_nodes)
    # filtered_data = dataset
    filtered_data = None

    val_filtered_data = test_filtered_data = filtered_data
    metrics = test(model, dataset, 'cpu', filtered_data=filtered_data, return_metrics=True) 



def preprocess_tf_metrics(h_pred, t_pred, pos_h_index, pos_t_index, num_h_negative, num_t_negative, h_batch, t_batch, h_mask, t_mask):
    """
    Preprocess data for TensorFlow metrics calculation.
    
    This function processes head and tail predictions, creates labels, filters predictions,
    and calculates Mean Reciprocal Rank (MRR) for both head and tail predictions.
    
    Args:
    h_pred, t_pred: Tensor, predictions for heads and tails
    pos_h_index, pos_t_index: Tensor, indices of positive samples for heads and tails
    num_h_negative, num_t_negative: Tensor, number of negative samples for heads and tails
    h_batch, t_batch: Tensor, batch data for heads and tails
    h_mask, t_mask: Tensor, masks for filtering valid samples
    
    Returns:
    mrr_head, mrr_tail: float, Mean Reciprocal Rank for head and tail predictions
    """
    
    # Helper function to process both head and tail data
    def process_batch(pred, batch, mask, pos_index, num_negative):
        pred_np = pred.cpu().detach().numpy()
        pos_index_np = pos_index.cpu().numpy()
        num_negative_np = num_negative.cpu().numpy()
        
        labels = []
        predictions_filtered = []
        
        for i, sample in enumerate(batch.cpu().numpy()[:,:,0] if batch.shape[-1] == 1 else batch.cpu().numpy()[:,:,1]):
            # Create label vector: 1 for positive sample, 0 for others
            label = np.zeros(len(sample))
            label[pos_index_np[i]] = 1
            
            # Ensure positive sample is included in the mask
            mask[i][pos_index_np[i]] = True
            
            # Filter predictions and labels based on the mask
            pred_filtered = pred_np[i][mask[i].cpu().numpy()]
            label_filtered = label[mask[i].cpu().numpy()]
            
            # Verify the number of filtered samples
            assert num_negative_np[i] + 1 == len(pred_filtered), 'Mismatch in number of valid samples'
            
            predictions_filtered.append(pred_filtered)
            labels.append(label_filtered)
        
        return labels, predictions_filtered
    
    # Process head and tail data
    labels_h, predictions_h_filtered = process_batch(h_pred, h_batch, h_mask, pos_h_index, num_h_negative)
    labels_t, predictions_t_filtered = process_batch(t_pred, t_batch, t_mask, pos_t_index, num_t_negative)
    
    # Convert to TensorFlow ragged tensors
    labels_tensor_h = tf.ragged.constant(labels_h, dtype=tf.float32)
    labels_tensor_t = tf.ragged.constant(labels_t, dtype=tf.float32)
    predictions_tensor_h = tf.ragged.constant(predictions_h_filtered, dtype=tf.float32)
    predictions_tensor_t = tf.ragged.constant(predictions_t_filtered, dtype=tf.float32)
    
    # Compute Mean Reciprocal Rank (MRR)
    mrr_metric = MRRMetric()
    mrr_metric.update_state(labels_tensor_h, predictions_tensor_h)
    mrr_head = mrr_metric.result().numpy()
    
    mrr_metric.reset_states()
    mrr_metric.update_state(labels_tensor_t, predictions_tensor_t)
    mrr_tail = mrr_metric.result().numpy()
    
    # Print results
    print(f'TF MRR average: {(mrr_head + mrr_tail) / 2:.3f}, TF MRR head: {mrr_head:.3f}, TF MRR tail: {mrr_tail:.3f}')
    
    return mrr_head, mrr_tail





@torch.no_grad()
def test(model, test_data, device, filtered_data=None, return_metrics=True):
    world_size = util.get_world_size()
    rank = util.get_rank()
    # test_triplets=[n_triplets, 3]
    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, 16, sampler=sampler)
    model.eval()
    rankings = []
    num_negatives = []
    tail_rankings, num_tail_negs = [], []  # for explicit tail-only evaluation needed for 5 datasets
    mrr_head_all, mrr_tail_all = [], []
    for i,batch in enumerate(test_loader):
        print('\nbatch',i, 'of',len(test_loader), batch.shape)#,batch)
        t_batch, h_batch = tasks.all_negative(test_data, batch) # t_batch=[batch_size=16, 271, 3] all the negative tails. For each of the 16 heads, 1594 negative tails are sampled
        print('t_batch', t_batch.shape, 'h_batch', h_batch.shape)
        t_pred = model(test_data, t_batch)
        h_pred = model(test_data, h_batch) # h_pred=[batch_size,271] scores for each of the 271 negative heads
        # if we are in transductive: filtered is the whole dataset, i.e., we check that in corruptions there's not any triple from the whole dataset
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
        mrr_head_all.append(mrr_head)
        mrr_tail_all.append(mrr_tail)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

        tail_rankings += [t_ranking]
        num_tail_negs += [num_t_negative]
    
    print('\n','TF avg MRR', np.round((np.mean(mrr_head_all)+np.mean(mrr_tail_all))/2,3),'TF avg MRR head', np.round(np.mean(mrr_head_all),3), 'TF avg MRR tail', np.round(np.mean(mrr_tail_all),3) )

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)

    # ugly repetitive code for tail-only ranks processing
    tail_ranking = torch.cat(tail_rankings)
    num_tail_neg = torch.cat(num_tail_negs)
    all_size_t = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_t[rank] = len(tail_ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_t, op=dist.ReduceOp.SUM)

    # obtaining all ranks 
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative

    # the same for tails-only ranks
    cum_size_t = all_size_t.cumsum(0)
    all_ranking_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_ranking_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = tail_ranking
    all_num_negative_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_num_negative_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = num_tail_neg
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_ranking_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative_t, op=dist.ReduceOp.SUM)

    print('\nMetrics from Ultra:')
    metrics = {}
    if rank == 0:
        for metric in ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']:
            if "-tail" in metric:
                _metric_name, direction = metric.split("-")
                if direction != "tail":
                    raise ValueError("Only tail metric is supported in this mode")
                _ranking = all_ranking_t
                _num_neg = all_num_negative_t
            else:
                _ranking = all_ranking 
                _num_neg = all_num_negative 
                _metric_name = metric
            
            if _metric_name == "mr":
                score = _ranking.float().mean()
            elif _metric_name == "mrr":
                score = (1 / _ranking.float()).mean()
            elif _metric_name.startswith("hits@"):
                values = _metric_name[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (_ranking - 1).float() / _num_neg
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (_ranking <= threshold).float().mean()
            print("%s: %g" % (metric, score))
            metrics[metric] = score
    mrr = (1 / all_ranking.float()).mean()

    return mrr if not return_metrics else metrics














################################################ MODEL ############################################################ 



def nested_dict(n, type):
    """
    Creates a nested defaultdict with n levels.

    Parameters:
    n (int): The number of nested levels for the defaultdict.
    type: The default data type for the defaultdict.

    Returns:
    defaultdict: A nested defaultdict with n levels.
    """
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))



class EntityNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1,get_scores=False, **kwargs):

        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
            )

        feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim
        self.mlp = nn.Sequential()
        mlp = []
        self.get_scores = get_scores
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())

        # mlp.append(nn.Linear(feature_dim, 1))
        # self.mlp = nn.Sequential(*mlp)

        self.mlp_embedd = nn.Sequential(*mlp)
        if get_scores:
            mlp.append(nn.Linear(feature_dim, 1))
        self.mlp_score = nn.Sequential(*mlp)

    
    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        batch_size = len(r_index)

        # initialize queries (relation types of the given triples)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary
        for layer in self.layers:
            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, relation_representations, batch, all_representations=False):
        h_index, t_index, r_index = batch.unbind(-1)
        # initial query representations are those from the relation graph
        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)
            # UPDATE THE NUMBER OF EDGES
            data.num_edges = data.edge_index.shape[1]

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        # h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=torch.div(data.num_relations, 2, rounding_mode='floor'))
        # here Im supposed to have head with all the same indices for every query
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations of the head entities
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]

        if not all_representations:
            # Get the tail indices [19,num_negative+1], then unesqueeze it to [19,num_negative+1,1], then expand (repeat) it to [19,num_negative+1,64] 
            index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
            # here, for every query (pos and neg), I get the tail embeddings
            feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)
            # # From these embeddings of the pos and negatives queries, Im just interested in the postive query (self representation)
            # feature = feature[:, :1, :]

        embedd = self.mlp_embedd(feature)
        scores = self.mlp_score(feature).squeeze(-1) if self.get_scores else None
        # scores = self.mlp(feature).squeeze(-1) if self.get_scores else None
        return scores, embedd


class Ultra(nn.Module):

    def __init__(self):
        # kept that because super Ultra sounds cool
        super(Ultra, self).__init__()
        self.relation_model = RelNBFNet(input_dim=64, hidden_dims=[64, 64, 64, 64, 64, 64], message_func='distmult', aggregate_func='sum', short_cut=True, layer_norm=True)
        self.entity_model = EntityNBFNet(input_dim=64, hidden_dims=[64, 64, 64, 64, 64, 64], message_func='distmult', aggregate_func='sum', short_cut=True, layer_norm=True,get_scores=True)
    
    def convert_to_domain(self, entity_representations, fol):
        '''
        Given the entity representations, I put them in a dict with different domains. I could take the dic with the index for every constant and domain, but it is easier and faster this way
        The order followed here is the same order followed when creating constant_to_global_index
        '''
        # self.constant_to_global_index = defaultdict(OrderedDict)
        entity_embeddings = {domain.name:[] for domain in fol.domains}
        for domain in fol.domains:
            # Add fixed constants.
            for i, constant in enumerate(domain.constants):
                # self.constant_to_global_index[domain.name][constant] = i
                # entity_embeddings[domain.name].append(entity_representations[i].detach().numpy())
                entity_embeddings[domain.name].append(entity_representations[i].detach().cpu().numpy())
        
        # # convert to tf tensors
        # for key in entity_embeddings.keys():
        #     entity_embeddings[key] = tf.convert_to_tensor(entity_embeddings[key])

        return entity_embeddings


    def forward(self, data, batch, all_representations=False):
        # batch shape: (bs, 1+num_negs, 3)

        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        query_rels = batch[:, 0, 2]
        # relation_representations:  torch.Size([16, 360, 64])=(queries,n_relationsx2,dim_embedd)
        relation_representations = self.relation_model(data.relation_graph, query=query_rels)
        
        scores,entity_representations = self.entity_model(data, relation_representations, batch)
        # entity_representations, scores = self.entity_model(data, relation_representations, batch,all_representations=all_representations) # [16,1594,64] = [batch_size, num_negatives, embedd_size]
        atom_representations = entity_representations
        # OPTION 2, GET THE ENTITIES AND RELATIONS REPRESENTATIONS
        return scores, atom_representations
        
 




