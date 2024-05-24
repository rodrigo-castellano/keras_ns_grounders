from functools import reduce
from torch_scatter import scatter_add
from torch_geometric.data import Data
import torch
import torch.nn as nn
from ULTRA.ultra import layers
from ULTRA.ultra.models import RelNBFNet,BaseNBFNet

import numpy as np
import tensorflow as tf

from collections import defaultdict

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


def pytorch_to_tf(pytorch_tensor):
    numpy_array = pytorch_tensor.detach().cpu().numpy()
    tf_tensor = tf.convert_to_tensor(numpy_array)
    return tf_tensor

def tf_to_pytorch(tf_tensor,type='float'):

    if tf.executing_eagerly():
        # Eager mode: Directly use numpy()
        numpy_array = tf_tensor.numpy()
    else:
        # Graph mode: Use eval() within a session
        with tf.compat.v1.Session() as sess:
            # Convert RaggedTensor to Tensor
            tensor = tf.RaggedTensor.to_tensor(tf_tensor)
            # Evaluate the tensor in the session
            numpy_array = tensor.eval()

    if type == 'float':
        pytorch_tensor = torch.tensor(numpy_array,dtype=torch.float32)  
    elif type == 'int':
        pytorch_tensor = torch.tensor(numpy_array,dtype=torch.int64)  

    return pytorch_tensor



def build_relation_graph(graph):

    # expect the graph is already with inverse edges

    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    device = graph.device

    # Here I need that, if there are 30 entites, the max index is 30. Because the indeces are for the train and test set, it may happen that in the test set that there are indices greater than 30. 
    # Therefore, since here the indices can be local. I create a mapping from (0,1,4,10,120,250,1122)->{0:0,1:1,4:2,10:3,120:4,250:5,1122:6}. 
    ent_idx = graph.edge_index.numpy()
    if graph.num_nodes < torch.max(graph.edge_index):
        mapping = {int(i): idx for idx, i in enumerate(torch.unique(graph.edge_index))} 
        edge_index = torch.tensor([[mapping[int(ent_idx[0,i])],mapping[int(ent_idx[1,i])]] for i in range(ent_idx.shape[1])], device=device).T

    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(dim=0)  # (num_edges, 2) # I concatenate the heads and the relations and remove the duplicates (h,r) in the queries/edges   
    Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0])

    EhT = torch.sparse_coo_tensor(
        torch.flip(Eh, dims=[1]).T, 
        torch.ones(Eh.shape[0], device=device) / Dh[Eh[:, 0]], 
        (num_rels, num_nodes)
    )
    Eh = torch.sparse_coo_tensor(
        Eh.T, 
        torch.ones(Eh.shape[0], device=device), 
        (num_nodes, num_rels)
    )
    Et = torch.vstack([edge_index[1], edge_type]).T.unique(dim=0)  # (num_edges, 2)

    Dt = scatter_add(torch.ones_like(Et[:, 1]), Et[:, 0])
    assert not (Dt[Et[:, 0]] == 0).any()

    EtT = torch.sparse_coo_tensor(
        torch.flip(Et, dims=[1]).T, 
        torch.ones(Et.shape[0], device=device) / Dt[Et[:, 0]], 
        (num_rels, num_nodes)
    )
    Et = torch.sparse_coo_tensor(
        Et.T, 
        torch.ones(Et.shape[0], device=device), 
        (num_nodes, num_rels)
    )

    Ahh = torch.sparse.mm(EhT, Eh).coalesce()
    Att = torch.sparse.mm(EtT, Et).coalesce()
    Aht = torch.sparse.mm(EhT, Et).coalesce()
    Ath = torch.sparse.mm(EtT, Eh).coalesce()

    hh_edges = torch.cat([Ahh.indices().T, torch.zeros(Ahh.indices().T.shape[0], 1, dtype=torch.long).fill_(0)], dim=1)  # head to head
    tt_edges = torch.cat([Att.indices().T, torch.zeros(Att.indices().T.shape[0], 1, dtype=torch.long).fill_(1)], dim=1)  # tail to tail
    ht_edges = torch.cat([Aht.indices().T, torch.zeros(Aht.indices().T.shape[0], 1, dtype=torch.long).fill_(2)], dim=1)  # head to tail
    th_edges = torch.cat([Ath.indices().T, torch.zeros(Ath.indices().T.shape[0], 1, dtype=torch.long).fill_(3)], dim=1)  # tail to head
    
    rel_graph = Data(
        edge_index=torch.cat([hh_edges[:, [0, 1]].T, tt_edges[:, [0, 1]].T, ht_edges[:, [0, 1]].T, th_edges[:, [0, 1]].T], dim=1), 
        edge_type=torch.cat([hh_edges[:, 2], tt_edges[:, 2], ht_edges[:, 2], th_edges[:, 2]], dim=0),
        num_nodes=num_rels, 
        num_relations=4
    )

    graph.relation_graph = rel_graph
    return graph








class EntityNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1,embedd_out=False, **kwargs):

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
        feature_dim = input_dim + (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1])
        self.mlp = nn.Sequential()
        mlp = []
        self.embedd_out = embedd_out
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        if not embedd_out:
            mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)
 
    
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

    def forward(self, data, relation_representations, batch):
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
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        # here Im supposed to have head with all the same indices for every query
        # IN TRAINING, I GET AN ERROR IN H_INDEX, BECAUSE I SHOULD HAVE ONLY HEAD OR TAIL CORRUPTION, BUT IF I HAVE THEM MIXED, I GET AN ERROR. I NEED TO ADAPT THE FUNCTION
        #  TO PUT ALL THE NEGATIVES IN THE TAIL SIDE
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations of the head entities 
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        # (batch_size, num_negative + 1, dim)  
        embedd = self.mlp(feature)
        return embedd







class Ultra(nn.Module):

    def __init__(self):
        # kept that because super Ultra sounds cool
        super(Ultra, self).__init__()
        self.relation_model = RelNBFNet(input_dim=64, hidden_dims=[64, 64, 64, 64, 64, 64], message_func='distmult', aggregate_func='sum', short_cut=True, layer_norm=True)
        self.entity_model = EntityNBFNet(input_dim=64, hidden_dims=[64, 64, 64, 64, 64, 64], message_func='distmult', aggregate_func='sum', short_cut=True, layer_norm=True,embedd_out=True)


    def split_head_tail_negatives(self, batch, relation_representations):
        '''
        Problem: In the same list of corrptions, I cannot have positives and negatives, so I need to split it in two queries
        From batch:[n_queries,n_corruptions,3] I output [n_queries+splits,n_corruptions,3] where the first dim is for head and the second for tail corruptions
        Given a batch of queries, check for every query, check for each corruption if it is a head or tail corruption. 
            - Initialize both dimensions to -1
            - If it is a head corruption, put it in the first dim, if it is a tail corruption, put it in the second dim.
            - If there are head and tail corruptions, put them in the first and second dim respectively

        return:[n_queries+splits,n_corruptions,3], where the first dim is for head and the second for tail corruptions. Usually, n_queries+splits = 2*n_queries    

        Comment: in training it only does corruptions to tail, and in testing it does corruptions to head and tail, but in a separate array. In our framework, during training,
          both head and tail corruptions are in the same array, so I need to split them.

        Bonus: if I increate the queries, I have to adjust the relation representations
        '''
        n_queries, n_corruptions, _ = batch.shape
        original_head = batch[:, 0, 0]  # [n_queries]
        negative_heads = batch[:, :, 0]  # [n_queries, n_corruptions]
        is_head = (original_head[:, None] == negative_heads)  # bool: [n_queries, n_corruptions]

        # Precompute the conditions for which no modifications are needed
        no_modifications_needed = torch.all(is_head, dim=1) | torch.all(~is_head, dim=1)

        # Now, what I do is, for each query, and if is_head of that query isn't all true or all false,
        # I go through every corruption, and if it is true, I leave it as it is, otherwise, I flip it by
        # inverting the head and tail, and for the relation I add the number of relations
        new_queries = []
        new_relation_representation = []

        for i in range(n_queries):
            if no_modifications_needed[i]:
                new_queries.append(batch[i])
                new_relation_representation.append(relation_representations[i])
            else:
                heads_pos = [batch[i][0].unsqueeze(0)]  # Add the positive triple
                tails_pos = [batch[i][0].unsqueeze(0)]  # Add the positive triple
                for j in range(1, n_corruptions):  # for every corruption
                    if is_head[i][j]:
                        heads_pos.append(batch[i][j].unsqueeze(0))
                    else:
                        tails_pos.append(batch[i][j].unsqueeze(0))
                if len(heads_pos) > 1:  # if there are tail corruptions
                    new_queries.append(torch.cat(heads_pos, dim=0))
                    new_relation_representation.append(relation_representations[i])
                if len(tails_pos) > 1:  # if there are head corruptions
                    new_queries.append(torch.cat(tails_pos, dim=0))
                    new_relation_representation.append(relation_representations[i])
        new_queries = torch.stack(new_queries)
        relation_representations = torch.stack(new_relation_representation)
        return new_queries, relation_representations
    
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
                entity_embeddings[domain.name].append(pytorch_to_tf(entity_representations[i]))
        
        # convert to tf tensors
        for key in entity_embeddings.keys():
            entity_embeddings[key] = tf.convert_to_tensor(entity_embeddings[key])

        return entity_embeddings
    
    def split_corruptions(self, batch):
        '''
        Given a batch, split it in subbatches with the same number of corruptions in each subbatch. In this way it is easy to convert it to a pytorch version.
        '''
        n_queries = batch.shape[0]
        subbatches = {}
        for i in range(n_queries):
            n_corruptions = len(batch[i])
            if n_corruptions not in subbatches:
                subbatches[n_corruptions] = []
            subbatches[n_corruptions].append([batch[i]])
        return subbatches

    def forward(self, data, batch):
        
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs

        # Batch: [16=batch_size,1594,3]=[16 heads(tails), 1594 is, for each head(tail), the number of negative tails(heads), 3 is indeces full negative triple(h, t, r)]
        # For all the heads(tails) in the batch, take the first element of the negatives(for all negatives, the relation is the same), and the relation index
        # query_rels: torch.Size([16]) 
        # batch = tf_to_pytorch(batch,type='int') 

        # if it is a tuple, convert it to numpy, if batch is in numpy format, leave it as it is. If it is a symbolic tensor, leave it as it is
        print('before: type(batch)',type(batch),type(batch[0]),type(batch[0][0]),type(batch[0][0][0]))
        if isinstance(batch, tuple):
            batch = np.array(batch)
        elif isinstance(batch, np.ndarray):
            pass
        elif isinstance(batch, torch.Tensor) or isinstance(batch, tf.Tensor):
            batch = batch.detach().numpy()
        else:
            batch = batch.numpy()
        print('after: type(batch)',type(batch),type(batch[0]),type(batch[0][0]),type(batch[0][0][0]))

        batches = self.split_corruptions(batch)
        for key in batches.keys():
            batches[key] = np.concatenate(batches[key], axis=0)
            batches[key] = torch.tensor(batches[key], dtype=torch.int64)

        # For each batch, get the relation representations
        all_relation_representations = []
        all_entity_representations = []
        for key,batch in batches.items():
            query_rels = batch[:, 0, 2]

            # For each query, get a representation of all the relations of dim 64
            # relation_representations:  torch.Size([16, 360, 64])=(queries,n_relationsx2,dim_embedd)
            relation_representations = self.relation_model(data.relation_graph, query=query_rels)   
            # Given the batch [16,1594,3], do a prediction, for each of the 16 heads(tails) in the queries, of all possible tails(heads) candidates
            # score:  torch.Size([16, 1594])


            # Instead of an array of [n_queries, n_corruptions, 3], I get [n_queries*2, n_corruptions/2, 3] in training, where for each q, half of the corrup. are negatives (head corr) and the other half are tail corruptions
            # I put them as a two different queries, so that I can pass them to the entity model
            batch,relation_representations = self.split_head_tail_negatives(batch,relation_representations)
            # It may be that the queries have indices of entities greater than the number of entities in the graph, so I need to filter them. SHUOLD NOT HAPPEN!!!
            # batch = torch.tensor(batch, dtype=torch.int64)
            entity_representations = self.entity_model(data, relation_representations, batch) # [16,1594,64] = [batch_size, num_negatives, embedd_size]
            # PLEASE make sure that in entity_representations:[1,354=n_nodes,64], the first node correspond to the first index and so on
            all_relation_representations.append(relation_representations)
            all_entity_representations.append(entity_representations)


        # By now, as a temporary solution, I will take the average of hte relative embeddings of each entity
        entity_representations = torch.mean(entity_representations, dim=0)
        relation_representations = relation_representations[:,data.num_relations//2:,:] # Dont take the inverse relations
        relation_representations = torch.mean(relation_representations, dim=0)
        # Here, if the entities have different domains, I put them in a dict with different domains
        entity_representations = self.convert_to_domain(entity_representations,data.fol)

        # convert them to tf tensors
        # relation_representations = tf.convert_to_tensor(relation_representations.detach().numpy())
        relation_representations = pytorch_to_tf(relation_representations)
        # for key in entity_representations.keys():
        #     entity_representations[key] = tf.convert_to_tensor(entity_representations[key])

        return entity_representations,relation_representations 



 




