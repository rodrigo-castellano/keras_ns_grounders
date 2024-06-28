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
        # self.mlp = nn.Sequential(*mlp)
        self.mlp_embedd = nn.Sequential(*mlp)
        if get_scores:
            mlp.append(nn.Linear(feature_dim, 1))
        self.mlp_score = nn.Sequential(*mlp)
    
    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        batch_size = len(r_index)
        # initialize queries (relation types of the given triples)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index] #[n_nodes=tail node of each query,embedd_size of each node]
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

    def forward(self, data, relation_representations, batch, atom_repr=False):
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
        # print('feature:',feature.shape)

        # if atom_repr:                                                                                                                                                                                                                     
        #     # Get the tail indices [19,num_negative+1], then unesqueeze it to [19,num_negative+1,1], then expand (repeat) it to [19,num_negative+1,64] 
        #     index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        #     # here, for every query (pos and neg), I get the tail embeddings
        #     feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)
        #     # From these embeddings of the pos and negatives queries, Im just interested in the postive query (self representation)
        #     feature = feature[:, :1, :]

        embedd = self.mlp_embedd(feature)
        scores = self.mlp_score(feature) if self.get_scores else None
        # print('embedd:',embedd.shape)
        # print('scores:',scores.shape)
        return embedd, scores







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


    def forward(self, data, batch, atom_repr=False):
        
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs

        # print('batch:',batch.shape)
        query_rels = batch[:, 0, 2]

        # relation_representations:  torch.Size([16, 360, 64])=(queries,n_relationsx2,dim_embedd)
        relation_representations = self.relation_model(data.relation_graph, query=query_rels)
        # print('relation_representations:',relation_representations.shape)
  
        entity_representations, scores = self.entity_model(data, relation_representations, batch,atom_repr=atom_repr) # [16,1594,64] = [batch_size, num_negatives, embedd_size]
        # PLEASE make sure that in entity_representations:[1,354=n_nodes,64], the first node correspond to the first index and so on
        # print('entity_representations:',entity_representations.shape)

        atom_representations = entity_representations

        # OPTION 2, GET THE ENTITIES AND RELATIONS REPRESENTATIONS


        return scores, atom_representations
        
 




