from keras import Model
from keras.layers import Dense, Layer
import keras_ns as ns
import tensorflow as tf
from keras_ns.nn.constant_embedding import *
from keras_ns.nn.reasoning import *
from keras_ns.nn.kge import KGEFactory, KGELayer
from keras_ns.logic import FOL, Rule
from typing import Dict, List
from keras_ns.logic.semantics import GodelTNorm
from typing import Dict, List, Union
import tensorflow_probability as tfp

import torch
from torch import nn
from ULTRA.ultra.models import Ultra,RelNBFNet,BaseNBFNet,EntityNBFNet
from ULTRA.ultra import layers



class KGEModel(Model):

    def __init__(self, fol:FOL,
                 kge: str,
                 kge_regularization: float,
                 constant_embedding_size: int,
                 predicate_embedding_size: int,
                 kge_atom_embedding_size: int,
                 kge_dropout_rate: float,
                 num_adaptive_constants: int=0):
        super().__init__()
        self.fol = fol
        self.predicate_index_tensor = tf.constant(
            [i for i in range(len(self.fol.predicates))], dtype=tf.int32)
        self.predicate_embedder = PredicateEmbeddings(
            fol.predicates,
            predicate_embedding_size,
            regularization=kge_regularization)
        self.constant_embedder = ConstantEmbeddings(
            domains=fol.domains,
            constant_embedding_sizes_per_domain={
                domain.name: constant_embedding_size
                for domain in fol.domains},
            regularization=kge_regularization)
        if num_adaptive_constants > 0:
            self.adaptive_constant_embedder = AdaptiveConstantEmbeddings(
                domains=fol.domains,
                constant_embedder=self.constant_embedder,
                constant_embedding_size=constant_embedding_size,
                num_adaptive_constants=num_adaptive_constants)
        else:
            self.adaptive_constant_embedder = None

        self.kge_embedder, self.output_layer = KGEFactory(
            name=kge,
            atom_embedding_size=kge_atom_embedding_size,
            relation_embedding_size=kge_atom_embedding_size,
            regularization=kge_regularization,
            dropout_rate=kge_dropout_rate)
        assert self.kge_embedder is not None


    def create_triplets(self,
                        constant_embeddings: Dict[str, tf.Tensor],
                        predicate_embeddings: tf.Tensor,
                        A_predicates: Dict[str, tf.Tensor]):
        predicate_embeddings_per_triplets = []
        # What I do here is taking the embedding of each predicate and repeating it for each grounded atom of that predicate
        # For example, if I have a predicate with 3 grounded atoms, I will repeat the embedding of that predicate 3 times, 
        # resulting in a tensor of shape [3,200] where 200 is the embedding size of the predicate.
        # I do this for each predicate in the A_predicates dictionary, so I get [n_predicates, n_atoms/grounding per predicate, embed_size_predicate]
        for p,indices in A_predicates.items():
            idx = self.fol.name2predicate_idx[p]
            p_embeddings = tf.expand_dims(predicate_embeddings[idx], axis=0)  # 1E
            # shape: [1,1918,200]=[1,number of grounded atoms for that predicate, embed size of the predicate]
            predicate_embeddings_per_triplets.append(tf.repeat(p_embeddings, tf.shape(indices)[0], axis=0))  #PE
        # shape=[n_predicates, n_atoms/grounding per predicate, embed_size_predicate]
        predicate_embeddings_per_triplets = tf.concat(predicate_embeddings_per_triplets,
                                                      axis=0)

        constant_embeddings_for_triplets = []
        for p,constant_idx in A_predicates.items():
            constant_idx = tf.cast(constant_idx, tf.int32)
            predicate = self.fol.name2predicate[p]
            one_predicate_constant_embeddings = []
            # Here, for each domain of the predicate (LocInSR(subregion,region)->for subregion), I get the embeddings of the constants that
            # are grounded in that domain. If LocInSR has 58 groundings/atoms, I will get the representation of the subregion constants in 
            # those atoms(58,200). I do the same for the region domain, so I get a tensor of shape [58,2,200] for LocIn where 2 is the arity of the predicate. 
            for i,domain in enumerate(predicate.domains):
                constants = tf.gather(constant_embeddings[domain.name],
                                      constant_idx[..., i], axis=0)
                one_predicate_constant_embeddings.append(constants)
            # shape (predicate_batch_size, predicate_arity, constant_embedding_size)
            one_predicate_constant_embeddings = tf.stack(one_predicate_constant_embeddings,
                                                         axis=-2)
            constant_embeddings_for_triplets.append(one_predicate_constant_embeddings)
        # For all the queries, I have divided them by predicates. Once I have, for each predicate, the embeddings of the constants, i.e., 
        # for LocInSR I have 58 atoms -> (58,2,200), for NeighOf .... I concatenate them to get a tensor of shape [58+..,2,200] = [3889,2,200]
        constant_embeddings_for_triplets = tf.concat(constant_embeddings_for_triplets,
                                                     axis=0)
        tf.debugging.assert_equal(tf.shape(predicate_embeddings_per_triplets)[0],
                                  tf.shape(constant_embeddings_for_triplets)[0])
        # Shape TE, T2E with T number of triplets.
        # At the end I get for both predicates and embeddings a tensor of shape [n_atoms,ctes_in_atoms(arity),embed_size_predicate] and [n_atoms,2,embed_size_constant]
        return predicate_embeddings_per_triplets, constant_embeddings_for_triplets

    def call(self, inputs):
        # X_domains type is Dict[str, inputs]
        # A_predicate: Dict[predicate_name, List[Tuple[Index1, ..., IndexN]]]
        # For x_domains, I get each domain value (country,region...) represented by a index
        # For A_predicates, I get the predicate name and the constant indices for each grounding
        (X_domains, A_predicates) = inputs 

        if self.adaptive_constant_embedder is not None:
            # Create a mask to fix the values that are not in the domain.
            X_domains_fixed_mask = {
                name:tf.where(x < len(self.fol.name2domain[name].constants),
                              True, False) for name,x in X_domains.items()}
            # Set to 0 the values that are not in the domain.
            X_domains_fixed = {
                name:tf.where(X_domains_fixed_mask[name], x, 0)
                for name,x in X_domains.items()}
            # Get the embeddings for the fixed values and the adaptive values.
            constant_embeddings_fixed = self.constant_embedder(X_domains_fixed)
            constant_embeddings_adaptive = self.adaptive_constant_embedder(
                X_domains)
            constant_embeddings = {
                name:tf.where(
                    # Expand dim to broadcast to the embeddings size.
                    tf.expand_dims(X_domains_fixed_mask[name], axis=-1),
                    constant_embeddings_fixed[name],
                    constant_embeddings_adaptive[name])
                for name in X_domains.keys()}
        else:
            constant_embeddings = self.constant_embedder(X_domains)

        predicate_embeddings = self.predicate_embedder(self.predicate_index_tensor)
        # Given the embedds of the constants and the predicates, I create the triplets with the embeddings of the atoms and the predicates. 
        # A_predicates indicates the indeces of the constants for each grounding of the predicate, i.e., the queries
        predicate_embeddings_per_triplets, constant_embeddings_for_triplets = \
            self.create_triplets(constant_embeddings, predicate_embeddings, A_predicates)
        # tf.print('\n\npredicate_embeddings_per_triplets',predicate_embeddings_per_triplets.shape)
        # tf.print('constant_embeddings_for_triplets',constant_embeddings_for_triplets.shape)
        # Given the triplets with their embeddings obtained in create_triplets, I get the embeddings of the atoms with e.g. Transe
        atom_embeddings = self.kge_embedder((predicate_embeddings_per_triplets,
                                             constant_embeddings_for_triplets))
        # Shape TE
        return atom_embeddings




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
        # print('self.num_mlp_layers',self.num_mlp_layers,'input_dim',input_dim,'hidden_dims',hidden_dims, 'self.concat_hidden',self.concat_hidden,'sum(hidden_dims)',sum(hidden_dims),'hidden_dims[-1]',hidden_dims[-1])
        feature_dim = input_dim + (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1])
        self.mlp = nn.Sequential()
        mlp = []
        self.embedd_out = embedd_out
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        if embedd_out:
            mlp.append(nn.Linear(feature_dim, feature_dim//2))
        else:
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
            # print('------------------------------------')
            # print('data.num_relations', data.num_relations)
            # print('data.num_edges', data.num_edges)
            # print('boundary', boundary.shape)
            # print('layer_input', layer_input.shape)
            # print('size', size)
            # print('edge_weight', edge_weight.shape)
            # print('data.edge_index', data.edge_index.shape)
            # print('data.edge_type', data.edge_type.shape)
            # print('query', query.shape)
            # print('------------------------------------')
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
            # print('hiddens',hiddens.shape, 'node_query',node_query.shape)
            output = torch.cat(hiddens + [node_query], dim=-1)
            # print('output',output.shape)
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
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations 
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1]) 
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim) 
        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        if self.embedd_out:
            score = self.mlp(feature)
            return score
        else:
            score = self.mlp(feature).squeeze(-1)
            return score.view(shape)



class Ultra(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_cfg):
        # kept that because super Ultra sounds cool
        super(Ultra, self).__init__()
        self.relation_model = RelNBFNet(input_dim=64, hidden_dims=[64, 64, 64, 64, 64, 64], message_func='distmult', aggregate_func='sum', short_cut=True, layer_norm=True)
        self. entity_model = EntityNBFNet(input_dim=64, hidden_dims=[64, 64, 64, 64, 64, 64], message_func='distmult', aggregate_func='sum', short_cut=True, layer_norm=True,embedd_out=True)

        
    def forward(self, data, batch):
        
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs

        # Batch: [16=batch_size,1594,3]=[16 heads(tails), 1594 is, for each head(tail), the number of negative tails(heads), 3 is indeces full negative triple(h, t, r)]
        # For all the heads(tails) in the batch, take the first element of the negatives(for all negatives, the relation is the same), and the relation index
        # query_rels: torch.Size([16]) 
        # query_rels = batch[:, 0, 2]
        query_rels = []
        for i in range(len(batch)):
            query_rels.append(batch[i][0][2])
        query_rels = torch.tensor(query_rels)
        query_rels = query_rels.to(torch.int64)
        # convert edge_index and edge_type to torch tensor, from (1159, 2) to (2, 1159) and (1159) to (1159)
        data.edge_index = torch.tensor(data.edge_index, dtype=torch.long).t().contiguous()
        data.edge_type = torch.tensor(data.edge_type, dtype=torch.long)

        # For each query, get a representation of all the relations of dim 64
        # relation_representations:  torch.Size([16, 360, 64])=(queries,n_relationsx2,dim_embedd)
        relation_representations = self.relation_model(data, query=query_rels)        
        # print('\n Relation representations obtained', relation_representations.shape,'\n')
        # CHECK WHY I GET AN EMBEDD OF SIZE 128 INSTEAD OF 64

        # Given the batch [16,1594,3], do a prediction, for each of the 16 heads(tails) in the queries, of all possible tails(heads) candidates
        # score:  torch.Size([16, 1594])

        # For the batch, I need to distinguish corruptions from heads and tails, because there are a different number of corruptions, e.g., for heads, I have 1594 corruptions, 
        # for tails, I have 564 corruptions, [16,1594,3] and [16,564,3]. I need to split batch in as many batches as different number of second dimensions in batch I have, otherwise 
        # I cannot pass it to entity_model
        batches = {}
        for i in range(len(batch)): 
            if len(batch[i]) not in batches:
                batches[len(batch[i])] = []
            batches[len(batch[i])].append(batch[i])

        # Convert each batch to a torch tensor
        for key in batches.keys():
            batches[key] = torch.tensor(np.array(batches[key])).long()
        # For each batch, call the entity model. Take into account that I need to pass the relation representations for each query. If there are 50 queries, there are 50 relation representations
        #  ,e.g. [50,267,64] for 267 relations and 64 embedding size. If to the entity model I pass 10 queries and in the next loop I pass 40 queries, I need to do the same with the relation representations will be the same
        rel_start = 0
        rel_end = 0
        entity_representations = []
        for key in batches.keys():
            # print( '\nnumber of queries', batches[key].shape[0],', number of negatives', batches[key].shape[1],',', batches[key].shape)
            rel_end +=  len(batches[key])
            entity_representations.append(self.entity_model(data, relation_representations[rel_start:rel_end,:,:], batches[key],))
            rel_start += len(batches[key])
            # print('entity_representations',entity_representations.shape)
        # Return a list with each element being a query with [num_negatives, embedd]
        return entity_representations,relation_representations




class ultra_model():

    def __init__(self, 
                 fol:FOL,
                 kge: str,
                 kge_regularization: float,
                 constant_embedding_size: int,
                 predicate_embedding_size: int,
                 kge_atom_embedding_size: int,
                 kge_dropout_rate: float):
        
        self.fol = fol
        self.predicate_index_tensor = tf.constant(
            [i for i in range(len(self.fol.predicates))], dtype=tf.int32)
        
        self.Ultra = Ultra(rel_model_cfg=RelNBFNet,entity_model_cfg=EntityNBFNet)
        self.Ultra = self.Ultra.to('cpu')

        self.output_layer = KGEFactory(
            name=kge,
            atom_embedding_size=kge_atom_embedding_size,
            relation_embedding_size=kge_atom_embedding_size,
            regularization=kge_regularization,
            dropout_rate=kge_dropout_rate)
        assert self.output_layer is not None


    def call(self, inputs, data_gen):
        # X_domains type is Dict[str, inputs]
        # A_predicate: Dict[predicate_name, List[Tuple[Index1, ..., IndexN]]]
        # For x_domains, I get each domain value (country,region...) represented by a index
        # For A_predicates, I get the predicate name and the constant indices for each grounding
        (X_domains, A_predicates) = inputs      
        entity_representations,relation_representations = self.Ultra(data_gen, data_gen.Q_triplets)
        print('relation_representations',relation_representations.shape)
        for i in range(len(entity_representations)):
            print('entity_representations',entity_representations[i].shape)
        print(paco)
        # NOW, ONCE I HAVE  entity_representations torch.Size([6, 199, 64]), relation_representations torch.Size([50, 267, 64]), I NEED TO TURN IT TO TRIPLET FORM (3889, 2, 200),(3889, 200) 
        # 3389 is the number of groundings (it comes from A_predicates, summing all groundings for every predicate). The problem is how to pass from relative to global indeces
        # In test should be fine if I have batch size of 1, but I should adapt it to any batch size, and also for training

        # EVENTUALLY I NEED TO GET THE EMBEDDINGS FOR ALL THE 


        # For every query in A_predicates, produce the corruptions. batch shape: (bs, 1+num_negs, 3)
        # t_batch, h_batch = tasks.all_negative(test_data, batch)
        # t_pred = model(test_data, t_batch)
        # h_pred = model(test_data, h_batch)
        # I HAVE TO MODIFY THE MODEL TO OUTPUT THE EMBEDDINGS OF THE ENTITIES AND RELATIONS INSTEAD OF THE SCORES
    
        # constant_embeddings = entities embeddings from ultra
        # predicate_embeddings = relations embeddings from ultra 

        # predicate_embeddings_per_triplets, constant_embeddings_for_triplets = \
        #     self.create_triplets(constant_embeddings, predicate_embeddings, A_predicates)
        # atom_embeddings = self.kge_embedder((predicate_embeddings_per_triplets,
        #                                      constant_embeddings_for_triplets))
        # return atom_embeddings

 


class CollectiveModel(Model):

    def __init__(self,
                 data_gen_train,
                 data_gen_test,
                 fol: FOL,
                 rules: List[Rule],
                 *,  # all named after this point
                 use_ultra: bool, 
                 kge: str,
                 kge_regularization: float,
                 constant_embedding_size: int,
                 predicate_embedding_size: int,
                 kge_atom_embedding_size: int,
                 kge_dropout_rate: float,
                 # same model for all reasoning depths
                 reasoner_atom_embedding_size: int,
                 reasoner_formula_hidden_embedding_size: int,
                 reasoner_regularization: float,
                 reasoner_single_model: bool,
                 reasoner_dropout_rate: float,
                 reasoner_depth: int,
                 aggregation_type: str,
                 signed: bool,
                 temperature: float,
                 model_name: str,
                 resnet: bool,
                 filter_num_heads: int,
                 filter_activity_regularization: float,
                 num_adaptive_constants: int,
                 cdcr_use_positional_embeddings: bool,
                 cdcr_num_formulas: int):
        super().__init__()
        self.data_gen_train = data_gen_train
        self.data_gen_test = data_gen_test
        # Reasoning depth of the model structure.
        self.reasoner_depth = reasoner_depth
        # Reasoning depth currently used, this can differ from
        # self.reasoner_depth during multi-stage learning (like if
        # pretraining the KGEs).
        self.enabled_reasoner_depth = reasoner_depth
        self.resnet = resnet
        self.logic = GodelTNorm()
        self.use_ultra = use_ultra
        if self.use_ultra: 
            self.ultra_model = ultra_model(fol,kge,kge_regularization,constant_embedding_size,predicate_embedding_size,kge_atom_embedding_size,kge_dropout_rate)
            self.output_layer = self.ultra_model.output_layer
        else:   
            self.kge_model = KGEModel(fol, kge,
                                    kge_regularization,
                                    constant_embedding_size,
                                    predicate_embedding_size,
                                    kge_atom_embedding_size,
                                    kge_dropout_rate,
                                    num_adaptive_constants)
            # CONCEPT LAYER
            self.output_layer = self.kge_model.output_layer
        self.model_name = model_name



        # REASONING LAYER
        self.reasoning = None
        if reasoner_depth > 0 and len(rules) > 0:
            self.reasoning = []
            for i in range(reasoner_depth):
              if i > 0 and reasoner_single_model:
                  self.reasoning.append(self.reasoning[0])
                  continue

              if model_name == 'dcr':
                  self.reasoning.append(DCRReasoningLayer(
                      templates=rules,
                      formula_hidden_size=reasoner_formula_hidden_embedding_size,
                      aggregation_type=aggregation_type,
                      temperature=temperature,
                      signed=signed,
                      filter_num_heads=filter_num_heads,
                      regularization=reasoner_regularization,
                      dropout_rate=reasoner_dropout_rate))

              elif model_name == 'cdcr':
                  self.reasoning.append(ClusteredDCRReasoningLayer(
                      num_formulas=cdcr_num_formulas,
                      use_positional_embeddings=cdcr_use_positional_embeddings,
                      templates=rules,
                      formula_hidden_size=reasoner_formula_hidden_embedding_size,
                      aggregation_type=aggregation_type,
                      temperature=temperature,
                      signed=signed,
                      filter_num_heads=filter_num_heads,
                      regularization=reasoner_regularization,
                      dropout_rate=reasoner_dropout_rate))

              elif model_name == 'r2n':
                  self.reasoning.append(R2NReasoningLayer(
                      rules=rules,
                      formula_hidden_size=reasoner_formula_hidden_embedding_size,
                      atom_embedding_size=reasoner_atom_embedding_size,
                      aggregation_type=aggregation_type,
                      output_layer=self.output_layer,
                      regularization=reasoner_regularization,
                      dropout_rate=reasoner_dropout_rate))

              elif model_name == 'sbr':
                  self.reasoning.append(SBRReasoningLayer(
                      rules=rules,
                      aggregation_type=aggregation_type))

              elif model_name == 'gsbr':
                  self.reasoning.append(GatedSBRReasoningLayer(
                      rules=rules,
                      aggregation_type=aggregation_type,
                      regularization=reasoner_regularization))

              elif model_name == 'rnm':
                  self.reasoning.append(RNMReasoningLayer(
                      rules=rules,
                      aggregation_type=aggregation_type,
                      regularization=reasoner_regularization))

              else:
                  assert False, 'Unknown model name %s' % model_name

        self._explain_mode = False

    def explain_mode(self, mode=True):
        self._explain_mode = mode

    def call(self, inputs, training=False, *args, **kwargs):
        if self._explain_mode:
            # No explanations are posible when reasoning is disabled.
            assert self.reasoning is not None
            # Check that we are using an explainable model.
            assert self.model_name == 'dcr' or self.model_name == 'cdcr'

        # X_domains type is Dict[str, tensor[constant_indices_in_domain]]
        # A_predicate: Dict[predicate_name, List[Tuple[Index1, ..., IndexN]]]
        #              e.g. mapping predicate_name -> tensor [num_groundings, arity]
        #                   with constant indices for each grounding.
        (X_domains, A_predicates, A_rules, Q) = inputs
        if self.use_ultra: 
            if training == True:
                atom_embeddings = self.ultra_model.call((X_domains, A_predicates),self.data_gen_train)
            else:
                atom_embeddings = self.ultra_model.call((X_domains, A_predicates),self.data_gen_test)
        else: 
            atom_embeddings = self.kge_model((X_domains, A_predicates))
        tf.print('Training: ', training)
        for key in A_predicates.keys():
            tf.print('shape of A_predicates[',key,']: ', tf.shape(A_predicates[key]))
        tf.print('shape of atom_embeddings: ', tf.shape(atom_embeddings))
        tf.print('self.output_layer(atom_embeddings)',  tf.shape(self.output_layer(atom_embeddings)))
        tf.print('Q', tf.shape(Q))


        concept_output = tf.expand_dims(self.output_layer(atom_embeddings), -1)

        explanations = None
        if self.reasoning is not None:
            task_output = concept_output  # initialization
            for i in range(self.enabled_reasoner_depth):

                if self._explain_mode and i == self.enabled_reasoner_depth - 1:
                    explanations = self.reasoning[i].explain(
                        [task_output, atom_embeddings, A_rules])
                    
                task_output, atom_embeddings = self.reasoning[i]([
                    task_output, atom_embeddings, A_rules])
        else:
            task_output = tf.identity(concept_output)
        # tf.print('atom_embeddings', tf.shape(atom_embeddings))
        # tf.print('concept_output', tf.shape(concept_output), concept_output )
        # tf.print('task_output', tf.shape(task_output), task_output)
        # tf.print('tf.squeeze(task_output, -1)', tf.shape(tf.squeeze(task_output, -1)))
        # tf.print('Q', tf.shape(Q), Q)
        task_output = tf.gather(params=tf.squeeze(task_output, -1), indices=Q)
        # tf.print('task output', tf.shape(task_output))
        concept_output = tf.gather(params=tf.squeeze(concept_output, -1),
                                   indices=Q)
        if self.resnet and self.reasoning is not None:
            task_output = self.logic.disj_pair(task_output, concept_output)

        if self._explain_mode:
            return concept_output, task_output, explanations
        else:
            return {'concept':concept_output, 'task':task_output}