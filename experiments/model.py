from keras import Model
from keras.layers import Dense 
import tensorflow as tf
from keras_ns.nn.constant_embedding import *
from keras_ns.nn.reasoning import *
from keras_ns.nn.kge import KGEFactory 
from keras_ns.logic import FOL, Rule
from typing import Dict, List
from keras_ns.logic.semantics import GodelTNorm
from typing import Dict, List

import torch


class KGEModel(Model):

    def __init__(self, fol:FOL,
                 kge: str,
                 kge_regularization: float,
                 constant_embedding_size: int,
                 predicate_embedding_size: int,
                 kge_atom_embedding_size: int,
                 kge_dropout_rate: float,
                 num_adaptive_constants: int=0,
                 use_ultra: bool=False,
                 device: str = 'cpu'):
        super().__init__()
        self.fol = fol
        self.predicate_index_tensor = tf.constant(
            [i for i in range(len(self.fol.predicates))], dtype=tf.int32)
        
        # CONSTANT AND PREDICATE EMBEDDINGS
        self.use_ultra = use_ultra
        if not self.use_ultra:
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
            
            # self.constant_embedder = ConstantEmbeddingsSparse(
            #     domains=fol.domains,
            #     constant_embedding_sizes_per_domain={
            #         domain.name: constant_embedding_size
            #         for domain in fol.domains},
            #     regularization=kge_regularization)
        else: 
            # To adapt the size of the embeddings given by ultra
            self.predicate_projection = Sequential([
                Dense(128, activation='relu'),
                Dense(150, activation='relu'),
                Dense(200, activation='relu')])
            
            self.constant_projection = Sequential([
                Dense(128, activation='relu'),
                Dense(150, activation='relu'),
                Dense(200, activation='relu')])
        
        if num_adaptive_constants > 0:
            self.adaptive_constant_embedder = AdaptiveConstantEmbeddings(
                domains=fol.domains,
                constant_embedder=self.constant_embedder,
                constant_embedding_size=constant_embedding_size,
                num_adaptive_constants=num_adaptive_constants)
        else:
            self.adaptive_constant_embedder = None

        # ATOM EMBEDDINGS
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
        '''For A_predicates, take the emebdding representation of the predicates and the constants and create the triplets for the KGE model.
        For instance, if I have a predicate with 3 grounded atoms, I will repeat the embedding of that predicate 3 times, and put it with the 
        embeddings of the constants for each grounded atom
        - output: 
            predicate_embeddings_per_triplets: [n_predicates, n_atoms/grounding per predicate, embed_size_predicate]
            constant_embeddings_for_triplets: [n_atoms,2=n_domains,embed_size_constant]'''
            
        

        for p,indices in A_predicates.items():
            idx = self.fol.name2predicate_idx[p]
            # Repeat the predicate embedding for each atom in the predicate
            p_embeddings = tf.expand_dims(predicate_embeddings[idx], axis=0)  # [1,1,200]=[1,1, embed_size_predicate]
            predicate_embeddings_per_triplets.append(tf.repeat(p_embeddings, tf.shape(indices)[0], axis=0))  # [1,1918,200]=[1,n_groundings for that predicate, embed_size_predicate]
        predicate_embeddings_per_triplets = tf.concat(predicate_embeddings_per_triplets,axis=0) # shape=[n_predicates, n_groundings for that predicate, embed_size_predicate]

        constant_embeddings_for_triplets = []
        for p,constant_idx in A_predicates.items():
            # tf.print('\npredicate',p)
            # print('predicate',p,'constant_idx',constant_idx.shape,'constant_idx',constant_idx)
            constant_idx = tf.cast(constant_idx, tf.int32) # all the groundings idx for that predicate
            predicate = self.fol.name2predicate[p]
            one_predicate_constant_embeddings = []
            # Here, for each domain of the predicate (LocInSR(subregion,region)->for subregion), I get the embeddings of the constants that
            # are grounded in that domain. If LocInSR has 58 groundings/atoms, I will get the representation of the subregion constants in 
            # those atoms(58,200). I do the same for the domain region, so I get a tensor of shape [58,2,200] for LocIn where 2 is the arity of the predicate. 
            for i,domain in enumerate(predicate.domains):
                # print('domain',domain.name)
                # print('constant_embeddings of domain',constant_embeddings[domain.name].shape)
                # print('constant_idx',constant_idx.shape,constant_idx)
                # print('constant_idx[...,i]',constant_idx[...,i].shape,constant_idx[...,i])
                # Take the embeddings of the constants in the domain i of the predicate
                # tf.print('constant_idx',constant_idx[...,i].shape,constant_idx[...,i],summarize=-1) if domain.name == 'countries' else None
                # tf.print('cte_emb',constant_embeddings[domain.name].shape,constant_embeddings[domain.name][:,5],summarize=-1) if domain.name == 'countries' else None
                # If I have A_pred=[country,region]=[[1,2],[3,4],...], I get for country: [1,3] which are local! they're the pos of the ctes in X_domain
                # In X_domain, in pos i I have the global idx of that cte (which has been created to create the embedds)
                constants = tf.gather(constant_embeddings[domain.name],
                                      constant_idx[..., i], axis=0) # constant_idx[..., i] takes the idx of the constants for that domain (in predicate p)
                # tf.print('constants',constants.shape,constants[:,5],summarize=-1) if domain.name == 'countries' else None
                one_predicate_constant_embeddings.append(constants)
            # shape (predicate_batch_size, predicate_arity, constant_embedding_size)
            one_predicate_constant_embeddings = tf.stack(one_predicate_constant_embeddings,axis=-2)
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

    def call(self, inputs,
            embeddings=None # For ULTRA
            ):
        '''
        X_domains type is Dict[str, inputs]
        A_predicate: Dict[predicate_name, List[Tuple[Index1, ..., IndexN]]]
        For x_domains, I get each domain value (country,region...) represented by a index
        For A_predicates, I get the predicate name and the constant indices for each grounding
        '''
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
            if not self.use_ultra:
                constant_embeddings = self.constant_embedder(X_domains) # For the constant embedds, I always need global idx to get consistent embedds every batch
        
        if not self.use_ultra:
            print('USING KGE')
            predicate_embeddings = self.predicate_embedder(self.predicate_index_tensor) # Embedds for every pred in fol (global idx)
        else: 
            print('USING ULTRA WITH KGE')
            (constant_embeddings, predicate_embeddings) = embeddings
            # Project embeddings to the new size
            predicate_embeddings = self.predicate_projection(predicate_embeddings)
            constant_embeddings = {k: self.constant_projection(v) for k, v in constant_embeddings.items()} 

        # Given the embedds of the constants and the predicates, I create the triplets with the embeddings of the atoms and the predicates. 
        # A_predicates indicates the indeces of the constants for each grounding of the predicate, i.e., the queries
        predicate_embeddings_per_triplets, constant_embeddings_for_triplets = \
            self.create_triplets(constant_embeddings, predicate_embeddings, A_predicates) 
        
        # Given the triplets with their embeddings obtained in create_triplets, I get the embeddings of the atoms with e.g. Transe
        atom_embeddings = self.kge_embedder((predicate_embeddings_per_triplets,
                                             constant_embeddings_for_triplets))
        return atom_embeddings
    


class ULTRA_bridge(Model):

    def __init__(self):
        super().__init__()
        
        # To adapt the size of the embeddings given by ultra
        self.atom_projection = Sequential([
            Dense(128, activation='relu'),
            Dense(114, activation='relu'),
            Dense(100, activation='relu'),
            Dense(100, activation='relu'),])

        # Define the output layer as a method
        self.output_layer = self._output_layer

    def _output_layer(self, inputs):
        outputs = tf.reduce_sum(inputs, axis=-1)
        outputs = tf.nn.sigmoid(outputs)
        return outputs

    def call(self,embeddings):
        # Project embeddings to the new size
        atom_embeddings = self.atom_projection(embeddings)  
        return atom_embeddings


class CollectiveModel(Model):

    def __init__(self,
                 fol: FOL,
                 rules: List[Rule],
                 *,  # all named after this point
                 use_ultra: bool,
                 use_ultra_with_kge: bool,
                 kge: str,
                 kge_regularization: float,
                 constant_embedding_size: int,
                 predicate_embedding_size: int,
                 kge_atom_embedding_size: int,
                 kge_dropout_rate: float,
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
                 cdcr_num_formulas: int,
                 device: str = 'cpu'):
        super().__init__()

        self.testing = False
        self.reasoner_depth = reasoner_depth
        # Reasoning depth currently used, this can differ from  self.reasoner_depth during multi-stage learning (like if pretraining the KGEs).
        self.enabled_reasoner_depth = reasoner_depth
        self.resnet = resnet
        self.logic = GodelTNorm()
        self.use_ultra = use_ultra
        self.use_ultra_with_kge = use_ultra_with_kge
        if not self.use_ultra:
            self.kge_model = KGEModel(fol, kge,
                                    kge_regularization,
                                    constant_embedding_size,
                                    predicate_embedding_size,
                                    kge_atom_embedding_size,
                                    kge_dropout_rate,
                                    num_adaptive_constants,
                                    device='cpu',
                                    use_ultra=self.use_ultra_with_kge)
            self.output_layer = self.kge_model.output_layer # CONCEPT LAYER
        else:
            self.ULTRA_bridge = ULTRA_bridge()
            self.output_layer = self.ULTRA_bridge.output_layer
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

    def test_mode(self, dataset_type, mode=False):
        self.testing = mode
        self.dataset_type = dataset_type

    def call(self, inputs, training=False, *args, **kwargs):
        '''
        X_domains type is Dict[str, tensor[constant_indices_in_domain]]
        A_predicate: Dict[predicate_name, List[Tuple[Index1, ..., IndexN]]]
                     e.g. mapping predicate_name -> tensor [num_groundings, arity]
                     with constant indices for each grounding.
        '''

        if self._explain_mode:
            # No explanations are posible when reasoning is disabled.
            assert self.reasoning is not None
            # Check that we are using an explainable model.
            assert self.model_name == 'dcr' or self.model_name == 'cdcr'

        (X_domains, A_predicates, A_rules, Q, embeddings) = inputs
        
        if self.use_ultra:
            print('USING ULTRA')
            atom_embeddings = self.ULTRA_bridge(embeddings)
        else:
            atom_embeddings = self.kge_model((X_domains, A_predicates),embeddings=embeddings)
 
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
        task_output = tf.gather(params=tf.squeeze(task_output, -1), indices=Q)
        concept_output = tf.gather(params=tf.squeeze(concept_output, -1),
                                   indices=Q)
        if self.resnet and self.reasoning is not None:
            task_output = self.logic.disj_pair(task_output, concept_output)

        if self._explain_mode:
            return concept_output, task_output, explanations
        else:
            return {'concept':concept_output, 'task':task_output}