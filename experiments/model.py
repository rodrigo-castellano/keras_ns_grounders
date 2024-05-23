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
from ULTRA.ultra import layers
from ultra_utils import Ultra

from collections import defaultdict, OrderedDict


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
            # for key, value in constant_embeddings.items():
            #     print(key, value.shape)
        
        predicate_embeddings = self.predicate_embedder(self.predicate_index_tensor) 
        # print('predicate_embeddings', predicate_embeddings.shape)
        # print('constant_embeddings')
        # for key, value in constant_embeddings.items():
        #     print(key, value.shape)
        # Given the embedds of the constants and the predicates, I create the triplets with the embeddings of the atoms and the predicates. 
        # A_predicates indicates the indeces of the constants for each grounding of the predicate, i.e., the queries
        predicate_embeddings_per_triplets, constant_embeddings_for_triplets = \
            self.create_triplets(constant_embeddings, predicate_embeddings, A_predicates)
        # print('predicate_embeddings_per_triplets',predicate_embeddings_per_triplets.shape)
        # print('constant_embeddings_for_triplets',constant_embeddings_for_triplets.shape)
        # Given the triplets with their embeddings obtained in create_triplets, I get the embeddings of the atoms with e.g. Transe
        atom_embeddings = self.kge_embedder((predicate_embeddings_per_triplets,
                                             constant_embeddings_for_triplets))
        # print('atom_embeddings',atom_embeddings.shape)
        # Shape TE
        return atom_embeddings
    



class KGEModel_4Ultra(Model):

    def __init__(self, fol:FOL,
                 kge: str,
                 kge_regularization: float,
                 constant_embedding_size: int,
                 predicate_embedding_size: int,
                 kge_atom_embedding_size: int,
                 kge_dropout_rate: float,
                 num_adaptive_constants: int=0,
                 device: str = 'cpu'):
        super().__init__()
        self.fol = fol
        self.predicate_index_tensor = tf.constant(
            [i for i in range(len(self.fol.predicates))], dtype=tf.int32)
        
        # self.predicate_embedder = PredicateEmbeddings(
        #     fol.predicates,
        #     predicate_embedding_size,
        #     regularization=kge_regularization)
        # self.constant_embedder = ConstantEmbeddings(
        #     domains=fol.domains,
        #     constant_embedding_sizes_per_domain={
        #         domain.name: constant_embedding_size
        #         for domain in fol.domains},
        #     regularization=kge_regularization)
        
        # if num_adaptive_constants > 0:
        #     self.adaptive_constant_embedder = AdaptiveConstantEmbeddings(
        #         domains=fol.domains,
        #         constant_embedder=self.constant_embedder,
        #         constant_embedding_size=constant_embedding_size,
        #         num_adaptive_constants=num_adaptive_constants)
        # else:
        self.adaptive_constant_embedder = None

        self.Ultra = Ultra()
        self.Ultra = self.Ultra.to('cpu')

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

    def call(self, inputs,data_gen, Q, Q_global):
        # X_domains type is Dict[str, inputs]
        # A_predicate: Dict[predicate_name, List[Tuple[Index1, ..., IndexN]]]
        # For x_domains, I get each domain value (country,region...) represented by a index
        # For A_predicates, I get the predicate name and the constant indices for each grounding
        (X_domains, A_predicates) = inputs 

        # HERE USE ULTRA TO OBTAIN THE EMBEDDINGS
        constant_embeddings, predicate_embeddings = self.Ultra(data_gen, Q_global)

        # print('predicate_embeddings', predicate_embeddings.shape)
        # for key in constant_embeddings.keys():
        #     print(key, constant_embeddings[key].shape)

        # Given the embedds of the constants and the predicates, I create the triplets with the embeddings of the atoms and the predicates. 
        # A_predicates indicates the indeces of the constants for each grounding of the predicate, i.e., the queries
        predicate_embeddings_per_triplets, constant_embeddings_for_triplets = \
            self.create_triplets(constant_embeddings, predicate_embeddings, A_predicates)
        # Given the triplets with their embeddings obtained in create_triplets, I get the embeddings of the atoms with e.g. Transe
        atom_embeddings = self.kge_embedder((predicate_embeddings_per_triplets,
                                             constant_embeddings_for_triplets))
        # print('atom_embeddings',atom_embeddings.shape)
        # Shape TE
        return atom_embeddings


class CollectiveModel(Model):

    def __init__(self,
                 data_gen_train,
                 data_gen_valid,
                 data_gen_test,
                #  ultra_embeddings,
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
                 cdcr_num_formulas: int,
                 device: str = 'cpu'):
        super().__init__()

        self.testing = False
        self.data_gen_train = data_gen_train
        self.data_gen_valid = data_gen_valid
        self.data_gen_test = data_gen_test
        # self.ultra_embeddings = ultra_embeddings

        # Reasoning depth of the model structure.
        self.reasoner_depth = reasoner_depth
        # Reasoning depth currently used, this can differ from  self.reasoner_depth during multi-stage learning (like if pretraining the KGEs).
        self.enabled_reasoner_depth = reasoner_depth
        self.resnet = resnet
        self.logic = GodelTNorm()
        self.use_ultra = use_ultra
        if self.use_ultra: 
            self.ultra_model = KGEModel_4Ultra(fol,kge,kge_regularization,constant_embedding_size,predicate_embedding_size,
                kge_atom_embedding_size,kge_dropout_rate,device='cpu')
            # self.ultra_model = KGEModel_wUltra(fol, kge,
            #                         kge_regularization,
            #                         constant_embedding_size,
            #                         predicate_embedding_size,
            #                         kge_atom_embedding_size,
            #                         kge_dropout_rate,
            #                         num_adaptive_constants)
            self.output_layer = self.ultra_model.output_layer
    
        else:   
            self.kge_model = KGEModel(fol, kge,
                                    kge_regularization,
                                    constant_embedding_size,
                                    predicate_embedding_size,
                                    kge_atom_embedding_size,
                                    kge_dropout_rate,
                                    num_adaptive_constants)
            self.output_layer = self.kge_model.output_layer # CONCEPT LAYER
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

    def test_mode(self, dataset_type):
        return self.dataset_type

    def call(self, inputs, testing=False, training=False, dataset_type=None, *args, **kwargs):
        if self._explain_mode:
            # No explanations are posible when reasoning is disabled.
            assert self.reasoning is not None
            # Check that we are using an explainable model.
            assert self.model_name == 'dcr' or self.model_name == 'cdcr'


        # X_domains type is Dict[str, tensor[constant_indices_in_domain]]
        # A_predicate: Dict[predicate_name, List[Tuple[Index1, ..., IndexN]]]
        #              e.g. mapping predicate_name -> tensor [num_groundings, arity]
        #                   with constant indices for each grounding.
        (X_domains, A_predicates, A_rules, Q, Q_global) = inputs

        # def numpy_func(x):
        #     # This will run in eager mode
        #     if not isinstance(x, np.ndarray):
        #         x = x.numpy()
        #     return x
        
        # Q_global = tf.numpy_function(numpy_func, [Q_global], tf.float32)
        
        # print('****************************Training mode:',training, 'Testing mode:',testing, 'Dataset type:',dataset_type)
        # print('training:',self.train,'val:',self.val,'test:',self.test)
        if self.use_ultra: # NEED TO ACCOMODATE THE VALIDATION DATASET!!!!!
            if training == True:
                # print('****************************Using ULTRA embeddings from training set')
                atom_embeddings = self.ultra_model.call((X_domains, A_predicates),self.data_gen_train,Q,Q_global)
                # atom_embeddings = self.ultra_model.call((X_domains, A_predicates),self.ultra_embeddings['train']['constant'],self.ultra_embeddings['train']['predicate'])
            else:
                if testing == False:
                    # print('****************************Using ULTRA embeddings from validation set')
                    atom_embeddings = self.ultra_model.call((X_domains, A_predicates),self.data_gen_valid,Q,Q_global)
                    # atom_embeddings = self.ultra_model.call((X_domains, A_predicates),self.ultra_embeddings['valid']['constant'],self.ultra_embeddings['valid']['predicate'])
                else:
                    # print('****************************Using ULTRA embeddings from test set')
                    atom_embeddings = self.ultra_model.call((X_domains, A_predicates),self.data_gen_test,Q,Q_global) 
                    # atom_embeddings = self.ultra_model.call((X_domains, A_predicates),self.ultra_embeddings['test']['constant'],self.ultra_embeddings['test']['predicate'])
        else: 
            atom_embeddings = self.kge_model((X_domains, A_predicates))
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
        

    # def evaluate(self, *args, **kwargs):
    #     self.train_data = kwargs.pop('train_data', False)
    #     self.val_data = kwargs.pop('val_data', False)
    #     self.test_data = kwargs.pop('test_data', False)
    #     self.testing = kwargs.pop('testing', False)
    #     # print('TESTING MODE+++++++++++++++++++++++++++',self.testing, self.train_data, self.val_data, self.test_data)
    #     return super().evaluate(*args, **kwargs)


    # def evaluate(self, x, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, return_dict=False, **kwargs):
    #     self.train = kwargs.pop('train', False)
    #     self.val = kwargs.pop('val', False)
    #     self.test = kwargs.pop('test', False)
    #     dataset_type = 'testing' if self.test else 'validation' if self.val else 'training' if self.train else 'unknown'
    #     print('TESTING MODE+++++++++++++++++++++++++++', self.train, self.val, self.test)
    #     self.dataset_type = dataset_type
        
    #     # Convert inputs to dataset if necessary
    #     dataset = tf.data.Dataset.from_tensor_slices((x, y))
    #     if batch_size is not None:
    #         dataset = dataset.batch(batch_size)

    #     # Iterate through the dataset and explicitly call the `call` method
    #     for step, (batch_x, batch_y) in enumerate(dataset):
    #         y_pred = self.call(batch_x, training=False, dataset_type=self.dataset_type)
    #         loss = self.compiled_loss(batch_y, y_pred, regularization_losses=self.losses)
    #         self.compiled_metrics.update_state(batch_y, y_pred)
    #         if verbose:
    #             print(f'Step {step}: Loss = {loss.numpy()}')

    #     # Gather and return metrics results
    #     results = {m.name: m.result().numpy() for m in self.metrics}
    #     return results if return_dict else list(results.values())
    

    # def train_step(self, data):
    #     x, y = data
    #     with tf.GradientTape() as tape:
    #         y_pred = self(x, training=True)
    #         # print('!!!!!!!!!!!!!!!!!!!! train step')
    #         loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    #     gradients = tape.gradient(loss, self.trainable_variables)
    #     self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    #     self.compiled_metrics.update_state(y, y_pred)
    #     return {m.name: m.result() for m in self.metrics}
    
    # def test_step(self, data):
    #     x, y = data
    #     y_pred = self(x, training=False)
    #     # print('!!!!!!!!!!!!!!!!!!!! test step')
    #     loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    #     self.compiled_metrics.update_state(y, y_pred)
    #     return {m.name: m.result() for m in self.metrics}
    
    # def predict_step(self, data):
    #     # print('!!!!!!!!!!!!!!!!!!!! predict step')
    #     x = data
    #     return self(x, training=False, dataset_type='predicting')