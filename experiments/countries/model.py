from keras import Model
from keras.layers import Dense, Layer
import keras_ns as ns
import tensorflow as tf
from keras_ns.nn.constant_embedding import AdaptiveConstantEmbeddings, ConstantEmbeddings
from keras_ns.nn.reasoning import *
from keras_ns.nn.kge import AtomEmbeddingLayer
from keras_ns.logic import FOL, Rule
from typing import Dict, List
from keras_ns.logic.semantics import GodelTNorm
from typing import Dict, List, Union
import tensorflow_probability as tfp

def logit(x):

    """ Computes the logit function, i.e. the logistic sigmoid inverse. """
    x = tf.clip_by_value(x, 1e-6, 1.0)
    x = 1.0 / x - 1.0
    x = tf.clip_by_value(x, 1e-6, 1e6)
    return -tf.math.log(x)

# Take a generic layer and gumbelifies it.
class MakeGumbel(Layer):

    # The passed layer must fold a shape (..., N) into (..., 1) like for example
    #  dense layer.
    def __init__(self, temperature, layer=Dense(1, activation=None), **kwargs):
        super().__init__(**kwargs)
        self.parameter_bernoulli = layer
        assert temperature > 0.
        self.coolness = 1. / temperature

    def call(self, inputs, *args, **kwargs):
        atom_embeddings = inputs
        logits = self.parameter_bernoulli(atom_embeddings)
        dist = tfp.distributions.Logistic(logits * self.coolness, self.coolness)
        #if keras.backend.learning_phase():
        concept_output = tf.sigmoid(dist.sample())
        #else:
        #  concept_output = tf.one_hot(tf.math.argmax(logits), tf.shape(logits)[-1])
        concept_output = tf.squeeze(concept_output, axis=-1)
        return concept_output


class KGEModel(Model):

    def __init__(self, fol:FOL,
                 kge_embedder,
                 kge_regularization,
                 constant_embedding_size: int,
                 kge_atom_embedding_size: int,
                 dropout_rate_embedder: float,
                 num_adaptive_constants: int = 0):
        super().__init__()
        self.fol = fol
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
        self.kge_embedder = AtomEmbeddingLayer(
            embedder_class=kge_embedder,
            atom_embedding_size=kge_atom_embedding_size,
            predicates=fol.predicates,
            regularization=kge_regularization,
            dropout_rate=dropout_rate_embedder)

    def call(self, inputs):
        # X_domains type is Dict[str, inputs]
        # A_predicate: Dict[predicate_name, List[Tuple[Index1, ..., IndexN]]]
        (X_domains, A_predicates) = inputs
        if self.adaptive_constant_embedder is not None:
            X_domains_fixed_mask = {
                name:tf.where(x < len(self.fol.name2domain[name].constants),
                              True, False) for name,x in X_domains.items()}
            X_domains_fixed = {
                name:tf.where(X_domains_fixed_mask[name], x, 0)
                for name,x in X_domains.items()}
            constant_embeddings_fixed = self.constant_embedder(X_domains_fixed)
            constant_embeddings_adaptive = self.adaptive_constant_embedder(
                X_domains)
            constant_embeddings = {
                name:tf.where(
                    # Expand dim to broadcast the where to the embeddings size.
                    tf.expand_dims(X_domains_fixed_mask[name], axis=-1),
                    constant_embeddings_fixed[name],
                    constant_embeddings_adaptive[name])
                for name in X_domains.keys()}
        else:
            constant_embeddings = self.constant_embedder(X_domains)  
        atom_embeddings = self.kge_embedder([constant_embeddings, A_predicates])
        return atom_embeddings


class CollectiveModel(Model):

    def __init__(self,
                 fol: FOL,
                 rules: List[Rule],
                 kge_embedder: tf.keras.layers.Layer,
                 kge_regularization: float,
                 constant_embedding_size: int,
                 kge_atom_embedding_size: int,
                 dropout_rate_embedder: float,
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
                 use_gumbel: bool,
                 filter_num_heads: int,
                 filter_activity_regularization: float,
                 num_adaptive_constants: int):
        super().__init__()
        # Reasoning depth of the model structure.
        self.reasoner_depth = reasoner_depth
        # Reasoning depth currently used, this can differ from
        # self.reasoner_depth during multi-stage learning (like if
        # pretraining the KGEs).
        self.enabled_reasoner_depth = reasoner_depth
        self.resnet = resnet
        self.logic = GodelTNorm()

        self.kge_model = KGEModel(fol,
                                  kge_embedder,
                                  kge_regularization,
                                  constant_embedding_size,
                                  kge_atom_embedding_size,
                                  dropout_rate_embedder,
                                  num_adaptive_constants)
        
        self.model_name = model_name

        # OUTPUT LAYER
        kge_output_layer = kge_embedder.output_layer()
        if use_gumbel:
            layer = tf.keras.layers.Lambda(
                lambda x: tf.expand_dims(logit(kge_output_layer(x)), axis=-1))
            self.output_layer = MakeGumbel(temperature=temperature, layer=layer)
        else:
            self.output_layer = kge_output_layer

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
                      num_formulas=3,  # make it a param
                      use_gumbel=True,
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

    def call(self, inputs, *args, **kwargs):
        # No explanations are posible when reasoning is disabled.
        assert self.reasoning is not None or not self._explain_mode
        # Check that we are using an explainable model.
        assert self.model_name == 'dcr' or self.model_name == 'cdcr' or self.model_name == 'r2n' or self.model_name == 'sbr' or self.model_name == 'gsbr' or self.model_name == 'rnm' or self.model_name == 'no_reasoner'

        # X_domains type is Dict[str, tensor[constant_indices_in_domain]]
        # A_predicate: Dict[predicate_name, List[Tuple[Index1, ..., IndexN]]]
        #              e.g. mapping predicate_name -> tensor [num_groundings, arity]
        #                   with constant indices for each grounding.
        (X_domains, A_predicates, A_rules, Q) = inputs
        atom_embeddings = self.kge_model((X_domains, A_predicates))
        # tf.print('atom_embeddings', atom_embeddings.shape, atom_embeddings)

        concept_output = tf.expand_dims(self.output_layer(atom_embeddings), -1)
        # tf.print('concept_output', concept_output.shape, concept_output)

        explanations = None
        if self.reasoning is not None:
            task_output = concept_output  # initialization
            for i in range(self.enabled_reasoner_depth):
                if self._explain_mode and i == self.enabled_reasoner_depth - 1:
                    explanations = self.reasoning[i].explain(
                        [task_output, atom_embeddings, A_rules])
                    # tf.print('explanations', explanations.shape, explanations)
                # tf.print('reasoning', i)
                # tf.print('task_output', task_output.shape, task_output)
                # tf.print('atom_embeddings', atom_embeddings.shape, atom_embeddings)
                # tf.print('A_rules',  A_rules)
                # tf.print('reasoning', i)
                # tf.print('task_output', task_output.shape )
                # tf.print('atom_embeddings', atom_embeddings.shape )
                # tf.print('A_rules',  A_rules)

                task_output, atom_embeddings = self.reasoning[i]([
                    task_output, atom_embeddings, A_rules])
                # tf.print('task_output', task_output.shape, task_output)
                # tf.print('atom_embeddings', atom_embeddings.shape, atom_embeddings)

        if self.reasoning is not None:
            task_output = tf.gather(params=tf.squeeze(task_output, -1), indices=Q)
        # tf.print('task_output after reasoning', task_output.shape, task_output)
        concept_output = tf.gather(params=tf.squeeze(concept_output, -1),
                                   indices=Q)
        # tf.print('concept_output after reasoning', concept_output.shape, concept_output)
        if self.resnet and self.reasoning is not None:
            task_output = self.logic.disj_pair(task_output, concept_output)
        else: # no resnet, set task_output to 0
            task_output = tf.zeros_like(concept_output)
        # tf.print('task_output after resnet', task_output.shape, task_output )

        if self._explain_mode:
            return concept_output, task_output, explanations
        else:
            return {'concept':concept_output, 'task':task_output}