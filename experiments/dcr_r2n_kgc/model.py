from keras import Model
from collections import OrderedDict
from keras.layers import Dense
import keras_ns as ns
import tensorflow as tf
from keras_ns.layers import DCRReasoningLayer, AtomEmbeddingLayer, ConstantEmbeddings
from keras_ns.logic import FOL
import tensorflow_probability as tfp

class InputLayer(Model):

    def __init__(self, fol : FOL,
                 kge_embedder,
                 kge_regularization,
                 constant_embedding_size,
                 kge_atom_embedding_size,
                 dropout_rate_embedder):
        super().__init__()
        self.constant_embedder = ConstantEmbeddings(
            domains=fol.domains,
            constant_embedding_sizes_per_domain={
                domain.name: constant_embedding_size
                for domain in fol.domains},
            regularization=kge_regularization)
        self.kge_embedder = AtomEmbeddingLayer(
            embedder_class=kge_embedder,
            atom_embedding_size=kge_atom_embedding_size,
            predicates=fol.predicates,
            regularization=kge_regularization,
            dropout_rate=dropout_rate_embedder)

    def __call__(self, inputs, *args, **kwargs):
        (X_domains, A_predicates) = inputs
        constant_embeddings = self.constant_embedder(X_domains)
        kge_atom_embeddings = self.kge_embedder([constant_embeddings,
                                                 A_predicates])
        return kge_atom_embeddings

class KGCModel(Model):

    def __init__(self, fol : FOL, rules,
                 kge_embedder,
                 *,
                 kge_regularization,
                 constant_embedding_size,
                 kge_atom_embedding_size,
                 dropout_rate_embedder,
                 reasoner_single_model,
                 reasoner_atom_embedding_size,
                 reasoner_formula_hidden_embedding_size,
                 reasoner_regularization,
                 reasoner_dropout_rate,
                 reasoner_depth):
        super().__init__()
        # Reasoning depth of the model structure.
        self.reasoner_depth = reasoner_depth
        # Reasoning depth currently used, this can differ from
        # self.reasoner_depth during multi-stage learning (like if
        # pretraining the KGEs).
        self.enabled_reasoner_depth = reasoner_depth

        self.input_model = InputLayer(fol,
                                  kge_embedder,
                                  kge_regularization,
                                  constant_embedding_size,
                                  kge_atom_embedding_size,
                                  dropout_rate_embedder)


        # REASONING LAYER
        if reasoner_depth > 0 and len(rules) > 0:
            self.reasoning = []
            for i in range(reasoner_depth):
              if i > 0 and reasoner_single_model:
                  self.reasoning.append(self.reasoning[0])
              self.reasoning.append(DCRReasoningLayer(
                    temperature=1.,
                    rules=rules,
                    formula_hidden_size=reasoner_formula_hidden_embedding_size,
                    atom_embedding_size=reasoner_atom_embedding_size,
                    regularization=reasoner_regularization,
                    dropout_rate=reasoner_dropout_rate,
                    aggregate_type="max"))
        else:
            self.reasoning = None

        # OUTPUT LAYER
        # self.output_layer =  kge_embedder.output_layer()
        self.parameter_benoulli = Dense(1, activation=None)


        self._explain_mode = False

    def explain_mode(self, turn = True):
        self._explain_mode = turn

    def call(self, inputs, *args, **kwargs):
        (X_domains, A_predicates, A_rules, Q) = inputs
        atom_embeddings = self.input_model((X_domains, A_predicates))
        logits = self.parameter_benoulli(atom_embeddings)
        temperature = 1e-2
        dist = tfp.distributions.Logistic(logits / temperature, 1. / temperature)
        concept_output = tf.sigmoid(dist.sample())
        # concept_output = tfp.distributions.RelaxedBernoulli(temperature = 1e-1,logits=logits).sample()
        # concept_output = tf.expand_dims(self.output_layer(atom_embeddings),-1)
        if self._explain_mode:
            to_explain = []
        atom_embeddings  = tf.concat((atom_embeddings,concept_output), axis=-1)
        concept_output = tf.gather(params=tf.squeeze(concept_output,-1), indices=Q)
        # tf.print("concept_output", concept_output[-1:], summarize=-1)



        if self.reasoning is not None:
            for i in range(self.enabled_reasoner_depth):
                if self._explain_mode:
                    atom_embeddings, expls = self.reasoning[i]([atom_embeddings, A_rules], explain = True)
                    to_explain.append(list(expls))
                else:
                    atom_embeddings = self.reasoning[i]([atom_embeddings, A_rules])


        if self._explain_mode:
            return to_explain
        #
        task_output = atom_embeddings[:,-1]
        task_output = tf.gather(params=task_output, indices=Q)

        # tf.print("task_output", task_output[:3], summarize=-1)
        return concept_output, task_output

