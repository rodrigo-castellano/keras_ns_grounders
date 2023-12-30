from keras import Model
from keras.layers import Dense, Layer
import keras_ns as ns
import tensorflow as tf
from keras_ns.layers import DCRReasoningLayer, AtomEmbeddingLayerPerPredicate, DomainWiseMLP
from keras_ns.logic import FOL
from typing import List


class NeuralPredicate(Layer):

    def __init__(self, atom_embedding_size, **kwargs):

        super().__init__(**kwargs)
        self.nn1 = Dense(atom_embedding_size, activation="relu")
        self.nn2 = Dense(1, activation="sigmoid")

    def call(self, inputs, *args, **kwargs):
        x= tf.concat(tf.unstack(inputs, axis= 1), axis= -1) # it is unary
        emb = self.nn1(x)
        return tf.concat((emb,self.nn2(emb)), axis=-1)


class InputLayer(Layer):

    def __init__(self, fol : FOL,
                 constant_embedding_size,
                 atom_embedding_size,
                 regularization=0,
                 dropout_rate = 0.):
        super().__init__()
        self.constant_embedder = DomainWiseMLP(domains=fol.domains,
                                               constant_embedding_sizes_per_domain={d.name: constant_embedding_size
                                                                                    for d in fol.domains})
        predicate_embedders = {p.name: NeuralPredicate(atom_embedding_size) for p in fol.predicates}
        self.atom_embedder = AtomEmbeddingLayerPerPredicate(
            embedders=predicate_embedders,
            atom_embedding_size=atom_embedding_size+1, # we also have the concept truth value
            predicates=fol.predicates,
            regularization=regularization,
            dropout_rate=dropout_rate)

    def call(self, inputs, *args, **kwargs):
        (X_domains, A_predicates) = inputs
        constant_embeddings = self.constant_embedder(X_domains)
        atom_embeddings = self.atom_embedder([constant_embeddings,
                                                 A_predicates])
        return atom_embeddings

class CollectiveModel(Model):

    def __init__(self, fol : FOL, rules,
                 input_regularization,
                 constant_embedding_size,
                 atom_embedding_size,
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
                                  constant_embedding_size,
                                  atom_embedding_size,
                                  input_regularization,
                                  dropout_rate_embedder)


        # REASONING LAYER
        if reasoner_depth > 0 and len(rules) > 0:
            self.reasoning: List[DCRReasoningLayer] = []
            for i in range(reasoner_depth):
              if i > 0 and reasoner_single_model:
                  self.reasoning.append(self.reasoning[0])
              self.reasoning.append(DCRReasoningLayer(
                    rules=rules,
                    formula_hidden_size=reasoner_formula_hidden_embedding_size,
                    atom_embedding_size=reasoner_atom_embedding_size,
                    temperature=100.,
                    regularization=reasoner_regularization,
                    dropout_rate=reasoner_dropout_rate,
                    aggregate_type="sum",
                    resnet=False))
        else:
            self.reasoning = None

        # OUTPUT LAYER
        self.output_layer =  Dense(1, activation=tf.sigmoid)
        self._explain_mode = False

    def explain_mode(self):
        self._explain_mode = True

    def call(self, inputs, *args, **kwargs):
        (X_domains, A_predicates, A_rules, Q) = inputs
        atom_embeddings = self.input_model((X_domains, A_predicates))
        if self._explain_mode:
            to_explain = []
        tf.print(atom_embeddings[:, -1], summarize=-1)
        concept_output = atom_embeddings[:, -1]
        concept_output = tf.gather(params=concept_output, indices=Q)


        if self.reasoning is not None:
            for i in range(self.enabled_reasoner_depth):
                tf.print(atom_embeddings[:, -1], summarize=-1)
                if self._explain_mode:
                    atom_embeddings, expls = self.reasoning[i]([atom_embeddings, A_rules], explain = True)
                    to_explain.append(list(expls))
                else:
                    atom_embeddings = self.reasoning[i]([atom_embeddings, A_rules])
                tf.print(atom_embeddings[:, -1], summarize=-1)


        if self._explain_mode:
            return to_explain

        task_output = atom_embeddings[:,-1]
        task_output = tf.gather(params=task_output, indices=Q)
        return concept_output, task_output

