from keras import Model
from keras.layers import Dense, Layer
import keras_ns as ns
import tensorflow as tf
from keras_ns.layers import DCRReasoningLayer, AtomEmbeddingLayerPerPredicate, DomainWiseMLP, ExplicitDomainEmbedders
from keras_ns.logic import FOL
from typing import List


class FlatNeuralPredicate(Layer):

    def __init__(self, atom_embedding_size, **kwargs):

        super().__init__(**kwargs)
        self.nn1 = tf.keras.Sequential([Dense(atom_embedding_size, activation="relu"), Dense(atom_embedding_size, activation="relu")])
        self.nn2 = Dense(1, activation="sigmoid")

    def call(self, inputs, *args, **kwargs):
        x= tf.concat(tf.unstack(inputs, axis= 1), axis= -1)
        emb = self.nn1(x)
        return tf.concat((emb,self.nn2(emb)), axis=-1)


class NeuralPredicateWithClass(Layer):

    def __init__(self, atom_embedding_size, n, **kwargs):

        super().__init__(**kwargs)
        self.nn1 = tf.keras.Sequential([Dense(atom_embedding_size, activation="relu"), Dense(atom_embedding_size, activation="relu")])
        self.nn2 = Dense(n, activation="softmax")

    def call(self, inputs, *args, **kwargs):

        x, c = tf.unstack(inputs, axis= 1)
        c = tf.cast(c[:,0], tf.int32)
        emb = self.nn1(x)
        pred = self.nn2(emb)
        pred = tf.expand_dims(tf.gather(pred, c, batch_dims=1), -1)
        return tf.concat((emb,pred), axis=-1)



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

        self.constant_embedder = ExplicitDomainEmbedders(domains=fol.domains,
                                                         embedders = {fol.domains[0].name: Dense(constant_embedding_size,activation = "relu"),
                                                                      # fol.domains[1].name: lambda x: tf.pad(x,[[0,0],[0,constant_embedding_size-1]])}) # we made the assumptions that ll the constants should be embedded in the same space. We really need to remove this.
                                                                      fol.domains[1].name: lambda x: tf.concat((tf.cast(tf.expand_dims(x, -1), tf.float32), tf.zeros([tf.shape(x)[0], constant_embedding_size-1])), axis=-1)}) # we made the assumptions that ll the constants should be embedded in the same space. We really need to remove this.
        predicate_embedders = {"class": NeuralPredicateWithClass(atom_embedding_size, n = len(fol.domains[1].constants)),
                               "r": FlatNeuralPredicate(atom_embedding_size)}
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


        if reasoner_depth > 0 and len(rules) > 0:
            self.reasoning: List[DCRReasoningLayer] = []
            for i in range(reasoner_depth):
              if i > 0 and reasoner_single_model:
                  self.reasoning.append(self.reasoning[0])
              self.reasoning.append(DCRReasoningLayer(
                    rules=rules,
                    formula_hidden_size=reasoner_formula_hidden_embedding_size,
                    atom_embedding_size=reasoner_atom_embedding_size,
                    temperature=0.5,
                    regularization=reasoner_regularization,
                    dropout_rate=reasoner_dropout_rate,
                    aggregate_type="sum",
                    resnet=False))
        else:
            self.reasoning = None

        # OUTPUT LAYER
        self.output_layer =  Dense(1, activation=tf.sigmoid)
        self._explain_mode = False

    def explain_mode(self, turn = True):
        self._explain_mode = turn

    def call(self, inputs, *args, **kwargs):
        (X_domains, A_predicates, A_rules, Q) = inputs
        atom_embeddings = self.input_model((X_domains, A_predicates))
        if self._explain_mode:
            to_explain = []
        concept_output  = atom_embeddings[:, -1]
        concept_output = tf.gather(params=concept_output, indices=Q)
        tf.print("concept_output", concept_output[-1:], summarize=-1)



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

        tf.print("task_output", task_output[:3], summarize=-1)
        return concept_output, task_output

