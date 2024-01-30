import math

import keras.layers
import tensorflow as tf
from keras.layers import Layer
import abc
from typing import List, Tuple
from collections import OrderedDict
from keras_ns.logic.commons import Predicate, FOL, Domain
from abc import ABCMeta, abstractmethod


class AtomEmbeddingLayer(Layer):
    def __init__(self, embedder_class,
                 atom_embedding_size,
                 predicates: List[Predicate],
                 regularization=0.0,
                 dropout_rate=0.0,
                 **kwargs):
        super().__init__()
        self.atom_embedding_size = atom_embedding_size
        self.regularization = regularization
        self.regularization_n3 = 0.0
        self.predicates = predicates
        self.embedders = {}
        self.embedder_class = embedder_class

        for predicate in self.predicates: 
            self.embedders[predicate.name] = self.embedder_class(
                atom_embedding_size=self.atom_embedding_size,
                regularization=regularization,
                dropout_rate=dropout_rate,
                **kwargs)
            # This shoud not be needed.
            #for l in self.embedders[predicate.name].losses:
            #    self.add_loss(l)

    def create_tuples(self, constants_embeddings,
                      tuples_tensor, predicate_domains):
        tuple_features_per_predicate = []
        for i,domain in enumerate(predicate_domains):
            # for each domain, take all the atoms in that predicate, where the constants are represented by indeces, 
            # and substute the indeces of each cte by its embedding
            constants = tf.gather(constants_embeddings[domain.name],
                                  tf.cast(tuples_tensor[..., i], tf.int32),
                                  axis=0)
            tuple_features_per_predicate.append(constants)
        # Returns shape (batch_size, predicate_arity, constant_embedding_size)
        return tf.stack(tuple_features_per_predicate, axis=-2)

    def call(self, inputs, **kwargs):
        # This is a GNN-like interface, constants embeddings (dictionary with
        # domain as keys) and atoms (dictionary with predicates as keys and
        # tuples of constants as values).
        constants_embeddings, predicate_to_constant_tuples = inputs

        # Now we embed the tuples in input to the predicates by stacking their
        # constants embeddings.
        tuple_features = {}
        for predicate in self.predicates:
            if predicate.name not in predicate_to_constant_tuples:
                continue

            def GetZeros(size: int):
                def Fn():
                    arity: int = len(predicate.domains)
                    return  tf.zeros([0, arity, size], dtype=tf.float32)
                return Fn
            def GetTuples():
                return self.create_tuples(
                    constants_embeddings=constants_embeddings,
                    tuples_tensor=predicate_to_constant_tuples[predicate.name],
                    predicate_domains=predicate.domains)
            # for the predicate, substitute the indeces of constants by their embeddings (getTuples), unless the embedds are 0 (getZeros)
            tuple_features[predicate.name] = tf.cond(
                tf.size(predicate_to_constant_tuples[predicate.name]) == 0,
                GetZeros(self.embedders[predicate.name].input_size()),
                GetTuples)
        # if kwargs['training'] != True :
        #     predicate_atoms2embeddings = []  # do not use list comprehansion, causes out os scope errors
        #     for predicate in self.predicates:
        #         if (predicate.name != 'neighborOf') and (predicate.name != 'locatedInSR'):
        #             print('predicate', predicate.name, flush=True)
        #             print('tuple_features[predicate.name]', tuple_features[predicate.name].shape,tuple_features[predicate.name], flush=True)
        #             embeddings = self.embedders[predicate.name](
        #                 tuple_features[predicate.name])
        #             predicate_atoms2embeddings.append(embeddings)
        #     print('finish predicate', flush=True)
        # else: 
        # Now we embed the tuples (per predicate) using the dynamically defined atom_embedders
        # Each element has shape (B,atom_emb_size).
        predicate_atoms2embeddings = []  # do not use list comprehansion, causes out os scope errors
        for predicate in self.predicates:
            embeddings = self.embedders[predicate.name](
                tuple_features[predicate.name])
            # shape (n_atoms for that predicate, n_domains, embedd size)
            predicate_atoms2embeddings.append(embeddings)
        # Put all the atoms of all the predicates in a unique dense tensor (instead of having it per predicate, we have it all together)
        # shape (n_atoms for all predicates, n_domains, embedd size)
        embeddings = tf.concat(predicate_atoms2embeddings, axis=0)

        # Specify the last dimension size for the following layers
        embeddings = tf.reshape(embeddings, [-1, self.atom_embedding_size])
        if self.regularization_n3 > 0.0:
            abs_embeddings = tf.math.abs(embeddings)
            self.add_loss(self.regularization_n3 * tf.reduce_sum(
                abs_embeddings * abs_embeddings * abs_embeddings))
        return embeddings

############################################
# KGE=Layer interface.
class KGELayer(metaclass=ABCMeta):
    @property
    @abstractmethod
    def input_size(self):
        pass

    @property
    @abstractmethod
    def output_size(self):
        pass

############################################
class DistMult(KGELayer, Layer):
    def __init__(self, atom_embedding_size, regularization=0.0,
                 regularization_n3=0.0,
                 dropout_rate=0.0, **kwargs):
        super().__init__()
        self.regularization = regularization
        self.regularization_n3 = regularization_n3
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)

        self.atom_embedding_size = atom_embedding_size
        init = tf.initializers.GlorotUniform()
        self.R = tf.Variable(
            initial_value=init(shape=[self.atom_embedding_size]),
            name='DistMult')

    def input_size(self):
        return self.atom_embedding_size

    def output_size(self):
        return self.atom_embedding_size

    def call(self, inputs, **kwargs):
        inputs = self.dropout_layer(inputs)
        R = self.dropout_layer(self.R)
        predicate_prod = tf.reduce_prod(inputs, axis=1)
        embeddings = R * predicate_prod
        if self.regularization > 0.0:
            self.add_loss(self.regularization * tf.nn.l2_loss(self.R))
        if self.regularization_n3 > 0.0:
            abs_embeddings = tf.math.abs(embeddings)
            self.add_loss(self.regularization_n3 *
                          tf.nn.l2_loss(abs_embeddings))
        return embeddings

    @classmethod
    def output_layer(cls):
        def __internal__(inputs):
            outputs = tf.reduce_sum(inputs, axis=-1)
            outputs = tf.nn.sigmoid(outputs)
            return outputs

        return __internal__

###########################################
class TransE(KGELayer, Layer):

    @classmethod
    def output_layer(cls):
        def __internal__(inputs):
            # outputs = 1.0 - 2.0 * (tf.nn.sigmoid(tf.reduce_mean(tf.square(inputs), axis=-1)) - 0.5)
            outputs = -tf.reduce_mean(tf.square(inputs), axis=-1)
            return outputs
        return __internal__


    def __init__(self, atom_embedding_size, regularization=0.0,
                 regularization_n3=0.0,
                 dropout_rate=0.0, **kwargs):
        super().__init__()
        self.regularization = regularization
        self.regularization_n3 = regularization_n3
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)

        self.atom_embedding_size = atom_embedding_size
        init = tf.initializers.GlorotUniform()
        self.R = tf.Variable(
            initial_value=init(shape=[self.atom_embedding_size]),
            name='TransE')

    def input_size(self):
        return self.atom_embedding_size

    def output_size(self):
        return self.atom_embedding_size

    def call(self, inputs, **kwargs):
        inputs = self.dropout_layer(inputs)
        R = self.dropout_layer(self.R)

        head = tf.squeeze(tf.gather(params=inputs, indices=[0], axis=1),axis=1)
        tail = tf.squeeze(tf.gather(params=inputs, indices=[1], axis=1), axis=1)

        if self.regularization > 0.0:
            self.add_loss(self.regularization * tf.nn.l2_loss(self.R))

        embeddings = R + head - tail
        if self.regularization_n3 > 0.0:
            abs_embeddings = tf.math.abs(embeddings)
            self.add_loss(self.regularization_n3 * (
                tf.nn.l2_loss(self.R) + tf.nn.l2_loss(head) + tf.nn.l2_loss(tail)))
        return embeddings

###########################################
class ModE(KGELayer, Layer):

    @classmethod
    def output_layer(cls):
        def __internal__(inputs):
            outputs = tf.exp(-tf.norm(inputs, axis=-1))
            return outputs
        return __internal__

    def __init__(self, atom_embedding_size, regularization=0.0,
                 regularization_n3=0.0,
                 dropout_rate=0.0, **kwargs):
        super().__init__()
        self.regularization = regularization
        self.regularization_n3 = regularization_n3
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)

        self.atom_embedding_size = atom_embedding_size
        init = tf.initializers.GlorotUniform()
        self.R = tf.Variable(
            initial_value=init(shape=[self.atom_embedding_size]),
            name='ModE')

    def input_size(self):
        return self.atom_embedding_size

    def output_size(self):
        return self.atom_embedding_size

    def call(self, inputs, **kwargs):
        inputs = self.dropout_layer(inputs)
        R = self.dropout_layer(self.R)

        head = tf.squeeze(tf.gather(params=inputs, indices=[0], axis=1),axis=1)
        tail = tf.squeeze(tf.gather(params=inputs, indices=[1], axis=1), axis=1)

        if self.regularization > 0.0:
            self.add_loss(self.regularization * tf.nn.l2_loss(self.R))

        embeddings = R * head - tail
        if self.regularization_n3 > 0.0:
            abs_embeddings = tf.math.abs(embeddings)
            self.add_loss(self.regularization_n3 * (
                tf.nn.l2_loss(self.R) + tf.nn.l2_loss(head) + tf.nn.l2_loss(tail)))
        return embeddings


class ComplEx(KGELayer, Layer):

    @classmethod
    def output_layer(cls):
        def __internal__(inputs):
            outputs = tf.reduce_sum(inputs, axis=-1)
            outputs = tf.nn.sigmoid(outputs)
            return outputs
        return __internal__

    def __init__(self, atom_embedding_size,
                 regularization=0.0, regularization_n3=0.0,
                 dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.atom_embedding_size = atom_embedding_size
        self.regularization = regularization
        self.regularization_n3 = regularization_n3
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
        init = tf.initializers.GlorotUniform()
        self.Rr = tf.Variable(
            initial_value=init(shape=[self.atom_embedding_size]),
            name='ComplExReal')
        self.Ri = tf.Variable(
            initial_value=init(shape=[self.atom_embedding_size]),
            name='ComplExImm')

    def input_size(self):
        return 2 * self.atom_embedding_size

    def output_size(self):
        return self.atom_embedding_size

    def call(self, inputs, **kwargs): 
        inputs = self.dropout_layer(inputs)
        Rr = self.dropout_layer(self.Rr)
        Ri = self.dropout_layer(self.Ri)

        # assert inputs.shape[-1] == self.input_size(), (
        #     'Wrong Shape (%s[-1] <=> %d', (inputs.shape,
        #                                    self.input_size()))

        head = tf.squeeze(tf.gather(params=inputs, indices=[0], axis=1),axis=1)
        tail = tf.squeeze(tf.gather(params=inputs, indices=[1], axis=1), axis=1)

        h_r, h_i = tf.split(head, 2, axis=1)
        t_r, t_i = tf.split(tail, 2, axis=1)

        e1 = h_r * t_r * Rr
        e2 = h_i * t_i * Rr
        e3 = h_r * t_i * Ri
        e4 = h_i * t_r * Ri
        embeddings = e1 + e2 + e3 - e4
        # if self.regularization > 0.0:
        #     self.add_loss(self.regularization * (tf.nn.l2_loss(self.Rr) + tf.nn.l2_loss(self.Ri)))
        if self.regularization_n3 > 0.0:
            abs_head = tf.math.abs(head)
            abs_tail = tf.math.abs(tail)
            abs_R = tf.math.abs(tf.concat([Rr,Ri], axis=0))
            self.add_loss(self.regularization_n3 * (
                tf.reduce_sum(abs_head * abs_head * abs_head) +
                tf.reduce_sum(abs_tail * abs_tail * abs_tail) +
                tf.reduce_sum(abs_R * abs_R * abs_R)))
        return embeddings


class RotatE(KGELayer, Layer):
    """`RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space`_ (RotatE), which defines each relation as a rotation from the source entity to the target entity in the complex vector space.
    Attributes:
        args: Model configuration parameters.
        epsilon: Calculate embedding_range.
        margin: Calculate embedding_range and loss.
    .. _RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space: https://openreview.net/forum?id=HkgEQnRqYQ
    """
    # Class attributes.
    epsilon = 2.0
    margin = 2.0

    def __init__(self, atom_embedding_size, regularization=0.0,
                 regularization_n3=0.0,
                 dropout_rate=0.0):
        super().__init__()
        assert atom_embedding_size > 0
        self.regularization = regularization
        self.regularization_n3 = regularization_n3
        if dropout_rate > 0.0:
            self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
        else:
            self.dropout_layer = tf.identity

        self.atom_embedding_size = atom_embedding_size
        self.embedding_range = (self.margin + self.epsilon) / (
            atom_embedding_size)

        init = tf.initializers.GlorotUniform()
        #self.R = tf.Variable(initial_value=init(shape=[atom_embedding_size]),
        #                     name='RotateERelation')
        self.R = tf.constant(init(shape=[atom_embedding_size]))
        self.norm_factor = math.pi / self.embedding_range

    @classmethod
    def output_layer(cls):
        def __internal__(inputs):
            outputs = 2.0 - tf.reduce_sum(inputs, axis=-1)
            return outputs
        return __internal__

    def input_size(self):
        return 2 * self.atom_embedding_size

    def output_size(self):
        return self.atom_embedding_size

    def call(self, inputs, training=False):
        """Calculating the score of triples.
        The formula for calculating the score is :math:`\gamma - \|h \circ r - t\|`
        Args:
            head_emb: The head entity embedding.
            relation_emb: The relation embedding.
            tail_emb: The tail entity embedding.
        Returns:
            embeddings: The output embeddings (B, atom_embedding_size)
        """
        # tf.print('I', inputs)
        inputs = self.dropout_layer(inputs)
        assert inputs.shape[-1] == self.input_size()

        assert inputs.shape[1] == 2
        head = tf.squeeze(tf.gather(params=inputs, indices=[0], axis=1),axis=1)
        tail = tf.squeeze(tf.gather(params=inputs, indices=[1], axis=1), axis=1)
        if tf.shape(head)[0] == 0 or tf.shape(tail)[0] == 0:
            return tf.zeros([0, self.atom_embedding_size])

        re_head, im_head = tf.split(head, 2, axis=-1)
        re_tail, im_tail = tf.split(tail, 2, axis=-1)

        R = self.dropout_layer(self.R)
        phase_relation = R * self.norm_factor
        #tf.print('R', R)
        re_relation = tf.math.cos(phase_relation)
        im_relation = tf.math.sin(phase_relation)
        #tf.print('CS', re_relation, im_relation)
        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        re_score = re_score - re_head
        im_score = im_score - im_head

        embeddings = tf.stack([re_score, im_score], axis=0) #(2,B,atom_emb_size)
        # tf.norm has inf grad for 0, so we craft a numerically stable version here.
        embeddings = tf.sqrt(1e-7 + tf.reduce_sum(tf.square(embeddings), axis=0))  #(B,atom_emb_size)
        #if self.regularization > 0.0:
        #    self.add_loss(self.regularization * tf.nn.l2_loss(self.R))
        #if self.regularization_n3 > 0.0:
        #    self.add_loss(self.regularization_n3 * tf.nn.l2_loss(embeddings))
        return embeddings


####################################
def KGEFactory(name):
  if name.casefold() == 'complex':
    return ComplEx
  elif name.casefold() == 'distmult':
    return DistMult
  elif name.casefold() == 'transe':
    return TransE
  elif name.casefold() == 'rotate':
    return RotatE
  elif name.casefold() == 'mode':
    return ModE
  print('Unknown KGE', name, flush=True)
  return None
