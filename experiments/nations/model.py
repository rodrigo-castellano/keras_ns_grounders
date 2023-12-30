from keras import Model
from keras.layers import Dense, Layer
import keras_ns as ns
import tensorflow as tf
from keras_ns.layers import ConstantEmbeddings, DCRReasoningLayer, AtomEmbeddingLayerPerPredicate, AtomEmbeddingLayer, DomainWiseMLP
from keras_ns.logic import FOL
from typing import List
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
        concept_output = tf.sigmoid(dist.sample())
        concept_output = tf.squeeze(concept_output, axis=-1)
        return concept_output


class KGEModel(Model):

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

    def call(self, inputs):
        (X_domains, A_predicates) = inputs
        constant_embeddings = self.constant_embedder(X_domains)
        kge_atom_embeddings = self.kge_embedder([constant_embeddings,
                                                 A_predicates])
        return kge_atom_embeddings

class CollectiveModel(Model):

    def __init__(self,
                 fol: FOL,
                 rules,  # List[Rule]?
                 kge_embedder: tf.keras.layers.Layer,
                 kge_regularization: float,
                 constant_embedding_size: int,
                 kge_atom_embedding_size: int,
                 dropout_rate_embedder: float,
                 # same model for all reasoning depths
                 reasoner_single_model: bool,
                 reasoner_atom_embedding_size: int,
                 reasoner_formula_hidden_embedding_size: int,
                 reasoner_regularization: float,
                 reasoner_dropout_rate: float,
                 reasoner_depth: int,
                 aggregation_type: str,
                 signed: bool,
                 temperature: float,
                 use_gumbel: bool,
                 filter_num_heads: int,
                 filter_activity_regularization: float):
        super().__init__()
        # Reasoning depth of the model structure.
        self.reasoner_depth = reasoner_depth
        # Reasoning depth currently used, this can differ from
        # self.reasoner_depth during multi-stage learning (like if
        # pretraining the KGEs).
        self.enabled_reasoner_depth = reasoner_depth

        self.kge_model = KGEModel(fol,
                                  kge_embedder,
                                  kge_regularization,
                                  constant_embedding_size,
                                  kge_atom_embedding_size,
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
                  atom_embedding_size=reasoner_atom_embedding_size, # embedding + concept truth value
                  aggregate_type=aggregation_type,
                  temperature=temperature,
                  signed=signed,
                  filter_num_heads=filter_num_heads,
                  regularization=reasoner_regularization,
                  dropout_rate=reasoner_dropout_rate,
                  resnet=False))
        else:
            self.reasoning = None

        # OUTPUT LAYER
        kge_output_layer = kge_embedder.output_layer()
        if use_gumbel:
            layer = tf.keras.layers.Lambda(
                lambda x: tf.expand_dims(logit(kge_output_layer(x)), axis=-1))
            self.output_layer = MakeGumbel(temperature=temperature, layer=layer)
        else:
            self.output_layer = kge_output_layer

        self._explain_mode = False

    def explain_mode(self):
        self._explain_mode = True

    def call(self, inputs, *args, **kwargs):
        (X_domains, A_predicates, A_rules, Q) = inputs
        atom_embeddings = self.kge_model((X_domains, A_predicates))

        concept_output = tf.expand_dims(self.output_layer(atom_embeddings), -1)
        if self._explain_mode:
            to_explain = []

        atom_embeddings = tf.concat((atom_embeddings, concept_output), axis=-1)

        if self.reasoning is not None:
            for i in range(self.enabled_reasoner_depth):
                if self._explain_mode:
                    atom_embeddings, expls = self.reasoning[i](
                        [atom_embeddings, A_rules], explain=True)
                    to_explain.append(list(expls))
                else:
                    atom_embeddings = self.reasoning[i]([atom_embeddings,
                                                         A_rules])

        if self._explain_mode:
            # List[Dict[rule_name: str, List[str]] with
            # shape[self.enabled_reasoner_depth, num_atom_embeddings]
            return to_explain

        task_output = atom_embeddings[:,-1]
        task_output = tf.gather(params=task_output, indices=Q)
        concept_output = tf.gather(params=tf.squeeze(concept_output, -1),
                                   indices=Q)
        #tf.print('FINAL_OUTPUT', task_output)
        #tf.print('PERC', tfp.stats.percentile(concept_output.to_tensor(
        #    default_value=0.0), q=[10, 30, 50, 70, 90, 99]))

        return concept_output, task_output
