from keras.layers import Embedding
from keras.layers import Layer
from keras.layers import Dense
from keras import Sequential
from keras.regularizers import L2
from typing import Dict, List, Tuple
from keras_ns.logic.commons import Domain
import tensorflow as tf

class ConstantEmbeddings(Layer):
    """Calls the constant rules_embedders, differenciating the behavior of
       the single domains."""
    def __init__(self, domains: List[Domain],
                 constant_embedding_sizes_per_domain: Dict[str, int],
                 regularization: float=0.0):
        super().__init__()
        self.embedder = {}
        self.domains = domains
        for domain in domains:
            self.embedder[domain.name] = Embedding(
                len(domain.constants),
                constant_embedding_sizes_per_domain[domain.name],
                embeddings_regularizer=L2(regularization))

    def call(self, domain_inputs, **kwargs):
        domain_features = {}
        for domain in self.domains:
            domain_features[domain.name] = self.embedder[domain.name](
                domain_inputs[domain.name])
        return domain_features

class PredicateEmbeddings(Layer):
    """Calls the predicate embedders."""
    def __init__(self, predicates: List[str],
                 predicate_embedding_size: int,
                 regularization: float=0.0):
        super().__init__()
        #self.predicates = predicates
        # self.predicate2index = {p:i for i,p in enumerate(predicates)}
        #self.table = tf.lookup.StaticHashTable(
        #    tf.lookup.KeyValueTensorInitializer(self.predicate2index.keys(),
        #                                        self.predicate2index.values()),
        #    default_value=-1)
        #self.regularization = regularization
        self.embedder = Embedding(len(predicates), predicate_embedding_size,
                                  embeddings_regularizer=L2(regularization))

    # Inputs is tensor of predicate idx.
    # Output is tensor of embeddings of each predicate.
    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return self.embedder(inputs)  #BE
    
class AdaptiveConstantEmbeddings(Layer):
    def __init__(self, domains: List[Domain],
                 constant_embedder: ConstantEmbeddings,
                 constant_embedding_size: int,
                 num_adaptive_constants: int):
        super().__init__()
        self.adaptive_constant2relevance = {}
        self.constant_embedder = constant_embedder
        self.num_adaptive_constants = num_adaptive_constants
        self.domains = domains
        for domain in domains:
            self.adaptive_constant2relevance[domain.name] = (
                Sequential([
                    Embedding(num_adaptive_constants, constant_embedding_size),
                    Dense(len(domain.constants), activation='softmax'),
                ]))

    def call(self, domain_inputs, **kwargs):
        embedder_inputs = {}
        for domain in self.domains:
            # Only constants that are outside of domain bounds ara
            # adaptive and should be embedded.
            # We zero out non-adaptive constants, which will be anyway
            # discarded later by the caller..
            flow = tf.math.maximum(
                0, domain_inputs[domain.name] - len(domain.constants))
            # Computes the distribution over the constants.
            flow = self.adaptive_constant2relevance[domain.name](flow)
            # Selects the best scoring constants.
            flow = tf.math.argmax(flow, axis=-1)
            tf.print('Selected constants domain', domain.name, flow)
            embedder_inputs[domain.name] = flow

        return self.constant_embedder(embedder_inputs)

class DomainWiseMLP(Layer):
    """Calls the constant rules_embedders, differenciating the behavior of
       the single domains."""
    def __init__(self, domains: List[Domain], constant_embedding_sizes_per_domain):
        super().__init__()
        self.embedder = {}
        self.domains = domains
        for domain in domains:
            m = Sequential()
            m.add(Dense(constant_embedding_sizes_per_domain[domain.name],activation = "relu"))
            self.embedder[domain.name] = m

    def call(self, domain_inputs, **kwargs):
        domain_features = {}
        for d in self.domains:
            domain_name = d.name
            domain_features[domain_name] = self.embedder[domain_name](domain_inputs[domain_name])
        return domain_features

class ExplicitDomainEmbedders(Layer):
    """Calls the constant rules_embedders, differenciating the behavior of the single domains."""
    def __init__(self, domains: List[Domain], embedders):
        super().__init__()
        self.embedder = embedders
        self.domains = domains

    def call(self, domain_inputs, **kwargs):
        domain_features = {}
        for domain in self.domains:
                domain_features[domain.name] = self.embedder[domain.name](domain_inputs[domain.name])
        return domain_features
