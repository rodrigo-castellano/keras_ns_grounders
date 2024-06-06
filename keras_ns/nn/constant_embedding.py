from keras.layers import Embedding
from keras.layers import Layer
from keras.layers import Dense
from keras import Sequential
from keras.regularizers import L2
from typing import Dict, List, Tuple
from keras_ns.logic.commons import Domain
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer


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
            # print('Domain', domain.name)
            # print('Constants', len(domain.constants), 'Embedding size', constant_embedding_sizes_per_domain[domain.name])
            self.embedder[domain.name] = Embedding(
                len(domain.constants),
                constant_embedding_sizes_per_domain[domain.name],
                embeddings_regularizer=L2(regularization))

    # domain_inputs is Dict domain->tensor of idx
    def call(self, domain_inputs: Dict[str, tf.Tensor], **kwargs):
        domain_features = {}
        for domain in self.domains:
            domain_features[domain.name] = self.embedder[domain.name](
                domain_inputs[domain.name])
            # print('Domain', domain.name, 'inputs', domain_inputs[domain.name].shape,domain_inputs[domain.name]  )
            # print('Domain', domain.name, 'outputs', domain_features[domain.name].shape,domain_features[domain.name])
        # To use an LM to get the embed. of the constants, we need to give the word (not the index), along with its text
        # How should I do, before the serializer I work directly with the word to get the text? or once I have the indeces, I retrieve the word from each index?
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

    
class LMEmbeddings():
    
    @staticmethod
    def attended_mean_pooling(model_output, attention_mask) -> tf.Tensor:
        """
        Performs mean pooling on the token embeddings based on the attention mask.

        Args:
            model_output (tf.Tensor): The output of the model.
            attention_mask (tf.Tensor): The attention mask to indicate which tokens to include in the pooling.

        Returns:
            tf.Tensor: The mean-pooled token embeddings.

        """
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = tf.cast(
            tf.broadcast_to(tf.expand_dims(attention_mask, -1), tf.shape(token_embeddings)),
            tf.float32
        )
        return tf.math.reduce_sum(token_embeddings * input_mask_expanded, axis=1) / tf.clip_by_value(tf.math.reduce_sum(input_mask_expanded, axis=1), 1e-9, tf.float32.max)
    
    def __init__(self, model_name: str):
        """
        Initialize the ConstantEmbedding class.

        Args:
            model_name (str): The name of the pre-trained model.

        Returns:
            None
        """
        super().__init__()
        self.lm = TFAutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lm.trainable = False
        
    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Applies the embedding model to the given inputs.

        Args:
            inputs (tf.Tensor): The input tensor to be embedded.

        Returns:
            tf.Tensor: The embedded tensor.
        """
        input_ids = self.tokenizer(inputs, return_tensors="tf", padding=True, truncation=True)['input_ids']
        model_output = self.lm(input_ids)
        return self.attended_mean_pooling(model_output, input_ids["attention_mask"])
        

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
            # print('Domain', domain.name)
            # print('Constants', len(domain.constants), 'Embedding size', constant_embedding_sizes_per_domain[domain.name])
            self.embedder[domain.name] = Embedding(
                len(domain.constants),
                constant_embedding_sizes_per_domain[domain.name],
                embeddings_regularizer=L2(regularization))

    # domain_inputs is Dict domain->tensor of idx
    def call(self, domain_inputs: Dict[str, tf.Tensor], **kwargs):
        domain_features = {}
        for domain in self.domains:
            domain_features[domain.name] = self.embedder[domain.name](
                domain_inputs[domain.name])
            # print('Domain', domain.name, 'inputs', domain_inputs[domain.name].shape,domain_inputs[domain.name]  )
            # print('Domain', domain.name, 'outputs', domain_features[domain.name].shape,domain_features[domain.name])
        # To use an LM to get the embed. of the constants, we need to give the word (not the index), along with its text
        # How should I do, before the serializer I work directly with the word to get the text? or once I have the indeces, I retrieve the word from each index?
        return domain_features
    
class LMCostantEmbeddings(LMEmbeddings):
    
    def __init__(self, model_name: str, domains: List[Domain], fol: FOL):
        """
        Initialize the ConstantEmbedding class.

        Args:
            model_name (str): The name of the pre-trained model.
            constant_embedding_size (int): The size of the constant embedding.
        """
        super().__init__(model_name)
        self._domains = domains
        self._embedder = {}
        
        # Initialize the embeddings for each domain, at least the one that i have at the moment
        for domain in domains:
            self._embedder[domain.name] = {}
            for constant in domain.constants:
                if self._embedder[domain.name].get(constant) is not None:
                    print(f"{'Warning'*10}: constant {constant} already exists in domain {domain.name}")
                self._embedder[domain.name][constant]= self(constant)
        
    def __call__(self, constant)
        
        