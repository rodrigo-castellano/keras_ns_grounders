from keras.layers import Dense, Embedding, Layer
from keras.models import Sequential
from keras.regularizers import L2
from typing import Dict, List, Tuple
import tensorflow as tf
import tensorflow_probability as tfp
import torch
from angle_emb import AnglE, Prompts
from ns_lib.logic.commons import Domain
from transformers import AutoModel, AutoTokenizer
from ns_lib.logic import FOL

class ConstantEmbeddings(Layer):
    """Calls the constant rules_embedders, differenciating the behavior of
       the single domains."""
    def __init__(self, domains: List[Domain],
                 constant_embedding_sizes_per_domain: Dict[str, int],
                 regularization: float=0.0,
                 has_features: bool=False):
        super().__init__()
        self.embedder = {}
        self.domains = domains
        for domain in domains:
            if has_features:
                # This should be replaced with the actual embedder,
                # make this a factory function call.
                self.embedder[domain.name] = Sequential([
                    Dense(constant_embedding_sizes_per_domain[domain.name],
                          kernel_regularizer=L2(regularization))])
            else:
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
        return domain_features
    

class ConstantEmbeddingsTrial(Layer):
    """Calls the constant rules_embedders, differenciating the behavior of
       the single domains."""
    def __init__(self, domains: List[Domain],
                 constant_embedding_sizes_per_domain: Dict[str, int],
                 regularization: float=0.0,
                 has_features: bool=False):
        super().__init__()
        self.embedder = {}
        self.domains = domains
       
        max_index = sum([len(domain.constants) for domain in domains]) 
        self.embedder = Embedding(
            max_index+ 1,
            constant_embedding_sizes_per_domain[domains[0].name],
            embeddings_regularizer=L2(regularization))

    # domain_inputs is Dict domain->tensor of idx
    def call(self, domain_inputs: Dict[str, tf.Tensor], **kwargs):
        domain_features = {}
        for domain in self.domains:
            # put the domain_inputs as a second key of a dict 
            domain_features[domain.name] = (domain_inputs[domain.name],self.embedder(
                domain_inputs[domain.name]))
        return domain_features

class ConstantEmbeddingsGlobal(Layer):
    """Calls the constant rules_embedders, differenciating the behavior of
       the single domains."""
    def __init__(self, domains: List[Domain],
                 constant_embedding_sizes_per_domain: Dict[str, int],
                 regularization: float=0.0,
                 has_features: bool=False):
        super().__init__()
        self.embedder = {}
        self.domains = domains
        self.constant_embedding_sizes_per_domain = constant_embedding_sizes_per_domain
        self.regularization = regularization
        self.has_features = has_features
        # I could make this more efficient: for domain 1, it goes from 0 to len(domain1), for dom2, it goes from (dom1,dom1+dom2)...
        max_index = sum([len(domain.constants) for domain in domains]) 
        for domain in domains:
            if self.has_features:
                # This should be replaced with the actual embedder,
                # make this a factory function call.
                self.embedder[domain.name] = Sequential([
                    Dense(self.constant_embedding_sizes_per_domain[domain.name],
                        kernel_regularizer=L2(self.regularization))])
            else:
                self.embedder[domain.name] = Embedding(
                    max_index+ 1,
                    self.constant_embedding_sizes_per_domain[domain.name],
                    embeddings_regularizer=L2(self.regularization))

    # domain_inputs is Dict domain->tensor of idx
    def call(self, domain_inputs: Dict[str, tf.Tensor], **kwargs):
        domain_features = {}
        cte_embeddings = {}
        for domain in self.domains:
            tf.print('X_Domain:', domain.name,summarize=-1)
            tf.print('Domain inputs:', domain_inputs[domain.name],summarize=-1)
            if domain_inputs[domain.name].shape[0] != 0:
                # if self.has_features:
                #     # This should be replaced with the actual embedder,
                #     # make this a factory function call.
                #     self.embedder[domain.name] = Sequential([
                #         Dense(self.constant_embedding_sizes_per_domain[domain.name],
                #             kernel_regularizer=L2(self.regularization))])
                # else:
                #     self.embedder[domain.name] = Embedding(
                #         tf.reduce_max(domain_inputs[domain.name]) + 1,
                #         self.constant_embedding_sizes_per_domain[domain.name],
                #         embeddings_regularizer=L2(self.regularization))

                domain_features[domain.name] = self.embedder[domain.name](
                    domain_inputs[domain.name])  #CE
                
                # Get the embeddings for the domain [200]
                embedding_size = self.constant_embedding_sizes_per_domain[domain.name] 
                # Assuming embeddings is a 2D tensor of shape (168, 200)
                embeddings = domain_features[domain.name]
                # tf.print('Embeddings:', embeddings.shape, embeddings[:,:3], summarize=-1)
                # Flatten the embeddings to have the shape (168 * 200,)
                embeddings_flattened = tf.reshape(embeddings, [-1])

                # The indices tensor
                indices = domain_inputs[domain.name]
                # Create the coordinates for each embedding. Each index should correspond to all 200 dimensions of an embedding
                indices_expanded = tf.expand_dims(indices, axis=1)
                coords = tf.concat([tf.repeat(indices_expanded, embedding_size, axis=0), 
                                    tf.tile(tf.range(embedding_size, dtype=tf.int32)[:, tf.newaxis], [indices.shape[0], 1])], axis=1)
                coords = tf.cast(coords, tf.int64)

                # The dense shape should reflect the maximum index and the size of each embedding
                dense_shape = [tf.reduce_max(indices).numpy() + 1, self.constant_embedding_sizes_per_domain[domain.name]]

                # Create the sparse tensor
                sparse_tensor = tf.SparseTensor(
                    indices=coords,  # List of coordinates
                    values=embeddings_flattened,  # List of values at those coordinates
                    dense_shape=dense_shape  # Shape of the dense tensor if it were fully populated
                )

                cte_embeddings[domain.name] = sparse_tensor 
            else:
                cte_embeddings[domain.name] = None
            print('Embeddings:', cte_embeddings[domain.name])
        return cte_embeddings


class PredicateEmbeddings(Layer):
    """Calls the predicate embedders."""
    def __init__(self, predicates: List[str],
                 predicate_embedding_size: int,
                 regularization: float=0.0,
                 has_features: bool=False):
        super().__init__()
        #self.predicates = predicates
        # self.predicate2index = {p:i for i,p in enumerate(predicates)}
        #self.table = tf.lookup.StaticHashTable(
        #    tf.lookup.KeyValueTensorInitializer(self.predicate2index.keys(),
        #                                        self.predicate2index.values()),
        #    default_value=-1)
        #self.regularization = regularization
        if has_features:
            # This should be replaced with the actual embedder,
            # make this a factory function call.
            self.embedder = Sequential([
                Dense(len(predicates), kernel_regularizer=L2(regularization))])
        else:
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
                 num_adaptive_constants: int,
                 dot_product: bool=False):
        super().__init__()
        self.adaptive_constant2relevance = {}
        self.constant_embedder = constant_embedder
        self.num_adaptive_constants = num_adaptive_constants
        self.domains = domains
        self.dot_product = dot_product
        for domain in domains:
            if not dot_product:
                self.adaptive_constant2relevance[domain.name] = (
                    Sequential([
                        Embedding(num_adaptive_constants, constant_embedding_size),
                    ]))
            else:
                self.adaptive_constant2relevance[domain.name] = (
                    Sequential([
                        Embedding(num_adaptive_constants, constant_embedding_size),
                        # Dense(len(domain.constants), activation=None),
                    ]))

    def call(self, domain_inputs, **kwargs):
        # TODO: check with Mike / Beppe
        # emb_inputs = self.constant_embedder(domain_inputs)

        constant_ranges = {domain.name:tf.range(0, len(domain.constants))  #CE
                            for domain in self.domains}
        constant_embeddings = self.constant_embedder(constant_ranges)

        embedder_outputs = {}
        for domain in self.domains:
            # Only constants that are outside of domain bounds are
            # adaptive and should be embedded.
            # We zero out non-adaptive constants, which will be anyway
            # discarded later by the caller.
            embedder_inputs = tf.math.maximum(
                0, domain_inputs[domain.name] - len(domain.constants))

            # TODO: check with Mike / Beppe
            if self.dot_product:
                adaptive_emb = self.adaptive_constant2relevance[domain.name](embedder_inputs)
                emb = constant_embeddings[domain.name]  #CE
                constant2relevance = tf.tensordot(adaptive_emb, tf.transpose(emb), axes=1)
                max_value = tf.reduce_max(constant2relevance, axis=-1, keepdims=True)
                mask = tf.cast(tf.equal(constant2relevance, max_value), adaptive_emb.dtype)
                embedder_outputs[domain.name] = tf.linalg.matmul(mask, emb)

            else:
                # Computes the distribution over the constants.
                logits = self.adaptive_constant2relevance[domain.name](
                    embedder_inputs)
                dist = tfp.distributions.Logistic(logits, 1.0)
                sample = tf.sigmoid(dist.sample())  # CC
                embedder_outputs[domain.name] = tf.linalg.matmul(
                    sample, constant_embeddings[domain.name])  #CE
                #deb = tf.math.argmax(embedder_outputs[domain.name], axis=-1)
                #tf.print('Selected constants domain', domain.name, deb)

        return embedder_outputs

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
    LLM = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

    LLM.trainable = False
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
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def __init__(self, fol: FOL, model_name: str):
        """
        Initialize the ConstantEmbedding class.

        Args:
            model_name (str): The name of the pre-trained model.

        Returns:
            None
        """
        # self.lm = AutoModel.from_pretrained(model_name, device_map = 'cuda')
        self.fol = fol
        
        
        # #Embed relation
        # self.embedded_relations = []
        # for predicate in self.fol.predicates:
        #     self.embedded_relations.append(1)
                
        # #Embed constants
        # self.embedded_constants = []
        # for domain in self.fol.domains:
        #     for constant in domain.constants:
        #         self.embedded_constants.append(1)
        
        # self.embedded_relations = tf.concat(self.embedded_relations, 0)
        # self.embedded_constants = tf.concat(self.embedded_constants, 0)
        
    def __call__(self, A_predicates: tf.Tensor) -> tf.Tensor:
        """
        Applies the embedding model to the given inputs.

        Args:
            inputs (tf.Tensor): The input tensor to be embedded.

        Returns:
            tf.Tensor: The embedded tensor.
        """
        # A_predicates_tensor = tf.convert_to_tensor(A_predicates)
        # batch_size = A_predicates_tensor.shape[0]
        # A_predicates_costants = tf.gather(self.embedded_constants,A_predicates_tensor[:,:2])
        # A_predicates_predicates = tf.expand_dims(tf.gather(self.embedded_relations, A_predicates_tensor[:,2]),1)
        # A_predicates_embeddings = tf.concat([A_predicates_costants, A_predicates_predicates], axis=1)  
        # #concat on the 2 dimension the embeddings of the constants and the predicates
        # A_predicates_embeddings  = tf.reshape(A_predicates_embeddings, [batch_size, -1])
        A_predicates_embeddings = self.LLM.encode(A_predicates)
        return A_predicates_embeddings