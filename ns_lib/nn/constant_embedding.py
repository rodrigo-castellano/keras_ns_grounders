from collections import OrderedDict, defaultdict
import os
import pickle
from keras.layers import Dense, Embedding, Layer
from keras.models import Sequential
from keras.regularizers import L2
from typing import Dict, List, Tuple
import tensorflow as tf
import tensorflow_probability as tfp
import torch
from ns_lib.logic.commons import Domain
# from transformers import AutoModel, AutoTokenizer
from ns_lib.logic import FOL
import wikipediaapi as wk
# from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np 
from sklearn.decomposition import PCA

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
    

class ConstantEmbeddings_Global(Layer):
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
    # Load model from HuggingFace Hub
    # tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
    # model = AutoModel.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
    embedded_constants = {}
    @staticmethod
    def encode(sentences) -> tf.Tensor:
        """
        Performs mean pooling on the token embeddings based on the attention mask.

        Args:
            model_output (tf.Tensor): The output of the model.
            attention_mask (tf.Tensor): The attention mask to indicate which tokens to include in the pooling.

        Returns:
            tf.Tensor: The mean-pooled token embeddings.

        """
        # Tokenize sentences
        encoded_input = LMEmbeddings.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = LMEmbeddings.model(**encoded_input)
    
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return F.normalize(embedding, p=2, dim=1)
    
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
        self.constant_to_global_unique_index = defaultdict(OrderedDict)
        self.global_unique_index_to_constant = defaultdict(OrderedDict)
        counter = 0
        for domain in self.fol.domains:
            for constant in domain.constants:
                self.constant_to_global_unique_index[domain.name][constant] = counter
                self.global_unique_index_to_constant[counter] = [domain.name, constant]
                counter += 1
        
        self.predicate_to_global_unique_index = defaultdict(OrderedDict)
        self.global_unique_index_to_predicate = defaultdict(OrderedDict)
        counter = 0
        for predicate in self.fol.predicates:
            self.predicate_to_global_unique_index[predicate] = counter
            self.global_unique_index_to_predicate[counter] = predicate
            counter += 1
            
        self.wikipedia = wk.Wikipedia('MyProjectName (merlin@example.com)', 'en')
        

        #Embed relation
        predicates_description = ["""The predicate locatedInCR(countries,regions)locatedInCR(countries,regions) indicates a relationship where a country is geographically situated within a specified region. It asserts that the entity represented by "countries" is contained within the spatial boundaries of the entity represented by "regions.""",
                                """The predicate asserts that a country is situated within a specific subregion. Here, "countries" represents the country or countries, and "subregions" represents the subregions within which these countries are located.""",
                                """The predicate asserts that a subregion is situated within a specific region. In this context, "subregions" represents the subregion or subregions, and "regions" represents the regions within which these subregions are located.""",
                                """The predicate asserts that two countries share a common border. Here, both arguments represent countries that are geographically adjacent to each other."""]
        
        
        if os.path.exists("constants_pca.pkl") and os.path.exists("relations_pca.pkl"):
            self.embedded_constants_tensor = pickle.load(open("constants_pca.pkl", "rb"))
            self.embedded_relations_tensor = pickle.load(open("relations_pca.pkl", "rb"))
            return
        
        self.embedded_relations = []
        for predicate, description in zip(self.fol.predicates,predicates_description):
            self.embedded_relations.append(self.encode(description))
        self.embedded_relations_tensor = tf.convert_to_tensor(np.array(self.embedded_relations))
                
        #Embed constants
        
        print("Embedding constants...")
        self.embedded_constants_tensor = []
        for domain in self.fol.domains:
            print("Embedding constants for domain: ", domain.name)
            LMEmbeddings.embedded_constants[domain.name] = {}
            for constant in tqdm(domain.constants):
                if constant in LMEmbeddings.embedded_constants[domain.name]:
                    continue
                self.embedded_constants[domain.name][constant] = self.encode(self.wikipedia.page(constant).summary.split(".")[0])
                self.embedded_constants_tensor.append(self.embedded_constants[domain.name][constant])
        self.embedded_constants_tensor = tf.convert_to_tensor(np.array(self.embedded_constants_tensor))
        
        self.pca = PCA(n_components=128)
        self.embedded_constants_tensor = pickle.load(open("constants.pkl", "rb"))
        self.pca.fit(tf.concat([self.embedded_relations_tensor, self.embedded_constants_tensor], 0)[:,0,:]) # (num_predicates + num_constants, embedding_size)
        self.embedded_relations_tensor = self.pca.transform(self.embedded_relations_tensor[:,0,:])
        self.embedded_constants_tensor = self.pca.transform(self.embedded_constants_tensor[:,0,:])
        pickle.dump(self.embedded_constants_tensor, open("constants_pca.pkl", "wb"))
        pickle.dump(self.embedded_relations_tensor, open("relations_pca.pkl", "wb"))
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
        A_as_np_array = tf.convert_to_tensor(A_predicates)
        
        constants_embeddings = tf.gather(self.embedded_constants_tensor, A_as_np_array[:,:2]) # (batch_size, 2, embedding_size)
        
        predicate_embeddings = tf.gather(self.embedded_relations_tensor, A_as_np_array[:,2]) # (batch_size, embedding_size)
        predicate_embeddings = tf.expand_dims(predicate_embeddings, 1) # (batch_size, 1, embedding_size)
        return constants_embeddings, predicate_embeddings