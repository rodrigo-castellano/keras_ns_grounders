from keras_ns.nn.kge import AtomEmbeddingLayer, DistMult, HyperPlaneAtomEmbedder, ComplEx, TransE
from keras_ns.nn.reasoning import ReasoningLayer, DeepLogicModelReasoner, GradientDescentUpdate, DCRReasoningLayer
from keras_ns.nn.constant_embedding import AdaptiveConstantEmbeddings, ConstantEmbeddings
from keras_ns.nn.constant_embedding import DomainWiseMLP, ExplicitDomainEmbedders
import tensorflow as tf
from keras.layers import Dense

def create_sequential(size_activation_pairs, regularization = 0):
    seq = tf.keras.Sequential()
    for s,a in size_activation_pairs:
        seq.add(Dense(s,activation=a, kernel_regularizer=tf.keras.regularizers.l2(regularization)))
    return seq
