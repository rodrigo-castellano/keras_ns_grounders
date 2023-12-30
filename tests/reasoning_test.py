from keras_ns.grounding.backward_chaining_grounder import BackwardChainingRule
from keras_ns.logic.commons import Atom, Predicate, Domain
from keras_ns.data import Dataset
import keras_ns as ns
from keras_ns.layers import LatentWorldSemanticReasoningLayerStacked
import numpy as np
import tensorflow as tf


d = Domain("domain",["c1", "c2", "c3"])
a = Predicate("a", [d, d])
b = Predicate("b", [d, d])
c = Predicate("c", [d, d])
facts = ["a(c1,c1)", "b(c1,c1)", "a(c2,c2)", "b(c2,c2)"]
facts = [Atom(s=c) for c in facts]
queries = ["c(c1,c1)", "c(c2,c2)", "c(c3,c3)"]
queries = [Atom(s=c) for c in queries]

dataset = Dataset(domains = [d], predicates= [a,b,c],queries= [queries],labels= [[1 for i in queries]],
         constants_features = None,facts= facts)

rules = [BackwardChainingRule("id1", weight= 1.0,
                 body =["a(X,Y)","b(X,Y)"] ,
                 head = ["c(X,Y)"])]

engine = ns.logic.BackwardChainingGrounding(facts=list(facts), rules=rules)
serializer = ns.logic.LogicSerializer(predicates=[a,b,c], constants=[d])
data_gen_train = ns.data.DataGenerator(dataset, serializer, engine)

# This is  done in ns.data.DataGenerator
ground_formulas = engine.ground(facts=facts, queries=queries)
(_, _, A_rules_data, _), _ = data_gen_train[0]

print("Checking the atoms created and indexed")
print(serializer.atom_to_index)

print("Checking str -> index -> str conversion")
print(ground_formulas[0].groundings[0])
in_, out_ = A_rules_data["id1"]
in_, out_ = list(in_.numpy()), list(out_.numpy())
print(in_[0], out_[0])
print([serializer.get_atom_str(int(a)) for a in np.concatenate((in_[0],out_[0]),axis=0)])


print("Reasoning")
atom_embeddings = np.zeros([9, 1])

atom_embeddings[0,0] = 1 #a(c1,c1)
atom_embeddings[3,0] = 1 #b(c1,c1)
atom_embeddings[1,0] = 1 #a(c2,c2)

reasoner = LatentWorldSemanticReasoningLayerStacked(rules, formula_hidden_size=1, atom_embedding_size=1,
                 resnet_rule = True, forward_formulas=False, regularization=0,
                 dropout_rate=0.0, rule_weight_embedding_size=1,
                 hard_rule = True, semiring = "product")

# Making also the resnet rule hard
reasoner._resnet_rul_var = tf.Variable(initial_value=1000 * tf.ones(shape=[1, 1]))

aear = atom_embeddings_after_reasoning = reasoner([atom_embeddings, A_rules_data]).numpy()
for i in serializer.index_to_atom:
    print(serializer.index_to_atom[i], aear[i,0])


