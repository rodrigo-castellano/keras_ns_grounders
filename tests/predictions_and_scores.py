from keras_ns.logic import FOL, Domain, Predicate
from keras_ns.dataset import Dataset
import keras_ns as ns
from keras_ns.layers import LatentWorldSemanticReasoningLayerStacked
import numpy as np
import tensorflow as tf
from keras_ns.logic.commons import Rule, Atom




d = Domain("domain",["c1", "c2", "c3"])
a = Predicate("a", [d, d])
b = Predicate("b", [d, d])
c = Predicate("c", [d, d])
predicates = [a,b,c]
facts = ["a(c1,c1)", "b(c1,c1)", "a(c2,c2)", "b(c2,c2)"]
facts = [Atom(s=c) for c in facts]
queries = [["c(c1,c1)", "c(c2,c2)", "c(c3,c3)"], ["b(c1,c1)", "b(c2,c2)", "b(c3,c3)"]]
queries = [[Atom(s=c) for c in q] for q in queries]
fol = FOL(domains= [d], predicates = predicates , facts = facts)


dataset = Dataset(queries=queries, labels=[[1 for i in q] for q in queries], constants_features=None, format=format)


rules = [Rule("id1", weight= 1.0,
                 body =["a(X,Y)","b(X,Y)"] ,
                 head = ["c(X,Y)"])]

engine = ns.logic.BackwardChainingGrounder(facts=list(facts), rules=rules)
serializer = ns.logic.LogicSerializer(domains=[d], predicates=predicates, debug=True)
data_gen_train = ns.dataset.DataGenerator(dataset,fol, serializer, engine, batch_size=2)

for (X,A_atoms, A_formulas, Q),labels  in data_gen_train:
  s = [[serializer.get_atom_str(a) for a in q] for q in Q.numpy()]
  print(s)
