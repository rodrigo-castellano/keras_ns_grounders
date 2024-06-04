import keras_ns as ns
from dataset import KGCDataset
from keras_ns.grounding.backward_chaining_grounder import BackwardChainingGrounder
from keras_ns.logic.commons import Rule
import unittest




class SerializerTest(unittest.TestCase):


    def test_shapes(self):
        "Check whether LogicSerializer and LogicSerializerFast behave identically"

        dataset = KGCDataset(dataset_name="dummy",
                             base_path="data/")
        dataset_train = dataset.get_train()

        rules = [Rule('r0', body_atoms=[("r", "X", "Y")],
                          head_atoms=[("r", "Y", "X")])]

        engine= BackwardChainingGrounder(rules,list(dataset_train.known_facts))

        serializer_fast = ns.logic.LogicSerializerFast(
            predicates=dataset.predicates, domains=dataset.domains)
        serializer = ns.logic.LogicSerializer(
            predicates=dataset.predicates, domains=dataset.domains)

        data_gen_train = ns.dataset.DataGenerator(
            dataset_train, serializer, engine,
            batch_size=2, ragged=True)

        data_gen_train_fast = ns.dataset.DataGenerator(
            dataset_train, serializer_fast, engine,
            batch_size=2, ragged=True)


        (X,Ap,Ar,Q), Y = data_gen_train[0]
        (X_f,Ap_f,Ar_f,Q_f), Y_f = data_gen_train_fast[0]

        #X
        for k in X:
            self.assertEqual(X[k].numpy().shape, X[k].numpy().shape)

        #Ap
        for k in Ap:
            self.assertEqual(Ap[k].numpy().shape, Ap_f[k].numpy().shape)

        #Af
        for k in Ar:
            self.assertEqual(Ar[k][0].numpy().shape, Ar_f[k][0].numpy().shape) #body
            self.assertEqual(Ar[k][1].numpy().shape, Ar_f[k][1].numpy().shape) #tail


        #Q
        self.assertEqual(Q.numpy().shape, Q_f.numpy().shape)

        #Y
        self.assertEqual(Y.numpy().shape, Y_f.numpy().shape)




SerializerTest().test_serializer_fast()