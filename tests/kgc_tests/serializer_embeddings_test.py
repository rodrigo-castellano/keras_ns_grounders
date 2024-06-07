import sys
import os
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../..'))
# sys.path.append(os.path.join(current_dir, '../..', 'ns_lib'))
import ns_lib as ns
# from dataset import KGCDataset
from ns_lib.grounding.backward_chaining_grounder import BackwardChainingGrounder
from ns_lib.logic.commons import Rule
import unittest
from dataset import KGCDataHandler
from typing import List, Tuple, Dict
from model import KGEModel

class SerializerTest(unittest.TestCase):


    def test_embeddings(self):
        "Check whether the serialization and embeddings are correct, and there is no stochasticity in the serialization/embeddings creating"

        data_handler = KGCDataHandler(dataset_name="dummy",
                             base_path="experiments\data")
        dataset_train = data_handler.get_dataset(split="train",number_negatives=2)

        rules = [Rule('r0', body_atoms=[("r", "X", "Y")],
                          head_atoms=[("r", "Y", "X")])]

        fol = data_handler.fol
        facts = list(data_handler.train_known_facts_set)
        domain2adaptive_constants: Dict[str, List[str]] = None

        engine = ns.grounding.BackwardChainingGrounder(
                    rules, facts=facts,
                    domains={d.name:d for d in fol.domains},
                    domain2adaptive_constants=domain2adaptive_constants,
                    pure_adaptive=False,
                    num_steps=1) 
        
        serializer = ns.serializer.LogicSerializerFast(
            predicates=fol.predicates, domains=fol.domains,
            constant2domain_name=fol.constant2domain_name,
            domain2adaptive_constants=domain2adaptive_constants)

  
        data_gen_train = ns.dataset.DataGenerator(
            dataset_train, fol, serializer, engine,
            batch_size=128, ragged=True,
            use_ultra=False, use_ultra_with_kge=False, 
            global_serialization=True)
 
        (X,Ap,Ar,Q,_), Y = data_gen_train[0]
        (X_f,Ap_f,Ar_f,Q_f,_), Y_f = data_gen_train[0]

        #X
        for k in X:
            self.assertEqual(X[k].numpy().shape, X_f[k].numpy().shape)

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

        print("Shapes are correct. Assesment of elements:")

        #X
        for k in X:
            self.assertTrue((X[k].numpy() == X_f[k].numpy()).all())

        #Ap
        for k in Ap:
            self.assertTrue((Ap[k].numpy() == Ap_f[k].numpy()).all())

        #Af
        for k in Ar:
            self.assertTrue((Ar[k][0].numpy() == Ar_f[k][0].numpy()).all()) #body
            self.assertTrue((Ar[k][1].numpy() == Ar_f[k][1].numpy()).all()) #tail

        #Q
        self.assertTrue((Q.numpy() == Q_f.numpy()).all())

        # print('Assesment of elements is correct. Now we will check the embeddings of the model.')
        # Different KGE have different inizialization of the embeddings
        kge_model = KGEModel(fol, 'complex',
                                0,
                                200,
                                200,
                                100,
                                0,
                                0,
                                False,
                                device='cpu',
                                use_ultra=False)

        concept_output, concept_embeddings = kge_model((X, Ap))
        print('\n\n\n\n')
        concept_output_1, concept_embeddings1 = kge_model((X_f, Ap_f))

        # print('Checking the shapes of the embeddings')

        #concept_embeddings
        # print('shape of concept_embeddings:', concept_embeddings.numpy().shape)
        self.assertEqual(concept_embeddings.numpy().shape, concept_embeddings1.numpy().shape)


        # print('Checking the values of the embeddings')
        # print('concept_embeddings:', concept_embeddings.numpy())
        # print('concept_embeddings_1:', concept_embeddings_1.numpy())
        self.assertTrue((concept_embeddings.numpy() == concept_embeddings1.numpy()).all())

        print('The embeddings are the same. The test is successful')

    def test_serializer(self):
        "Check whether the indices correspond through batches to the same constants"

        data_handler = KGCDataHandler(dataset_name="dummy",
                             base_path="experiments\data")
        dataset_train = data_handler.get_dataset(split="train",number_negatives=2)

        rules = [Rule('r0', body_atoms=[("r", "X", "Y")],
                          head_atoms=[("r", "Y", "X")])]

        fol = data_handler.fol
        facts = list(data_handler.train_known_facts_set)
        domain2adaptive_constants: Dict[str, List[str]] = None

        engine = ns.grounding.BackwardChainingGrounder(
                    rules, facts=facts,
                    domains={d.name:d for d in fol.domains},
                    domain2adaptive_constants=domain2adaptive_constants,
                    pure_adaptive=False,
                    num_steps=1) 
        
        serializer = ns.serializer.LogicSerializerFast(
            predicates=fol.predicates, domains=fol.domains,
            constant2domain_name=fol.constant2domain_name,
            domain2adaptive_constants=domain2adaptive_constants)

  
        data_gen_train = ns.dataset.DataGenerator(
            dataset_train, fol, serializer, engine,
            batch_size=128, ragged=True,
            use_ultra=False, use_ultra_with_kge=False, 
            global_serialization=True)
 
        (X,Ap,Ar,Q,_), Y = data_gen_train[0]
        print('\n\n\n')
        (X_f,Ap_f,Ar_f,Q_f,_), Y_f = data_gen_train[1]

        #X
        for k in X:
            self.assertEqual(X[k].numpy().shape, X_f[k].numpy().shape)

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

        # print("Shapes are correct. Assesment of elements:")

        #X
        for k in X:
            self.assertTrue((X[k].numpy() == X_f[k].numpy()).all())

        #Ap
        for k in Ap:
            self.assertTrue((Ap[k].numpy() == Ap_f[k].numpy()).all())

        #Af
        for k in Ar:
            self.assertTrue((Ar[k][0].numpy() == Ar_f[k][0].numpy()).all()) #body
            self.assertTrue((Ar[k][1].numpy() == Ar_f[k][1].numpy()).all()) #tail

        #Q
        self.assertTrue((Q.numpy() == Q_f.numpy()).all())





SerializerTest().test_serializer()