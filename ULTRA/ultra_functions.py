# In case I want to use ultra, substitute the DataGenerator class by this one, same with the rest of the functions


# RUN.PY
sys.path.append(os.path.join(current_dir, '..', 'ULTRA'))
LLM = False
ULTRA = False
ULTRA_WITH_KGE = False
args.use_ultra = ULTRA
args.use_ultra_with_kge = ULTRA_WITH_KGE
args.use_llm = LLM
if args.use_ultra:
    args.run_signature = 'ultra-'+args.run_signature 
elif args.use_ultra_with_kge:
    args.run_signature = 'ultra_kge-'+args.run_signature
elif args.use_llm:
    args.run_signature = 'llm-'+args.run_signature

# TRAINING.PY
from ns_lib.dataset import get_ultra_datasets


# DATASET PREPARATION
data_handler = KGCDataHandler(
    dataset_name=args.dataset_name,
    base_path=data_path,
    format=get_arg(args, 'format', None, True),
    domain_file= args.domain_file,
    train_file= args.train_file,
    valid_file=args.valid_file,
    test_file= args.test_file,
    fact_file= args.facts_file)

dataset_train = data_handler.get_dataset(split="train",number_negatives=args.num_negatives)
dataset_valid = data_handler.get_dataset(split="valid",number_negatives=args.valid_negatives, corrupt_mode=args.corrupt_mode)
dataset_test = data_handler.get_dataset(split="test",  number_negatives=args.test_negatives,  corrupt_mode=args.corrupt_mode)
if explain_enabled and enable_rules and (args.model_name == 'dcr' or args.model_name == 'cdcr'):
    dataset_test_positive_only = data_handler.get_dataset(split="test", number_negatives=0, corrupt_mode=args.corrupt_mode)

fol = data_handler.fol
domain2adaptive_constants: Dict[str, List[str]] = None
dot_product = get_arg(args, 'engine_dot_product', False)

num_adaptive_constants = get_arg(args, 'engine_num_adaptive_constants', 0)



# DEFINING RULES AND GROUNDING ENGINE
rules = []
engine = None

enable_rules = (args.reasoner_depth > 0 and args.num_rules > 0)
if enable_rules:
    rules = ns.utils.read_rules(join(data_path, args.dataset_name, args.rules_file),args)
    facts = list(data_handler.train_known_facts_set)
    engine = BuildGrounder(args, rules, facts, fol, domain2adaptive_constants)
serializer = ns.serializer.LogicSerializerFast(
    predicates=fol.predicates, domains=fol.domains,
    constant2domain_name=fol.constant2domain_name,
    domain2adaptive_constants=domain2adaptive_constants)

if args.use_ultra or args.use_ultra_with_kge:
        train_ultra, valid_ultra, test_ultra, _ = get_ultra_datasets(dataset_train, dataset_valid, dataset_test,data_handler,
                                                                        serializer,engine,global_serialization=args.global_serialization)
else:
    train_ultra, valid_ultra, test_ultra, _ = None, None, None, None



# DATA GENERATORS
print('***********Generating train data**************')
start = time.time()
data_gen_train = ns.dataset.DataGenerator(
    dataset_train, fol, serializer, engine,
    batch_size=args.batch_size, ragged=ragged,
    use_ultra=args.use_ultra, use_ultra_with_kge=args.use_ultra_with_kge,use_llm=args.use_llm,
    global_serialization=args.global_serialization, dataset_ultra=train_ultra)
end = time.time()
args.time_ground_train = np.round(end - start,2)
print("Time to create data generator train: ", np.round(end - start,2),'\n************************************')
start = time.time()
data_gen_valid = ns.dataset.DataGenerator(
    dataset_valid, fol, serializer, engine,
    batch_size=args.val_batch_size, ragged=ragged,
    use_ultra=args.use_ultra, use_ultra_with_kge=args.use_ultra_with_kge,use_llm=args.use_llm,
    global_serialization=args.global_serialization, dataset_ultra=valid_ultra)
end = time.time()
args.time_ground_valid = np.round(end - start,2)
print("Time to create data generator valid: ",  np.round(end - start,2),'\n************************************') 

start = time.time()
data_gen_test = ns.dataset.DataGenerator(
    dataset_test, fol, serializer, engine,
    batch_size=args.test_batch_size, ragged=ragged,
    use_ultra=args.use_ultra, use_ultra_with_kge=args.use_ultra_with_kge,use_llm=args.use_llm,
    global_serialization=args.global_serialization, dataset_ultra=test_ultra)
end = time.time()
args.time_ground_test = np.round(end- start,2)
print("Time to create data generator test: ",  np.round(end - start,2),'\n************************************')







# MODEL.PY

class KGEModel(Model):

    def __init__(self, fol:FOL,
                 kge: str,
                 kge_regularization: float,
                 constant_embedding_size: int,
                 predicate_embedding_size: int,
                 kge_atom_embedding_size: int,
                 kge_dropout_rate: float,
                 num_adaptive_constants: int=0,
                 dot_product: bool=False,
                 use_ultra: bool=False,
                 device: str = 'cpu',
                 global_serialization: bool = False):
        super().__init__()
        self.fol = fol
        self.predicate_index_tensor = tf.constant(
            [i for i in range(len(self.fol.predicates))], dtype=tf.int32)
        
        self.global_serialization = global_serialization
        
        # CONSTANT AND PREDICATE EMBEDDINGS
        self.use_ultra = use_ultra
        if not self.use_ultra:
            self.predicate_embedder = PredicateEmbeddings(
                fol.predicates,
                predicate_embedding_size,
                regularization=kge_regularization,
                has_features=False)
            if self.global_serialization:
                cte_embedder = ConstantEmbeddings_Global
            else:
                cte_embedder = ConstantEmbeddings
            self.constant_embedder = cte_embedder( 
                domains=fol.domains,
                constant_embedding_sizes_per_domain={
                    domain.name: constant_embedding_size
                    for domain in fol.domains},
                regularization=kge_regularization,
                has_features=False)
            self.dot_product = dot_product
        else: 
            # To adapt the size of the embeddings given by ultra
            self.predicate_projection = Sequential([
                Dense(128, activation='relu'),
                Dense(114, activation='relu'),
                Dense(100, activation='relu'),
                Dense(100, activation='relu')])
            
            self.constant_projection = Sequential([
                Dense(128, activation='relu'),
                Dense(114, activation='relu'),
                Dense(100, activation='relu'),
                Dense(100, activation='relu')])
        
        if num_adaptive_constants > 0:
            self.adaptive_constant_embedder = AdaptiveConstantEmbeddings(
                domains=fol.domains,
                constant_embedder=self.constant_embedder,
                constant_embedding_size=constant_embedding_size,
                num_adaptive_constants=num_adaptive_constants,
                dot_product=dot_product)
        else:
            self.adaptive_constant_embedder = None

        # ATOM EMBEDDINGS
        self.kge_embedder, self.output_layer = KGEFactory(
            name=kge,
            atom_embedding_size=kge_atom_embedding_size,
            relation_embedding_size=kge_atom_embedding_size,
            regularization=kge_regularization,
            dropout_rate=kge_dropout_rate)
        assert self.kge_embedder is not None

    # def create_triplets(self,
    #                     constant_embeddings: Dict[str, tf.Tensor],
    #                     predicate_embeddings: tf.Tensor,
    #                     A_predicates: Dict[str, tf.Tensor],
    #                     X_domains: Dict[str, tf.Tensor]):
    #     predicate_embeddings_per_triplets = []
        
    #     for p, indices in A_predicates.items():
    #         idx = self.fol.name2predicate_idx[p]
    #         p_embeddings = tf.expand_dims(predicate_embeddings[idx], axis=0)
    #         predicate_embeddings_per_triplets.append(tf.repeat(p_embeddings, tf.shape(indices)[0], axis=0))
        
    #     predicate_embeddings_per_triplets = tf.concat(predicate_embeddings_per_triplets, axis=0)
    #     constant_embeddings_for_triplets = []

    #     for p, constant_idx in A_predicates.items():
    #         constant_idx = tf.cast(constant_idx, tf.int32)
    #         predicate = self.fol.name2predicate[p]
    #         one_predicate_constant_embeddings = []
            
    #         for i, domain in enumerate(predicate.domains):
    #             if self.global_serialization:
    #                 indices_tensor = constant_embeddings[domain.name][0]
    #                 embeddings_tensor = constant_embeddings[domain.name][1]
                    
    #                 # Check if the tensors are empty
    #                 if tf.size(indices_tensor) > 0 and tf.size(embeddings_tensor) > 0:
    #                     # Create a lookup table
    #                     table = tf.lookup.StaticHashTable(
    #                         tf.lookup.KeyValueTensorInitializer(indices_tensor, tf.range(tf.shape(indices_tensor)[0])),
    #                         default_value=-1
    #                     )
                        
    #                     # Look up the indices
    #                     lookup_indices = table.lookup(constant_idx[..., i])
                        
    #                     # Gather the embeddings
    #                     constants = tf.gather(embeddings_tensor, lookup_indices)
    #                     one_predicate_constant_embeddings.append(constants)
    #             else:
    #                 constants = tf.gather(constant_embeddings[domain.name], constant_idx[..., i], axis=0)
    #                 one_predicate_constant_embeddings.append(constants)
            
    #         if one_predicate_constant_embeddings:
    #             one_predicate_constant_embeddings = tf.stack(one_predicate_constant_embeddings, axis=-2)
    #             constant_embeddings_for_triplets.append(one_predicate_constant_embeddings)
        
    #     if constant_embeddings_for_triplets:
    #         constant_embeddings_for_triplets = tf.concat(constant_embeddings_for_triplets, axis=0)
    #         tf.debugging.assert_equal(tf.shape(predicate_embeddings_per_triplets)[0],
    #                                 tf.shape(constant_embeddings_for_triplets)[0])
    #     else:
    #         constant_embeddings_for_triplets = tf.zeros_like(predicate_embeddings_per_triplets)
        
    #     return predicate_embeddings_per_triplets, constant_embeddings_for_triplets


    def create_triplets(self,
                        constant_embeddings: Dict[str, tf.Tensor],
                        predicate_embeddings: tf.Tensor,
                        A_predicates: Dict[str, tf.Tensor],
                        X_domains: Dict[str, tf.Tensor]):
        predicate_embeddings_per_triplets = []
        '''For A_predicates, take the emebdding representation of the predicates and the constants and create the triplets for the KGE model.
        For instance, if I have a predicate with 3 grounded atoms, I will repeat the embedding of that predicate 3 times, and put it with the 
        embeddings of the constants for each grounded atom
        - output: 
            predicate_embeddings_per_triplets: [n_predicates, n_atoms/grounding per predicate, embed_size_predicate]
            constant_embeddings_for_triplets: [n_atoms,2=n_domains,embed_size_constant]'''
            
        # tf.print('\nKGE X_domains')
        # for domain,constant_idx in X_domains.items():
        #     tf.print('KGE X_domains',domain,constant_idx,summarize=-1)
        # tf.print('KGE A_predicates')
        # for p,constant_idx in A_predicates.items():
        #     tf.print('KGE A_predicates',p,constant_idx,summarize=-1)
        # tf.print('KGE constant_embeddings')
        # for domain,constant_idx in constant_embeddings.items():
        #     tf.print('KGE constant_embeddings',domain,constant_idx.shape,summarize=-1)  
        # tf.print('KGE predicate_embeddings')
        # for predicate_embeddings in predicate_embeddings:
        #     tf.print('KGE predicate_embeddings',predicate_embeddings.shape,summarize=-1)

        for p,indices in A_predicates.items():
            idx = self.fol.name2predicate_idx[p]
            # Repeat the predicate embedding for each atom in the predicate
            p_embeddings = tf.expand_dims(predicate_embeddings[idx], axis=0)  # [1,1,200]=[1,1, embed_size_predicate]
            predicate_embeddings_per_triplets.append(tf.repeat(p_embeddings, tf.shape(indices)[0], axis=0))  # [1,1918,200]=[1,n_groundings for that predicate, embed_size_predicate]
        predicate_embeddings_per_triplets = tf.concat(predicate_embeddings_per_triplets,axis=0) # shape=[n_predicates, n_groundings for that predicate, embed_size_predicate]
        constant_embeddings_for_triplets = []

        for p,constant_idx in A_predicates.items():
            constant_idx = tf.cast(constant_idx, tf.int32) # all the groundings idx for that predicate
            predicate = self.fol.name2predicate[p]
            one_predicate_constant_embeddings = []
            '''Here, for each domain of the predicate (LocInSR(subregion,region)->for subregion), I get the embeddings of the constants that
            are grounded in that domain. If LocInSR has 58 groundings/atoms, I will get the representation of the subregion constants in 
            those atoms(58,200). I do the same for the domain region, so I get a tensor of shape [58,2,200] for LocIn where 2 is the arity of the predicate.''' 
            for i,domain in enumerate(predicate.domains):
                '''If I have A_pred=[country,region]=[[1,2],[3,4],...], I get for country: [1,3] which are local! they're the pos of the ctes in X_domain
                In X_domain, in pos i I have the global idx of that cte (which has been created to create the embedds)'''
                if self.global_serialization:
                    # Step 1: Create a dictionary that maps each index to its embedding
                    index_to_embedding = {index: emb for index, emb in zip(constant_embeddings[domain.name][0].numpy(), constant_embeddings[domain.name][1])}
                    # Step 2: Fetch the embeddings for the indices in `indices_list`
                    resulting_embeddings = [index_to_embedding[idx] for idx in constant_idx[..., i].numpy()]
                    # Step 3: Convert the list of embeddings to a tensor
                    constants = tf.convert_to_tensor(resulting_embeddings)
                    # Get the dense embeddings
                    # constants = self.get_embeddings_idx(constant_embeddings[domain.name], constant_idx[..., i]) if constant_embeddings[domain.name] is not None else tf.constant([], dtype=tf.float32)
                    one_predicate_constant_embeddings.append(constants)  if len(constants) > 0 else None
                else:
                    constants = tf.gather(constant_embeddings[domain.name],
                                        constant_idx[..., i], axis=0) # constant_idx[..., i] takes the idx of the constants for that domain (in predicate p)
                    one_predicate_constant_embeddings.append(constants)
                    
            # shape (predicate_batch_size, predicate_arity, constant_embedding_size)
            if self.global_serialization:
                one_predicate_constant_embeddings = tf.stack(one_predicate_constant_embeddings,axis=-2) if len(one_predicate_constant_embeddings) > 0 else tf.constant([], dtype=tf.float32)
                constant_embeddings_for_triplets.append(one_predicate_constant_embeddings) if len(one_predicate_constant_embeddings) > 0 else None
            else:
                one_predicate_constant_embeddings = tf.stack(one_predicate_constant_embeddings,axis=-2)
                constant_embeddings_for_triplets.append(one_predicate_constant_embeddings)
            
        '''For all the queries, I have divided them by predicates. Once I have, for each predicate, the embeddings of the constants, i.e., 
        for LocInSR I have 58 atoms -> (58,2,200), for NeighOf .... I concatenate them to get a tensor of shape [58+..,2,200] = [3889,2,200]
        tf.print('PREDICATE EMBEDDINGS PER TRIPLETS',predicate_embeddings_per_triplets.shape,predicate_embeddings_per_triplets)
        tf.print('CONSTANT EMBEDDINGS FOR TRIPLETS', [tensor.shape for tensor in constant_embeddings_for_triplets])'''

        constant_embeddings_for_triplets = tf.concat(constant_embeddings_for_triplets,axis=0) 
        tf.debugging.assert_equal(tf.shape(predicate_embeddings_per_triplets)[0],
                                  tf.shape(constant_embeddings_for_triplets)[0])
        # Shape TE, T2E with T number of triplets.
        # At the end I get for both predicates and embeddings a tensor of shape [n_atoms,ctes_in_atoms(arity),embed_size_predicate] and [n_atoms,2,embed_size_constant]
        return predicate_embeddings_per_triplets, constant_embeddings_for_triplets

    def call(self, inputs,embeddings=None):
        '''
        X_domains type is Dict[str, inputs]
        A_predicate: Dict[predicate_name, List[Tuple[Index1, ..., IndexN]]]
        For x_domains, I get each domain value (country,region...) represented by a index
        For A_predicates, I get the predicate name and the constant indices for each grounding
        '''
        (X_domains, A_predicates) = inputs
        if self.adaptive_constant_embedder is not None:
            # Create a mask to fix the values that are not in the domain.
            X_domains_fixed_mask = {
                name:tf.where(x < len(self.fol.name2domain[name].constants),
                              True, False) for name,x in X_domains.items()}
            # Set to 0 the values that are not in the domain.
            X_domains_fixed = {
                name:tf.where(X_domains_fixed_mask[name], x, 0)
                for name,x in X_domains.items()}
            # Get the embeddings for the fixed values and the adaptive values.
            constant_embeddings_fixed = self.constant_embedder(X_domains_fixed)
            constant_embeddings_adaptive = self.adaptive_constant_embedder(
                X_domains)
            constant_embeddings = {
                name:tf.where(
                    # Expand dim to broadcast to the embeddings size.
                    tf.expand_dims(X_domains_fixed_mask[name], axis=-1),
                    constant_embeddings_fixed[name],
                    constant_embeddings_adaptive[name])
                for name in X_domains.keys()}
        else: 
            if not self.use_ultra:
                constant_embeddings = self.constant_embedder(X_domains) # For the constant embedds, I always need global idx to get consistent embedds every batch

        if not self.use_ultra:
            # print('USING KGE')
            predicate_embeddings = self.predicate_embedder(self.predicate_index_tensor) # Embedds for every pred in fol (global idx)
        else: 
            print('USING ULTRA/LLM WITH KGE')
            (constant_embeddings, predicate_embeddings) = embeddings
            # Project embeddings to the new size
            predicate_embeddings = self.predicate_projection(predicate_embeddings)
            constant_embeddings = {k: self.constant_projection(v) for k, v in constant_embeddings.items()} 

        # Given the embedds of the constants and the predicates, I create the triplets with the embeddings of the atoms and the predicates. 
        # A_predicates indicates the indeces of the constants for each grounding of the predicate, i.e., the queries
        predicate_embeddings_per_triplets, constant_embeddings_for_triplets = \
            self.create_triplets(constant_embeddings, predicate_embeddings, A_predicates,X_domains) 
        
        # Given the triplets with their embeddings obtained in create_triplets, I get the embeddings of the atoms with e.g. Transe
        atom_embeddings = self.kge_embedder((predicate_embeddings_per_triplets,
                                             constant_embeddings_for_triplets))
        # Get the score for each atom    CAREFFUUUUUUUUUUL HERE
        atom_outputs = tf.expand_dims(self.output_layer(atom_embeddings), -1)
        return atom_outputs, atom_embeddings
    

class ULTRA_bridge(Model):

    def __init__(self):
        super().__init__()
        
        # To adapt the size of the embeddings given by ultra
        self.atom_projection = Sequential([
            Dense(128, activation='relu'),
            # Dense(128, activation='relu'),
            Dense(114, activation='relu'),
            Dense(100, activation='relu'),
            Dense(100, activation='relu'),])
        
        self.atom_score_projection = Sequential([
            Dense(1,activation='sigmoid')])

        # Define the output layer as a method
        self.output_layer = self._output_layer

    def _output_layer(self, inputs):
        # outputs = tf.reduce_sum(inputs, axis=-1)
        # outputs = tf.nn.sigmoid(outputs)
        outputs = self.atom_score_projection(inputs)
        outputs = tf.squeeze(outputs, 1)
        return outputs

    def call(self,embeddings):
        # Project embeddings to the new size
        atom_embeddings = self.atom_projection(embeddings)
        # return atom_embeddings

        # Get the score for each atom
        atom_outputs = tf.expand_dims(self.output_layer(atom_embeddings), -1) 
        return atom_outputs,atom_embeddings

class LLM_Bridge(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   
        self.predicate_projection = Sequential([
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(32, activation='relu')])
            
        self.constant_projection = Sequential([
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu')])
            
        self.dense_output = Sequential([
            Dense(1),
            Activation("sigmoid")
        ]) 
        self.output_layer = lambda inputs: tf.squeeze(self.dense_output(inputs),1)
    
    def call(self,embeddings):
        # Project embeddings to the new size
        (constant_embeddings, predicate_embeddings) = embeddings
        constant_embeddings_head = self.constant_projection(constant_embeddings[:,0,:])
        constant_embeddings_tail = self.constant_projection(constant_embeddings[:,1,:])
        predicate_embeddings = self.predicate_projection(predicate_embeddings)[:,0,:]
        atom_embeddings = tf.concat([constant_embeddings_head, constant_embeddings_tail, predicate_embeddings], axis=-1)
        # Get the score for each atom
        atom_outputs = tf.expand_dims(self.output_layer(atom_embeddings), -1) 
        return atom_outputs,atom_embeddings
    
 
class CollectiveModel(Model):

    def __init__(self,
                 fol: FOL,
                 rules: List[Rule],
                 *,  # all named after this point
                 use_ultra: bool,
                 use_ultra_with_kge: bool,
                 use_llm: bool,
                 kge: str,
                 kge_regularization: float,
                 constant_embedding_size: int,
                 predicate_embedding_size: int,
                 kge_atom_embedding_size: int,
                 kge_dropout_rate: float,
                 reasoner_atom_embedding_size: int,
                 reasoner_formula_hidden_embedding_size: int,
                 reasoner_regularization: float,
                 reasoner_single_model: bool,
                 reasoner_dropout_rate: float,
                 reasoner_depth: int,
                 aggregation_type: str,
                 signed: bool,
                 temperature: float,
                 model_name: str,
                 resnet: bool,
                 embedding_resnet: bool,
                 filter_num_heads: int,
                 filter_activity_regularization: float,
                 num_adaptive_constants: int,
                 dot_product: bool,
                 cdcr_use_positional_embeddings: bool,
                 cdcr_num_formulas: int,
                 r2n_prediction_type: str,
                 device: str = 'cpu',
                 global_serialization: bool = False):
        super().__init__()

        self.testing = False
        self.reasoner_depth = reasoner_depth
        # Reasoning depth currently used, this can differ from  self.reasoner_depth during multi-stage learning (like if pretraining the KGEs).
        self.enabled_reasoner_depth = reasoner_depth
        self.resnet = resnet
        self.embedding_resnet = embedding_resnet
        self.logic = GodelTNorm()
        self.use_ultra = use_ultra
        self.use_ultra_with_kge = use_ultra_with_kge
        self.use_llm = use_llm
        self.global_serialization = global_serialization
        if not self.use_ultra and not self.use_llm:
            self.kge_model = KGEModel(fol, kge,
                                    kge_regularization,
                                    constant_embedding_size,
                                    predicate_embedding_size,
                                    kge_atom_embedding_size,
                                    kge_dropout_rate,
                                    num_adaptive_constants,
                                    dot_product,
                                    device='cpu',
                                    use_ultra=self.use_ultra_with_kge,
                                    global_serialization=self.global_serialization)
            self.output_layer = self.kge_model.output_layer
        elif self.use_ultra:
            self.ULTRA_bridge = ULTRA_bridge()
            self.output_layer = self.ULTRA_bridge.output_layer
        elif self.use_llm:
            self.LLM_bridge = LLM_Bridge()
            self.output_layer = self.LLM_bridge.output_layer
            
        self.model_name = model_name

        # REASONING LAYER
        self.reasoning = None
        if reasoner_depth > 0 and len(rules) > 0:
            if self.embedding_resnet:
                self.embedding_resnet_weight = Sequential([
                    #Dense(16, activation='relu'),
                    #Dropout(0.2),
                    Dense(1,
                          #kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                          bias_regularizer=regularizers.L2(1e-4),
                          activity_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-5),
                          activation='sigmoid')])

            self.reasoning = []
            for i in range(reasoner_depth):
              if i > 0 and reasoner_single_model:
                  self.reasoning.append(self.reasoning[0])
                  continue

              if model_name == 'dcr':
                  self.reasoning.append(DCRReasoningLayer(
                      templates=rules,
                      formula_hidden_size=reasoner_formula_hidden_embedding_size,
                      aggregation_type=aggregation_type,
                      temperature=temperature,
                      signed=signed,
                      filter_num_heads=filter_num_heads,
                      regularization=reasoner_regularization,
                      dropout_rate=reasoner_dropout_rate))

              elif model_name == 'cdcr':
                  self.reasoning.append(ClusteredDCRReasoningLayer(
                      num_formulas=cdcr_num_formulas,
                      use_positional_embeddings=cdcr_use_positional_embeddings,
                      templates=rules,
                      formula_hidden_size=reasoner_formula_hidden_embedding_size,
                      aggregation_type=aggregation_type,
                      temperature=temperature,
                      signed=signed,
                      filter_num_heads=filter_num_heads,
                      regularization=reasoner_regularization,
                      dropout_rate=reasoner_dropout_rate))

              elif model_name == 'r2n':
                  def SumAndSigmoidOutput(x):
                      return tf.nn.sigmoid(tf.reduce_sum(x, axis=-1))
                  output_layer = (Lambda(SumAndSigmoidOutput, name='output_layer')
                                  if kge == 'rotate' else self.output_layer)
                  self.reasoning.append(R2NReasoningLayer(
                      rules=rules,
                      formula_hidden_size=reasoner_formula_hidden_embedding_size,
                      atom_embedding_size=reasoner_atom_embedding_size,
                      prediction_type=r2n_prediction_type,
                      aggregation_type=aggregation_type,
                      output_layer=output_layer,
                      regularization=reasoner_regularization,
                      dropout_rate=reasoner_dropout_rate))

              elif model_name == 'sbr':
                  self.reasoning.append(SBRReasoningLayer(
                      rules=rules,
                      aggregation_type=aggregation_type))

              elif model_name == 'gsbr':
                  self.reasoning.append(GatedSBRReasoningLayer(
                      rules=rules,
                      aggregation_type=aggregation_type,
                      regularization=reasoner_regularization))

              elif model_name == 'rnm':
                  self.reasoning.append(RNMReasoningLayer(
                      rules=rules,
                      aggregation_type=aggregation_type,
                      regularization=reasoner_regularization))

              elif model_name == 'dsl':
                  self.reasoning.append(DeepStocklogLayer(
                      rules=rules,
                      aggregation_type=aggregation_type,
                      regularization=reasoner_regularization))

              else:
                  assert False, 'Unknown model name %s' % model_name

        self._explain_mode = False

    def explain_mode(self, mode=True):
        self._explain_mode = mode

    def test_mode(self, dataset_type, mode=False):
        self.testing = mode
        self.dataset_type = dataset_type

    def call(self, inputs, training=False, *args, **kwargs):
        '''
        X_domains type is Dict[str, tensor[constant_indices_in_domain]]
        A_predicate: Dict[predicate_name, List[Tuple[Index1, ..., IndexN]]]
                     e.g. mapping predicate_name -> tensor [num_groundings, arity]
                     with constant indices for each grounding.
        '''

        if self._explain_mode:
            # No explanations are posible when reasoning is disabled.
            assert self.reasoning is not None
            # Check that we are using an explainable model.
            assert self.model_name == 'dcr' or self.model_name == 'cdcr'

        (X_domains, A_predicates, A_rules, Q, embeddings) = inputs

        if self.use_ultra:
            print('USING ULTRA')
            (concept_output, concept_embeddings) = embeddings
            concept_output,concept_embeddings = self.ULTRA_bridge(concept_embeddings)
            # print('concept_output',concept_output.shape)
            # print('concept_embeddings',concept_embeddings.shape)
        elif self.use_llm:
            concept_output,concept_embeddings = self.LLM_bridge(embeddings)
        else:
            concept_output, concept_embeddings = self.kge_model((X_domains, A_predicates),embeddings=embeddings)
 
        task_output = tf.identity(concept_output) # (len(sum A_pred),1)


        # create a tensor to test the model:  
        # shape = sum([len(v) for v in A_predicates.values()])
        # task_output = tf.ones([shape, 1], dtype=tf.float32)*0.5
        # concept_embeddings = tf.ones([shape, 100], dtype=tf.float32)*0.5
        # print('types','concept_embeddings',type(concept_embeddings),'concept_output',type(concept_output),'task_output',type(task_output))
        # print('Concept embeddings',concept_embeddings.shape) 
        # print('Concept output',concept_output.shape, concept_output[:10])
        # print('Task output',task_output.shape, task_output[:10])


        explanations = None
        if self.reasoning is not None:
            atom_embeddings = tf.identity(concept_embeddings)
            for i in range(self.enabled_reasoner_depth):
                if self._explain_mode and i == self.enabled_reasoner_depth - 1:
                    explanations = self.reasoning[i].explain(
                        [task_output, atom_embeddings, A_rules])
                task_output, atom_embeddings = self.reasoning[i]([
                    task_output, atom_embeddings, A_rules])
            if self.embedding_resnet: # CAREFUL IN CASE OF ULTRA, THE OUTPUT LAYER SHOULD BE GIVEN BY ULTRA
                # In this case we need to recompute the output from the updated embeddings.
                w = tf.clip_by_value(self.embedding_resnet_weight(tf.concat([concept_embeddings, atom_embeddings], axis=-1)), 1e-9, 1.0 - 1e-7)
                tf.print('embedding_resnet_weight', tf.reduce_mean(w))
                atom_embeddings = (1.0 - w) * tf.stop_gradient(concept_embeddings) + w * atom_embeddings
                task_output = tf.expand_dims(self.output_layer(atom_embeddings), axis=-1)

        task_output = tf.gather(params=tf.squeeze(task_output, -1), indices=Q)
        concept_output = tf.gather(params=tf.squeeze(concept_output, -1),indices=Q)
        # print('latest task_output',task_output.shape, task_output[:10])
        # print('latest concept_output',concept_output.shape, concept_output[:10])
        if self.resnet and self.reasoning is not None:
            task_output = self.logic.disj_pair(task_output, concept_output)

        if self._explain_mode:
            return concept_output, task_output, explanations
        else:
            return {'concept':concept_output, 'task':task_output}


































# DATASET.PY

import ultra_utils
from ultra_utils import Ultra as Ultra_modified
from ULTRA.ultra.models import Ultra
from ULTRA.ultra import tasks
import itertools
from ULTRA.ultra.tasks import build_relation_graph
from ns_lib.nn.constant_embedding import LMEmbeddings

def _from_strings_to_tensors(fol, serializer,
                             queries, labels, engine, ragged,
                             constants_features=None, deterministic=True, global_serialization=False):

    # Symbolic step
    facts_tuple = tuple(fol.facts)
    queries_tuple = tuple(ns.utils.to_flat(queries))
    if engine is not None:
        ground_formulas: Dict[str, RuleGroundings] = engine.ground(
            facts_tuple, queries_tuple, deterministic=deterministic)
        rules = engine.rules
    else:
        ground_formulas = {}
        rules = []

    
    A_predicates_global = queries_global = A_predicates_global_textualized = None
    if global_serialization:
        (domain_to_global, predicate_tuples, groundings, queries, (queries_global,A_predicates_global,A_predicates_global_textualized)) = (
            serializer.serialize_global_A_predicates(fol,queries=queries,rule_groundings=ground_formulas))
    else:
        (domain_to_global, predicate_tuples, groundings, queries) = (serializer.serialize(queries=queries,
                                    rule_groundings=ground_formulas))   

    # Convert constants(domain) indices from list to tf tensor
    input_domains_tf: Dict[DomainName, ConstantFeatures] = {}
    for d in fol.domains:
        if constants_features is not None and d.name in constants_features:
            # If available, global features for constants are gathered based on their global indices within their domains.
            global_features = constants_features[d.name]
            global_indices = domain_to_global[d.name]
            input_features = tf.gather(global_features, global_indices, axis=0)
            input_domains_tf[d.name] = input_features
        else:
            input_domains_tf[d.name] = tf.constant(domain_to_global[d.name],
                                                   dtype=tf.int32)
    # Creating the input dictionaries (atoms as tuples of domains,
    # atoms as dense ids, formulas as tuples of atoms)
    # Convert ctes indices with respect to predicates from list to tf tensor. (num_predicates, number_of_groundings, arity_of_predicate)
    # Dict[predicate_name, List[Tuple[constants_ids]]]
    input_atoms_tuples_tf: Dict[PredicateName, ConstantTuples] = {
        name:tf.constant(tuples, dtype=tf.int32) if len(tuples) > 0 else
             tf.zeros(shape=(0, fol.name2predicate[name].arity), dtype=tf.int32)
        for name,tuples in predicate_tuples.items()}

    # Same here, but for the groundings of the rules. (num_rules, 2, num_atoms (in body/head), arity_of_predicate)
    # Dict[formula_id, List[Tuple[atom_ids]]]
    input_formulas_tf: Dict[FormulaSignature, (AtomTuples, AtomTuples)] = {}
    for rule in rules:
        ai = len(rule.body)
        ao = len(rule.head)
        if rule.name in groundings and len(groundings[rule.name]) > 0:
            # adding batch dimension
            input_formulas_tf[rule.name] = (
                tf.constant(groundings[rule.name][0], dtype=tf.int32),
                tf.constant(groundings[rule.name][1], dtype=tf.int32))
        else:
            # empty tensor
            input_formulas_tf[rule.name] = (
                tf.zeros(shape=[0, ai], dtype=tf.int32),
                tf.zeros(shape=[0, ao], dtype=tf.int32))
            
    # TODO check how to understand if it is a good tensor or need to be ragged.
    if ragged:
        queries = tf.ragged.constant(queries, dtype=tf.int32)
        labels =  tf.ragged.constant(labels, dtype=tf.float32)
    else:
        queries = tf.constant(queries, dtype=tf.int32)
        labels = tf.constant(labels, dtype=tf.float32)

    # X_domains_data, A_predicates_data, A_rules_data, queries
    return (input_domains_tf, input_atoms_tuples_tf, 
            input_formulas_tf, queries, 
            (queries_global,A_predicates_global,A_predicates_global_textualized)), labels





class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 dataset: Dataset,
                 fol: FOL,
                 serializer,
                 engine=None,
                 deterministic=True,
                 batch_size=None,
                 ragged: bool=False,
                 name= "None",
                 use_ultra = False,
                 use_ultra_with_kge = False,
                 use_llm = False,
                 global_serialization = False,
                 dataset_ultra = None):
        
        self.global_serialization = global_serialization
        self.dataset = dataset
        self.deterministic = deterministic
        self.fol = fol
        self.engine = engine
        self.serializer = serializer
        self.ragged = ragged
        self._batch_size = (batch_size
                            if batch_size is not None and batch_size > 0
                            else -1)
        self.name = name
        self.use_ultra = use_ultra
        self.use_ultra_with_kge = use_ultra_with_kge
        self.use_llm = use_llm

        if self.use_ultra or self.use_ultra_with_kge:
            assert dataset_ultra is not None, 'You need to provide the aux dataset_ultra'
            self.aux_dataset = dataset_ultra
            self.Ultra = ultra_utils.load_ultra_model(Ultra, original=True)
            self.Ultra_modified = ultra_utils.load_ultra_model(Ultra_modified, original=False)

        if self.use_llm:
            self.llm_embedder = LMEmbeddings(self.fol, "")

        self._num_batches = 1
        if self._batch_size > 0:
            if len(self.dataset) % self._batch_size == 0:
                self._num_batches = len(self.dataset) // self._batch_size
            else:
                self._num_batches = len(self.dataset) // self._batch_size + 1

        if self._num_batches == 1:
            print('Building Full Batch Dataset', self.name)
            self._full_batch = self._get_batch(0, len(self.dataset))


    def __getitem__(self, item):
        if self._num_batches == 1:
            return self._full_batch
        else:
            return self._get_batch(item, self._batch_size)


    def __len__(self):
        return self._num_batches


    def _get_batch(self, i, b):
        print('getting batch')
        queries, labels = self.dataset[b*i:b*(i+1)]
        constants_features = self.dataset.constants_features

        ((X_domains_data, A_predicates_data, A_rules_data, Q, (Q_global,A_predicates_triplets,A_predicates_textualized)), y) = _from_strings_to_tensors(
            fol=self.fol,
            serializer=self.serializer,
            queries=queries,
            labels=labels,
            engine=self.engine,
            ragged=self.ragged,
            constants_features=constants_features,
            deterministic=self.deterministic,
            global_serialization=self.global_serialization) 

        embeddings = None

        if self.use_ultra_with_kge:  
            constant_embeddings, predicate_embeddings = self.Ultra(self.aux_dataset, Q_global,atom_repr=False) # embedds of the ctes,preds
            for key in constant_embeddings: # Convert embeddings to tf
                constant_embeddings[key] = tf.constant(constant_embeddings[key])
            embeddings = (constant_embeddings, tf.constant(predicate_embeddings, dtype=tf.float32))

        elif self.use_ultra: 
            self.Ultra.eval()
            self.Ultra_modified.eval()
            ultra_utils.mimic_ultra(self.aux_dataset, Ultra)
            # ultra_utils.mimic_test_function_ultra(self.aux_dataset,Q_global,self.Ultra)
            # scores,_ = ultra_utils.mimic_test_function_ultra_with_our_corruptions(self.aux_dataset, Q_global, self.Ultra)
            # scores, y = ultra_utils.get_ultra_outputs(self.aux_dataset,Q_global, self.Ultra_modified)
            # scores, atom_embeds = ultra_utils.get_ultra_outputs_nonfiltered_negatives(self.aux_dataset,Q_global, self.Ultra) # For train, instead of 4 negatives, I have all the negatives
            # scores, atom_embeds = self.get_ultra_embeddings(A_predicates_triplets,Q_global) # this is the official one
            # embeddings = (scores, atom_embeds)

        elif self.use_llm:
            constant_embeddings, predicate_embeddings = self.llm_embedder(A_predicates_triplets)
            embeddings = (constant_embeddings, predicate_embeddings)


        return (X_domains_data, A_predicates_data, A_rules_data, Q, embeddings), y



    def get_ultra_embeddings(self,A_pred,queries):
        '''
        Do the preprocessing of the data to give it to ultra. Take a modified ultra to get the embeddings of the atoms of A_pred.
        Return the embeddings of the atoms and the scores of the atoms of A_pred.
        # Option 1: get the embeddings of the atoms in A_predicates_data. But then I would need to calculate the negatives of the atoms in A_predicates_data
        # Option 2: for every atom in Q_global, calculate the embeddings of the atom and the negatives
        '''

        '''Process: use A_pred_global, get negatives for each triplet. Give it as input to ultra, and get the embeddings and the scores of the atoms.'''

        # Convert A_pred to a format that ultra can understand. For every atom in A_pred, create a list of the atom and the negatives
        batch = torch.tensor(A_pred, dtype=torch.int64)
        # Get the negatives of the atoms in A_pred. Take into account to filter atoms from edge_index, edge_type, target_edge_index, target_edge_type. 
        t_batch, h_batch = tasks.all_negative(self.aux_dataset, batch)
        # Get the embeddings of the atoms in A_pred and the negatives
        t_pred_scores, t_pred_embedd = self.Ultra_modified(self.aux_dataset, t_batch)
        h_pred_scores, h_pred_embedd = self.Ultra_modified(self.aux_dataset, h_batch)
        
        # Now, for the embeddings, I can return them in different ways. It is better to obtain the scores from the embeddings in our mode. 

        # # i) Do the average of the embeddings of all the entities. This is the most general way, but it is not the best way to represent the entities
        # t_pred_embedd = torch.mean(t_pred_embedd, dim=1)
        # h_pred_embedd = torch.mean(h_pred_embedd, dim=1)

        # ii) Take the embeddings of the entitiy that is in the positive query.
        # In t corruptions, the original index is the tail of the first element of the list, in h corruptions, the original index is the head of the first element of the list  
        t_index = batch[:,1]
        h_index = batch[:,0]
        # in the embeddings, the representation given by the index, in the first dimension, is the original representation of the entity
        t_embedd = torch.zeros(t_index.shape[0], t_pred_embedd.shape[2])
        h_embedd = torch.zeros(h_index.shape[0], h_pred_embedd.shape[2])
        for i in range(len(t_index)):
            t_embedd[i] = t_pred_embedd[i][t_index[i]]
            h_embedd[i] = h_pred_embedd[i][h_index[i]]

        # convert the scores and embeddings to tf
        t_pred_scores_tf = tf.squeeze(tf.constant(t_pred_scores.detach().numpy(), dtype=tf.float32))
        t_embedd = tf.squeeze(tf.constant(t_embedd.detach().numpy(), dtype=tf.float32))
        h_pred_scores_tf = tf.squeeze(tf.constant(h_pred_scores.detach().numpy(), dtype=tf.float32))
        h_embedd = tf.squeeze(tf.constant(h_embedd.detach().numpy(), dtype=tf.float32))



        # To calculate the metrics, gather the indices of the triplets of A_pred that are in the queries
        queries_positive = [q[0] for q in queries]
        indices = [i for i in range(len(A_pred)) if A_pred[i] in queries_positive]

        t_pred_scores_queries = t_pred_scores.detach().numpy()
        t_pred_scores_queries = t_pred_scores_queries[indices]

        scores = np.squeeze(t_pred_scores.detach().numpy())
        scores = scores[indices]
        labels_new = np.zeros(tuple(scores.shape))
        labels_new[:,0] = 1
        labels_tf = tf.constant(labels_new, dtype=tf.float32)
        # do a copy of the scores, because the function calculated_metrics_batched modifies the scores
        t_pred_scores_copy = tf.squeeze(tf.constant(t_pred_scores_queries))
        ultra_utils.calculated_metrics_batched(t_pred_scores_copy, labels_tf)

        return scores_tf, t_embedd
    














def obtain_queries(dataset,data_handler,serializer,engine,ragged,deterministic,global_serialization):
    queries, labels = dataset[:]
    constants_features = dataset.constants_features
    fol = data_handler.fol

    ((X_domains_data, A_predicates_data, A_rules_data, Q, (Q_global,A_predicates_triplets,A_predicates_textualized)), y) = _from_strings_to_tensors(
        fol=fol,
        serializer=serializer,
        queries=queries,
        labels=labels,
        engine=engine,
        ragged=ragged,
        constants_features=constants_features,
        deterministic=deterministic,
        global_serialization=global_serialization) 
    Q_global_positive = [q[0] for q in Q_global]
    # print('\nqueries positive', len(queries), [query[0] for query in queries][:20])
    # print('Q_global_positive', len(Q_global_positive), Q_global_positive[:20])
    return X_domains_data, A_predicates_data, Q_global_positive

def get_ultra_datasets(dataset_train, dataset_valid, dataset_test,data_handler,serializer,engine,ragged=True,deterministic=True,global_serialization=False):

    # Get the triplets
    X_domain_train, A_pred_train, train_triplets = obtain_queries(dataset_train,data_handler,serializer,engine,ragged,deterministic,global_serialization)
    X_domain_valid, A_pred_valid, valid_triplets = obtain_queries(dataset_valid,data_handler,serializer,engine,ragged,deterministic,global_serialization)
    X_domain_test, A_pred_test, test_triplets = obtain_queries(dataset_test,data_handler,serializer,engine,ragged,deterministic,global_serialization)

    def unique_ordered(triplets):
        return list(dict.fromkeys(tuple(t) for t in triplets))

    train_triplets = unique_ordered(train_triplets)
    valid_triplets = unique_ordered(valid_triplets)
    test_triplets = unique_ordered(test_triplets)

    # get the number of nodes and relations for the train,val,test set. Do it by getting unique the ones in train, val, test
    train_nodes = [X_domain_train[key].numpy().tolist() for key in X_domain_train]
    valid_nodes = [X_domain_valid[key].numpy().tolist() for key in X_domain_valid]
    test_nodes = [X_domain_test[key].numpy().tolist() for key in X_domain_test]

    # Flatten the lists
    train_nodes = set(itertools.chain(*train_nodes))
    valid_nodes = set(itertools.chain(*valid_nodes))
    test_nodes = set(itertools.chain(*test_nodes))
    num_node = len(train_nodes.union(valid_nodes).union(test_nodes) )

    # do the same for the relations
    train_relations = [key for key in A_pred_train]
    valid_relations = [key for key in A_pred_valid]
    test_relations = [key for key in A_pred_test]
    # take the unique number of relations
    unique_relations = list(set(train_relations+valid_relations+test_relations))
    num_relations_no_inv = torch.tensor(len(unique_relations))
    # num_relations_no_inv = len(data_handler.fol.predicates)

    train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
    train_target_etypes = torch.tensor([t[2] for t in train_triplets])
    train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
    train_etypes = torch.cat([train_target_etypes, train_target_etypes+num_relations_no_inv])

    # valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
    valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
    valid_etypes = torch.tensor([t[2] for t in valid_triplets])

    test_edges = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
    test_etypes = torch.tensor([t[2] for t in test_triplets])

    train_data = ultra_utils.Dataset_Ultra(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                        target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_relations_no_inv*2)
    train_data.num_edges = train_data.edge_index.shape[1]
    valid_data = ultra_utils.Dataset_Ultra(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                        target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations_no_inv*2)
    valid_data.num_edges = valid_data.edge_index.shape[1]
    test_data = ultra_utils.Dataset_Ultra(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                        target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations_no_inv*2)
    test_data.num_edges = test_data.edge_index.shape[1]

    # edge_index is the sum of all target_edge_index
    edge_index = torch.cat([train_target_edges, valid_edges, test_edges], dim=1)
    edge_type = torch.cat([train_target_etypes, valid_etypes, test_etypes])
    # num_nodes is given by the train set
    num_edges = None # is not defined in Ultra for the general dataset
    device = 'cpu'
    dataset = ultra_utils.Dataset_Ultra(edge_index, edge_type, num_relations_no_inv*2, num_node, num_edges, device)
    filtered_data = dataset 

    train_data = build_relation_graph(train_data)
    valid_data = build_relation_graph(valid_data)
    test_data = build_relation_graph(test_data)

    return train_data, valid_data, test_data, filtered_data




#SERIALIZER.PY


class LogicSerializerFast(IndexerBase):

    def __init__(self, predicates: List[Predicate], domains: List[Domain],
                 constant2domain_name: Dict[str, str],
                 domain2adaptive_constants: Dict[str, List[str]]=None):
        self.predicates = predicates
        self.domains = domains
        self.constant2domain_name = constant2domain_name

        self.constant_to_global_index = defaultdict(OrderedDict) # X_domains
        self.adaptive_constant2domain = defaultdict(OrderedDict)
        for domain in domains:
            # Add fixed constants. Global index for each constant in each domain (starting from 0 in each domain)
            for i, constant in enumerate(domain.constants):
                self.constant_to_global_index[domain.name][constant] = i
            # Add adaptive constants.
            if domain2adaptive_constants is not None:
                offset = len(domain.constants)
                adaptive_constants = domain2adaptive_constants.get(
                    domain.name, [])
                for i, constant in enumerate(adaptive_constants):
                    self.constant_to_global_index[domain.name][constant] = (
                        i + offset)
                    self.adaptive_constant2domain[constant] = domain.name



        self.predicate_to_domains = {}
        for predicate in self.predicates:
            self.predicate_to_domains[predicate.name] = [
                domain.name for domain in predicate.domains]
        
        # #####################################

        # Create a constant_to_global_unique_index in which each constant, even if in different domains, has a unique index. This is useful for ultra
        self.constant_to_global_unique_index = defaultdict(OrderedDict)
        counter = 0
        for domain in domains:
            for constant in domain.constants:
                self.constant_to_global_unique_index[domain.name][constant] = counter
                counter += 1

        # dic = {'afghanistan': 0, 'asia': 1, 'southern_asia': 2, 'aland_islands': 3, 'europe': 4, 'northern_europe': 5, 'albania': 6, 'southern_europe': 7, 'algeria': 8, 'africa': 9, 'northern_africa': 10, 'american_samoa': 11, 'oceania': 12, 'polynesia': 13, 'andorra': 14, 'angola': 15, 'middle_africa': 16, 'anguilla': 17, 'americas': 18, 'caribbean': 19, 'antigua_and_barbuda': 20, 'argentina': 21, 'south_america': 22, 'armenia': 23, 'western_asia': 24, 'aruba': 25, 'australia_and_new_zealand': 26, 'australia': 27, 'austria': 28, 'western_europe': 29, 'azerbaijan': 30, 'bahamas': 31, 'bahrain': 32, 'bangladesh': 33, 'barbados': 34, 'belarus': 35, 'eastern_europe': 36, 'belgium': 37, 'belize': 38, 'central_america': 39, 'benin': 40, 'western_africa': 41, 'bermuda': 42, 'northern_america': 43, 'bhutan': 44, 'bolivia': 45, 'bosnia_and_herzegovina': 46, 'botswana': 47, 'southern_africa': 48, 'brazil': 49, 'british_indian_ocean_territory': 50, 'eastern_africa': 51, 'british_virgin_islands': 52, 'brunei': 53, 'south_eastern_asia': 54, 'bulgaria': 55, 'burkina_faso': 56, 'burundi': 57, 'cambodia': 58, 'cameroon': 59, 'canada': 60, 'cape_verde': 61, 'cayman_islands': 62, 'central_african_republic': 63, 'central_asia': 64, 'central_europe': 65, 'chad': 66, 'chile': 67, 'china': 68, 'eastern_asia': 69, 'christmas_island': 70, 'cocos_keeling_islands': 71, 'colombia': 72, 'comoros': 73, 'cook_islands': 74, 'costa_rica': 75, 'croatia': 76, 'cuba': 77, 'curacao': 78, 'cyprus': 79, 'czechia': 80, 'denmark': 81, 'djibouti': 82, 'dominica': 83, 'dominican_republic': 84, 'dr_congo': 85, 'ecuador': 86, 'egypt': 87, 'el_salvador': 88, 'equatorial_guinea': 89, 'eritrea': 90, 'estonia': 91, 'ethiopia': 92, 'falkland_islands': 93, 'faroe_islands': 94, 'fiji': 95, 'melanesia': 96, 'finland': 97, 'france': 98, 'french_guiana': 99, 'french_polynesia': 100, 'gabon': 101, 'gambia': 102, 'georgia': 103, 'germany': 104, 'ghana': 105, 'gibraltar': 106, 'greece': 107, 'greenland': 108, 'grenada': 109, 'guadeloupe': 110, 'guam': 111, 'micronesia': 112, 'guatemala': 113, 'guernsey': 114, 'guinea': 115, 'guinea_bissau': 116, 'guyana': 117, 'haiti': 118, 'honduras': 119, 'hong_kong': 120, 'hungary': 121, 'iceland': 122, 'india': 123, 'indonesia': 124, 'iran': 125, 'iraq': 126, 'ireland': 127, 'isle_of_man': 128, 'israel': 129, 'italy': 130, 'ivory_coast': 131, 'jamaica': 132, 'japan': 133, 'jersey': 134, 'jordan': 135, 'kazakhstan': 136, 'kenya': 137, 'kiribati': 138, 'kosovo': 139, 'kuwait': 140, 'kyrgyzstan': 141, 'laos': 142, 'latvia': 143, 'lebanon': 144, 'lesotho': 145, 'liberia': 146, 'libya': 147, 'liechtenstein': 148, 'lithuania': 149, 'luxembourg': 150, 'macau': 151, 'macedonia': 152, 'madagascar': 153, 'malawi': 154, 'malaysia': 155, 'maldives': 156, 'mali': 157, 'malta': 158, 'marshall_islands': 159, 'martinique': 160, 'mauritania': 161, 'mauritius': 162, 'mayotte': 163, 'mexico': 164, 'moldova': 165, 'monaco': 166, 'mongolia': 167, 'montenegro': 168, 'montserrat': 169, 'morocco': 170, 'mozambique': 171, 'myanmar': 172, 'namibia': 173, 'nauru': 174, 'nepal': 175, 'netherlands': 176, 'new_caledonia': 177, 'new_zealand': 178, 'nicaragua': 179, 'niger': 180, 'nigeria': 181, 'niue': 182, 'norfolk_island': 183, 'northern_mariana_islands': 184, 'north_korea': 185, 'norway': 186, 'oman': 187, 'pakistan': 188, 'palau': 189, 'palestine': 190, 'panama': 191, 'papua_new_guinea': 192, 'paraguay': 193, 'peru': 194, 'philippines': 195, 'pitcairn_islands': 196, 'poland': 197, 'portugal': 198, 'puerto_rico': 199, 'qatar': 200, 'republic_of_the_congo': 201, 'reunion': 202, 'romania': 203, 'russia': 204, 'rwanda': 205, 'saint_barthelemy': 206, 'saint_kitts_and_nevis': 207, 'saint_lucia': 208, 'saint_martin': 209, 'saint_pierre_and_miquelon': 210, 'saint_vincent_and_the_grenadines': 211, 'samoa': 212, 'san_marino': 213, 'sao_tome_and_principe': 214, 'saudi_arabia': 215, 'senegal': 216, 'serbia': 217, 'seychelles': 218, 'sierra_leone': 219, 'singapore': 220, 'sint_maarten': 221, 'slovakia': 222, 'slovenia': 223, 'solomon_islands': 224, 'somalia': 225, 'south_africa': 226, 'south_georgia': 227, 'south_korea': 228, 'south_sudan': 229, 'spain': 230, 'sri_lanka': 231, 'sudan': 232, 'suriname': 233, 'svalbard_and_jan_mayen': 234, 'swaziland': 235, 'sweden': 236, 'switzerland': 237, 'syria': 238, 'taiwan': 239, 'tajikistan': 240, 'tanzania': 241, 'thailand': 242, 'timor_leste': 243, 'togo': 244, 'tokelau': 245, 'tonga': 246, 'trinidad_and_tobago': 247, 'tunisia': 248, 'turkey': 249, 'turkmenistan': 250, 'turks_and_caicos_islands': 251, 'tuvalu': 252, 'uganda': 253, 'ukraine': 254, 'united_arab_emirates': 255, 'united_kingdom': 256, 'united_states_minor_outlying_islands': 257, 'united_states': 258, 'united_states_virgin_islands': 259, 'uruguay': 260, 'uzbekistan': 261, 'vanuatu': 262, 'vatican_city': 263, 'venezuela': 264, 'vietnam': 265, 'wallis_and_futuna': 266, 'western_sahara': 267, 'yemen': 268, 'zambia': 269, 'zimbabwe': 270}
        # self.constant_to_global_unique_index = defaultdict(dict) 
        # for domain in domains:
        #     for constant in domain.constants:
        #         self.constant_to_global_unique_index[domain.name][constant] = dic[constant]

        self.predicate_to_global_index = defaultdict(dict)  # A_predicates with global index
        for i,predicate in enumerate(self.predicates):
            self.predicate_to_global_index[predicate.name] = i


    def serialize(self, queries:List[List[Tuple]],
                  rule_groundings:Dict[str, RuleGroundings]):
        '''
        Takes all the atoms from groundings and queries and returns 
          
          - atoms (predicate_to_constant_tuples): batch atoms represented by constant (local) indices and pred as str, e.g. {'LocIn': [[1,34],[2,3],...],...}
          
          - grundings (index_groundings): all groundings represented by atom (local) indices and pred implicit, e.g. {'rule1': ([[1],[3],...], [[1,34],[2,3],...]), 'rule2'...}
          
          - queries (index_queries): represented by atom (local) indices, e.g. [[1,3,4],[2,3,8],...] (first is the positive, rest are negative atoms)
          
          - constants map (domain_to_global), where the i-th element of the list is the local index and its value is the global index, e.g. {'countries': [0,1,2,3,...], 'regions': [0,3,1,2,...]}
          
                - atoms map (atom_to_index), where the key is the atom and the value is the local index, e.g. {('LocIn', 'morocco', 'spain'): 0, ('LocIn', 'morocco', 'france'): 1,...}
        '''
        domain_to_global = defaultdict(list)  # X_domains, it's a map, where the i-th element of the list is the local index and its value is the global index
        domain_to_local_constant_index = defaultdict(dict)  # helper, as X_domains but for local indices, created new every batch
        predicate_to_constant_tuples = defaultdict(list)  # A_predicates

        # Set of all atoms in the groundings to index, e.g. [('locatedInCR', 'bhutan', 'africa'),...]
        all_atoms = ns.utils.to_flat(queries)
        for rg in rule_groundings.values():
            for g in rg.groundings:
                all_atoms += g[0] # head
                all_atoms += g[1] # body
        all_atoms = sorted(list(set(all_atoms))) # Remove duplicates and order alphabetically

        # Bucket them per predicate, e.g.  {'LocIn': [('LocIn', 'morocco', 'spain'), ('LocIn', 'morocco', 'france'),...],...}
        all_atoms_per_predicate = {predicate.name: []
                                   for predicate in self.predicates}
        for atom in all_atoms:
            all_atoms_per_predicate[atom[0]].append(atom) 

        atom_to_index = {}
        count = 0
        for predicate in self.predicates: 
            constant_tuples = [] # For each predicate, a list of constant indices (local) is generated for each atom, e.g. LocIn:[[1,34],[2,3],...]
            domains = self.predicate_to_domains[predicate.name]
            for atom in all_atoms_per_predicate[predicate.name]: # For every atom in the bucketed atoms
                atom_to_index[atom] = count  # Assigns an index (local) to the atom, e.g. {LocIn(morocco,spain):0,LocIn(morocco,france):1,...}
                count += 1
                indices_cs = [] # This is for A_predicates, to get the constant indices in a list format, e.g.  (Morocco,Spain)->[1,34] (from the LocIn(morocco,spain) example)

                for c in atom[1:]:
                    domain = (self.constant2domain_name[c]    # get the domain of the constant
                              if c in self.constant2domain_name else
                              self.adaptive_constant2domain[c])
                    # Check that the domain of the constant i is the corresponding domain of the predicate, otherwise print the constant and the domain
                    assert domain == domains[atom[1:].index(c)], 'Domain of constant does not match the domain of the predicate, constant: %s, domain: %s, domain atom: %s, atom: %s' % (c,domain,domains[atom[1:].index(c)], atom)

                    # get the local indices of the ctes built so far for this batch
                    constant_index = domain_to_local_constant_index[domain] # It is a domain_to_global_index but for the local indices, created for every batch
                    if c not in constant_index:  
                        constant_index[c] = len(constant_index) # Add it to domain_to_local_constant_index if not already there
                        domain_to_global[domain].append(
                            self.constant_to_global_index[domain][c]) # get the global idx and append it to X_domains
                    indices_cs.append(constant_index[c]) # Append the local index of the constant to A_predicates
                constant_tuples.append(indices_cs) 
            predicate_to_constant_tuples[predicate.name] = constant_tuples

        index_groundings = {}
        for name,rule in rule_groundings.items():
            if len(rule.groundings) > 0:
                G_body = []
                G_head = []
                for g in rule.groundings:
                    G_body.append([atom_to_index[atom] for atom in g[1]])
                    G_head.append([atom_to_index[atom] for atom in g[0]])
                index_groundings[name] = G_body, G_head
        index_queries = [[atom_to_index[q] for q in Q] for Q in queries]
        return (
            # domain->[global_idx] where i-th element is the global index of the
            # i-th local constant. e.g. this maps a local index into a global one.
            domain_to_global,
            # These are the atoms expressed in form of local indices:
            # predicate->[(constant_local_idx)]
            predicate_to_constant_tuples,
            # rule -> [atom_local_idx_for_body, atom_local_idx_for_head]
            index_groundings,
            # [atom_local_indices_for_query]
            index_queries)
    



    def serialize_global_A_predicates(self, fol, queries:List[List[Tuple]],
                  rule_groundings:Dict[str, RuleGroundings]):
        domain_to_global = defaultdict(list)  # X_domains
        # domain_to_local_constant_index = defaultdict(dict)  # helper
        predicate_to_constant_tuples = defaultdict(list)  # A_predicates


        # Set of all atoms in the groundings to index
        all_atoms = ns.utils.to_flat(queries)
        for rg in rule_groundings.values():
            for g in rg.groundings:
                all_atoms += g[0] # head
                all_atoms += g[1] # body
        all_atoms = sorted(list(set(all_atoms)))

        # Bucket them per predicate
        all_atoms_per_predicate = {predicate.name: []
                                   for predicate in self.predicates}
        for atom in all_atoms:
            all_atoms_per_predicate[atom[0]].append(atom)

        # print('all_atoms_per_predicate')
        # for predicate in self.predicates:
        #     print('\nPredicate',predicate,all_atoms_per_predicate[predicate.name])

        # Create the index following the bucketed order:
        # A loop iterates over each predicate.
        # For each atom in the predicate, an index is assigned in atom_to_index.
        # Each constant in the atom is assigned an index relative to its domain.
        # A list of constant indices is generated for each atom.
        atom_to_index = {}
        count = 0
        for predicate in self.predicates:
            # print('predicate:',predicate.name)
            constant_tuples = [] # For each predicate, a list of constant indices is generated for each atom, e.g. LocIn:[[1,34],[2,3],...]
            for atom in all_atoms_per_predicate[predicate.name]: #For every atom in the queries
                atom_to_index[atom] = count  # HERE IT ASSIGNS AN INDEX TO EVERY ATOM (FOR EVERY PREDICATE), e.g. {LocIn(morocco,spain):0,LocIn(morocco,france):1,...}
                count += 1
                indices_cs = [] # This is for A_predicates, to get the constant indices in a list format, e.g.  (Morocco,Spain)->[1,34] (from the LocIn(morocco,spain) example)
                for c in atom[1:]: # for every constant in the atom
                    # check if that constant has a domain (in this case should be ctes)
                    domain = (self.constant2domain_name[c]
                              if c in self.constant2domain_name else
                              self.adaptive_constant2domain[c])
                    # print('     domain:',domain, c)
                    # constant_index = domain_to_local_constant_index[domain] # It is a domain_to_global_index but for the local indices
                    # if c not in constant_index:  # If the constant is not in the  local indices for constants mentioned before
                    assert c in self.constant_to_global_unique_index[domain], 'Constant not indexed'
                    if self.constant_to_global_unique_index[domain][c] not in domain_to_global[domain]:
                        # constant_index[c] = len(constant_index) # Add it to domain_to_local_constant_index
                        domain_to_global[domain].append(
                            self.constant_to_global_unique_index[domain][c]) # Append it to domain_to_global, which is what I return as X_domains, which is the different constants for each domain
                    # indices_cs.append(constant_index[c]) # Append the local index of the constant to the list of constant indices. INSTEAD, I SHOULD APPEND THE GLOBAL INDEX
                    indices_cs.append(self.constant_to_global_unique_index[domain][c]) # Append the global index of the constant to the list of constant indices
                # print('     indices:',indices_cs, atom[1:])
                constant_tuples.append(indices_cs) # Append the list of constant local indices to the list of constant indices for the predicate
            predicate_to_constant_tuples[predicate.name] = constant_tuples

        index_queries = [[atom_to_index[q] for q in Q] for Q in queries]
        index_groundings = {}
        for name,rule in rule_groundings.items():
            if len(rule.groundings) > 0:
                G_body = []
                G_head = []
                for g in rule.groundings:
                    G_body.append([atom_to_index[atom] for atom in g[1]])
                    G_head.append([atom_to_index[atom] for atom in g[0]])
                index_groundings[name] = G_body, G_head

        # Create indices for queries triples, in which, for each query, the index of the h,t,r is stored.
        # I need to do it here because otherwise ultra only gets the local atom index, and I need a global triplet index (which I cannot recover from the local atom index)
        # but also, at the begginging, when I initialize the dataset, I need to compute the triplet indeces for all the queries
        # I need to be careful with the domains. In regions, 1 is Europe, but in countries 1 is Spain. For Ultra, they will be the same
        queries_global = []
        for query in queries:
            triplet_index_query = []
            for atom in query:
                index_query = []
                for c in atom[1:]:
                    domain = (self.constant2domain_name[c]
                                if c in self.constant2domain_name else
                                self.adaptive_constant2domain[c])
                    # index_query.append(self.constant_to_global_index[domain][c])
                    index_query.append(self.constant_to_global_unique_index[domain][c])
                # Now get the index of the predicate
                predicate_idx = self.predicate_to_global_index[atom[0]]
                index_query.append(predicate_idx)
                # index_query.append(int(predicate_idx))
                triplet_index_query.append(index_query)
            queries_global.append(triplet_index_query)

        # Create indices for A_predicates triples, in which, for each atom, the index of the h,t,r is stored
        A_predicates_global = []
        A_predicates_global_textualized = []
        for predicate in predicate_to_constant_tuples.keys():
            # Now get the index of the predicate
            predicate_idx = self.predicate_to_global_index[predicate]
            domains = self.predicate_to_domains[predicate]
            domain_head = domains[0]
            domain_tail = domains[1]
            true_flag = True
            for atom in predicate_to_constant_tuples[predicate]:
                A_predicates_global.append(atom + [predicate_idx]) 
                head_position = list(self.constant_to_global_unique_index[domain_head].values()).index(atom[0])
                tail_position = list(self.constant_to_global_unique_index[domain_tail].values()).index(atom[1])
                head_text = list(self.constant_to_global_unique_index[domain_head].keys())[head_position]
                tail_text = list(self.constant_to_global_unique_index[domain_tail].keys())[tail_position]
                if true_flag:
                    A_predicates_global_textualized.append(f"{head_text} is a {domain_head} and it is {predicate} of {tail_text}, which is a {domain_tail}")
                    true_flag = False
                    continue
                A_predicates_global_textualized.append(f"{head_text} is a {domain_head} and it is not {predicate} of {tail_text}, which is a {domain_tail}")

        # print('constant_to_global_unique_index')
        # for domain in self.constant_to_global_unique_index.keys():
        #     print(domain,self.constant_to_global_unique_index[domain])   

        # print('\nSERIALIZER X_DOMAIN KEYS: ,',domain_to_global.keys())
        # for domain in domain_to_global.keys(): # order the constants in the domain
        #     domain_to_global[domain] = sorted(domain_to_global[domain])
        #     tf.print('SERIALIZER X_domain:',domain,domain_to_global[domain])
        # print('SERIALIZER Predicates:',self.predicate_to_global_index.keys())
        # for p,constant_idx in predicate_to_constant_tuples.items():
        #     predicate = fol.name2predicate[p]
            # tf.print('SERIALIZER A_predicate',p,'domains:',[domain.name for domain in predicate.domains],'cte_index',constant_idx)
            # for i,domain in enumerate(predicate.domains):
            #     if len(constant_idx) != 0:
            #         tf.print('SERIALIZER A_predicate:',p,'domain:',domain.name,np.array(constant_idx)[:,i])
            #     else:
            #         tf.print('SERIALIZER A_predicate:',p,'domain:',domain.name, 'empty')

        return (domain_to_global,
                predicate_to_constant_tuples,
                index_groundings,
                index_queries,(queries_global,A_predicates_global,A_predicates_global_textualized))