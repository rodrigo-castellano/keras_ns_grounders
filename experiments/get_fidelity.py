import os

path = 'experiments/ranks/'
path_fidelity = 'experiments/ranks/fidelity/'
files = os.listdir(path)

files_with_reasoner = [f for f in files if 'no_reasoner' not in f and 'fidelity' not in f]

# for each file, check if its 'no_reasoner' version is in the list
for f in files_with_reasoner:
    # model name is in the 3rd position (each is separated by '-')
    model_name = f.split('-')[2] 
    grounder = f.split('-')[1]
    no_reasoner = f.replace(model_name, 'no_reasoner').replace(grounder, 'backward_1_1')
    if no_reasoner in files:
        # read the files structured as query_1:query_2. Put in a dict
        with open(path + f, 'r') as file:
            model_dict = file.readlines()
            model_dict = {line.split(':')[0]: line.split(':')[1].strip() for line in model_dict}
        with open(path + no_reasoner, 'r') as file:
            kge_dict = file.readlines()
            kge_dict = {line.split(':')[0]: line.split(':')[1].strip() for line in kge_dict}
        # calculate the fidelity: for each query in model_dict, assert it is in kge_dict and check if the values are the same
        # if they are, increment the fidelity counter
        fidelity = 0
        for query, rank in model_dict.items():
            assert query in kge_dict, 'Query not found in counterpart file'
            if rank == kge_dict[query]:
                fidelity += 1
        fidelity = fidelity / len(model_dict)
        # print('Fidelity for', f, 'is', fidelity)
        # write the fidelity to a file
        os.makedirs(path_fidelity, exist_ok=True)
        with open(path_fidelity + f.replace('.txt','') + '_fidelity_' + str(round(fidelity,3))+'.txt', 'w') as file:
            file.write(f + ':' + str(fidelity) + '\n')
    else:
        print('No counterpart file: ', no_reasoner)