import os.path
import sys
import ast
import pandas as pd
from collections.abc import Iterable
import tensorflow as tf
import argparse
import numpy as np
import sys
from typing import Dict
import datetime
# from tensorflow_ranking.python.utils import sort_by_scores, ragged_to_dense
from ns_lib.logic.commons import Atom, Domain, Rule, RuleLoader


def ragged_to_dense(labels, predictions, weights):
  """Converts given inputs from ragged tensors to dense tensors.

  Args:
    labels: A `tf.RaggedTensor` of the same shape as `predictions` representing
      relevance.
    predictions: A `tf.RaggedTensor` with shape [batch_size, (list_size)]. Each
      value is the ranking score of the corresponding example.
    weights: An optional `tf.RaggedTensor` of the same shape of predictions or a
      `tf.Tensor` of shape [batch_size, 1]. The former case is per-example and
      the latter case is per-list.

  Returns:
    A tuple (labels, predictions, weights, mask) of dense `tf.Tensor`s.
  """
  _PADDING_LABEL = -1.
  _PADDING_PREDICTION = -1e6
  _PADDING_WEIGHT = 0.
  # TODO: Add checks to validate (ragged) shapes of input tensors.
  mask = tf.cast(tf.ones_like(labels).to_tensor(0.), dtype=tf.bool)
  labels = labels.to_tensor(_PADDING_LABEL)
  if predictions is not None:
    predictions = predictions.to_tensor(_PADDING_PREDICTION)
  if isinstance(weights, tf.RaggedTensor):
    weights = weights.to_tensor(_PADDING_WEIGHT)
  return labels, predictions, weights, mask


def _get_shuffle_indices(shape, mask=None, shuffle_ties=True, seed=None):
  """Gets indices which would shuffle a tensor.

  Args:
    shape: The shape of the indices to generate.
    mask: An optional mask that indicates which entries to place first. Its
      shape should be equal to given shape.
    shuffle_ties: Whether to randomly shuffle ties.
    seed: The ops-level random seed.

  Returns:
    An int32 `Tensor` with given `shape`. Its entries are indices that would
    (randomly) shuffle the values of a `Tensor` of given `shape` along the last
    axis while placing masked items first.
  """
  # Generate random values when shuffling ties or all zeros when not.
  if shuffle_ties:
    shuffle_values = tf.random.uniform(shape, seed=seed)
  else:
    shuffle_values = tf.zeros(shape, dtype=tf.float32)

  # Since shuffle_values is always in [0, 1), we can safely increase entries
  # where mask=False with 2.0 to make sure those are placed last during the
  # argsort op.
  if mask is not None:
    shuffle_values = tf.where(mask, shuffle_values, shuffle_values + 2.0)

  # Generate indices by sorting the shuffle values.
  return tf.argsort(shuffle_values, stable=True)

def sort_by_scores(scores,
                   features_list,
                   topn=None,
                   shuffle_ties=True,
                   seed=None,
                   mask=None):
  """Sorts list of features according to per-example scores.

  Args:
    scores: A `Tensor` of shape [batch_size, list_size] representing the
      per-example scores.
    features_list: A list of `Tensor`s to be sorted. The shape of the `Tensor`
      can be [batch_size, list_size] or [batch_size, list_size, feature_dims].
      The latter is applicable for example features.
    topn: An integer as the cutoff of examples in the sorted list.
    shuffle_ties: A boolean. If True, randomly shuffle before the sorting.
    seed: The ops-level random seed used when `shuffle_ties` is True.
    mask: An optional `Tensor` of shape [batch_size, list_size] representing
      which entries are valid for sorting. Invalid entries will be pushed to the
      end.

  Returns:
    A list of `Tensor`s as the list of sorted features by `scores`.
  """
  with tf.compat.v1.name_scope(name='sort_by_scores'):
    scores = tf.cast(scores, tf.float32)
    scores.get_shape().assert_has_rank(2)
    list_size = tf.shape(input=scores)[1]
    if topn is None:
      topn = list_size
    topn = tf.minimum(topn, list_size)

    # Set invalid entries (those whose mask value is False) to the minimal value
    # of scores so they will be placed last during sort ops.
    if mask is not None:
      scores = tf.where(mask, scores, tf.reduce_min(scores))

    # Shuffle scores to break ties and/or push invalid entries (according to
    # mask) to the end.
    shuffle_ind = None
    if shuffle_ties or mask is not None:
      shuffle_ind = _get_shuffle_indices(
          tf.shape(input=scores), mask, shuffle_ties=shuffle_ties, seed=seed)
      scores = tf.gather(scores, shuffle_ind, batch_dims=1, axis=1)

    # Perform sort and return sorted feature_list entries.
    _, indices = tf.math.top_k(scores, topn, sorted=True)
    if shuffle_ind is not None:
      indices = tf.gather(shuffle_ind, indices, batch_dims=1, axis=1)
    return [tf.gather(f, indices, batch_dims=1, axis=1) for f in features_list]

#######################################
# Utils.

def get_arg(args, name: str, default=None, assert_defined=False):
    value = getattr(args, name) if hasattr(args, name) else default
    if assert_defined:
        assert value is not None, 'Arg %s is not defined: %s' % (name, str(args))
    return value

def read_rules(path,args):
    rules = []
    with open(path, 'r') as f:
        for line in f:
            # if len(rules) < 11:
            # split by :
            line = line.split(':')
            # first element is the name of the rule
            rule_name = line[0]
            # second element is the weight of the rule
            rule_weight = float(line[1].replace(',', '.'))
            # third element is the rule itself. Split by ->
            rule = line[2].split('->')
            # second element is the head of the rule
            rule_head = rule[1]
            # remove the \n from the head and the space
            rule_head = [rule_head[1:-1]]
            # first element is the body of the rule
            rule_body = rule[0]
            # split the body by ,
            rule_body = rule_body.split(', ')
            # for every body element, if the last character is a " ", remove it
            for i in range(len(rule_body)):
                if rule_body[i][-1] == " ":
                    rule_body[i] = rule_body[i][:-1]
            # Take the vars of the body and head and put them in a dictionary
            all_vars = rule_body + rule_head
            var_names = {}
            for i in range(len(all_vars)):
                # split the element of the body by (
                open_parenthesis = all_vars[i].split('(')
                # Split the second element by )
                variables = open_parenthesis[1].split(')')
                # divide the variables by ,
                variables = variables[0].split(',')
                # Create a dictionary with the variables as keys and the value "countries" as values
                if 'nations' in args.dataset_name:
                    for var in variables:
                        var_names[var] = "countries"
                elif ('countries' in args.dataset_name) or ('test_dataset' in args.dataset_name):
                        var_names = {"X": "countries", "W": "subregions", "Z": "regions", "Y": "countries", "K": "countries"}
                        if 'nodomain' in args.dataset_name:
                            for var in variables:
                                var_names[var] = "cte" 
                elif 'kinship' in args.dataset_name:
                    # var_names = {"x": "people", "y": "people", "z": "people","a": "people", "b": "people","c": "people","d": "people"}
                    for var in variables:
                        var_names[var] = "people" 
                else: 
                    for var in variables:
                        var_names[var] = "cte" 

            # print all the info
            # if len(rules) < 1001:
            #     print('rule name: ', rule_name, 'rule weight: ', rule_weight, 'rule head: ', rule_head, 
            #         'rule body: ', rule_body, 'var_names: ', var_names)
            rules.append(Rule(name=rule_name,var2domain=var_names,body=rule_body,head=rule_head))
    print('number of rules: ', len(rules))
    return rules

# Recursive flattening to a 1D Iterable. TODO(check if this works).
# Example usages:
# Flatten([[1,2,3], 2, ['a','b']], tuple) -> (1,2,3,2,'a','b')
# Flatten([[1,2,3], 2, ['a','b']], list) -> [1,2,3,2,'a','b']
# Flatten(([1,2,3], 2, ('a','b')), list) -> [1,2,3,2,'a','b']
# Flatten(([1,2,3], 2, ('a','b')), np.array) -> np.array(1,2,3,2,'a','b')
def Flatten(lst: Iterable, flattening_function=tuple) -> Iterable:
    return flattening_function(Flatten(i, flattening_function)
                               if (not isinstance(elem, str) and
                                   isinstance(i, Iterable))
                               else i for i in lst)

# Can we just use the one above? Or this one? TODO cleanup.
def to_flat(nestedList):
    ''' Converts a nested list to a flat list '''
    flatList = []
    # Iterate over all the elements in given list
    for elem in nestedList:
        # Check if type of element is list
        if not isinstance(elem, str) and isinstance(elem, list):
            # Extend the flat list by adding contents of this element (list)
            flatList.extend(to_flat(elem))
        else:
            # Append the element to the list
            flatList.append(elem)
    return flatList

# Skips lines starting with comment_start, otherwise all lines are kept.
def read_file_as_lines(file, allow_empty=True, comment_start='#'):
    try:
        with open(file, 'r') as f:
            if comment_start:
                return [line.rstrip() for line in f.readlines()
                        if line[:len(comment_start)] != comment_start]
            else:
                return [line.rstrip() for line in f.readlines()]
    except IOError as e:
        if allow_empty:
            return []
        raise IOError("Couldn't open file (%s)" % file)

# Used by operation, TODO move to logic/commons.py or remove as there are already methods to do this.
def parse_atom(atom):
    spls = atom.split("(")
    atom_str = spls[0].strip()
    constant_str = spls[1].split(")")[0].split(",")

    return [atom_str] + [c.strip() for c in constant_str]


################################
# Callbacks.
class ActivateFlagAt(tf.keras.callbacks.Callback):
    """Activate a boolean tf.Variable at the beginning of a specific epoch"""

    def __init__(self, flag: tf.Variable, at_epoch : int):
        super().__init__()
        self.flag  = flag
        self.at_epoch = at_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch  == self.at_epoch - 1:
            print("Activating flag %s at epoch %d" % (self.flag.name, epoch + 1))
            self.flag.assign(True)

class PrintEachEpochCallback(tf.keras.callbacks.Callback):

    def __init__(self, model, fun):
        super().__init__()
        self.model  = model
        self.fun = fun

    def on_epoch_end(self, epoch, logs=None):
        print(self.fun(self.model))


# Checkpointer that allows both in-memory and filename checkpointing.
# This extends the functionalities of tf.keras.callbacks.ModelCheckpoint,
# which can save only to file and it is much slower for small
# non-persistent tests.
class MMapModelCheckpoint(tf.keras.callbacks.Callback):
  """Save models to Memory or files as a Keras callback."""

  def __init__(self, model: tf.keras.Model,
               monitor: str='val_loss',
               maximize: bool=True,
               verbose: bool=True,
               filepath: str=None,
               frequency: int = 1):

    self._model = model
    self.best_val = -sys.float_info.max if maximize else sys.float_info.max
    self.monitor = monitor
    self._weights_saved: bool = False
    self._best_weights = None
    self.best_epoch = None
    self.maximize = maximize
    self.verbose = verbose
    self.frequency = frequency
    # Basepath where checkpoints are saved.
    self._filepath: str = filepath
    self._last_checkpoint_filename: str = None

  def on_epoch_end(self, epoch, logs):
    if (epoch+1) % self.frequency != 0:
        return

    assert self.monitor  in logs, (
        'Unknown metric %s at epoch %d. Use the MMapModelCheckpoint.frequency if you are not validating at each step' % (self.monitor, epoch))
    val = logs[self.monitor]
    if (self.maximize and val >= self.best_val) or (
        not self.maximize and val <= self.best_val):
      self.best_val = val
      self.best_epoch = epoch
      if self.verbose:
        print('Checkpointing %s: new best val (%.3f)' % (self.monitor, val), flush=True)
      if self._filepath is not None:
          filename = '%s__epoch%d.ckpt' % (self._filepath, epoch)
          self._model.save_weights(filename)
          if self.verbose:
              print('Weights stored to %s' % filename, flush=True)
          self._last_checkpoint_filename = filename
      else:
          self._best_weights = self._model.get_weights()
      self._weights_saved = True


  def restore_weights(self):
    if not self._weights_saved:
        print('Can not restore the weights as they have not been saved yet')
        return

    assert self._model is not None
    if self.verbose:
        print('Restoring weights from epoch', self.best_epoch)

    if self._last_checkpoint_filename is not None:
        print('Restoring from file %s' % self._last_checkpoint_filename)
        self._model.load_weights(self._last_checkpoint_filename)
    else:
        # In memory restoring.
        assert self._best_weights is not None
        self._model.set_weights(self._best_weights)

#############################################
# Runtime utils.
class NSParser(argparse.ArgumentParser):

    def __init__(self):
        super().__init__()

        self.add_argument("-k", "--kge", type=str,
                            default='complex', help="The KGE embedder.")
        self.add_argument("-a", "--kge_atom_embedding_size", type=int,
                            default=10, help="Atom embedding size.")
        self.add_argument("-rd", "--reasoning_depth", type=int,
                            default=1, help="Reasoning depth.")
        self.add_argument("-e", "--epochs", type=int,
                            default=200, help="Epoch number for training.")
        self.add_argument("-s", "--seed", default=0, type=int,
                            help="Seed for random generators.")
        self.add_argument("-lr", "--learning_rate", default=0.01, type=float,
                            help="Learning rate.")

class Logger():


    def __init__(self, file):
        self.file = file
        if os.path.exists(self.file):
            self.df = pd.read_csv(self.file)
        else:
            self.df = None



    def log(self, args:dict):
        if self.df is None:
            self.df = pd.DataFrame(columns=[k for k in args.keys()])
        self.df = self.df.append(args, ignore_index = True)
        self.df.to_csv(self.file, index=False)


    def exists(self, args:dict):
        if self.df is None:
            return False
        ddf = self.df[list(args.keys())]
        s = pd.Series(args)
        r = (ddf == s)
        res = bool(r.all(axis=1).any())
        return res


from keras.callbacks import CSVLogger
import csv

class CustomCSVLogger(CSVLogger):
    def __init__(self, filename, separator=';', append=False):
        super().__init__(filename, separator, append)
        
    def on_train_begin(self, logs=None):
        logs = logs or {}

        if self.append:
            if tf.io.gfile.exists(self.filename):
                with tf.io.gfile.GFile(self.filename, "r") as f:
                    self.append_header = not bool(len(f.readline()))
            mode = "a"
        else:
            mode = "w"
        self.csv_file = tf.io.gfile.GFile(self.filename, mode)

    def on_train_end(self, logs=None):
        header = ['epoch'] + list(logs.keys())
        if logs is not None:
            self.csv_file.write(';'.join(header) + '\n') 
        self.csv_file.close()
        self.writer = None


class FileLogger():

    def __init__(self, folder='.\log_folder',folder_experiments='.\log_folder\experiments',folder_run='.\log_folder\indiv_runs'):
        self.folder = folder
        self.folder_experiments = folder_experiments
        self.folder_run = folder_run
        if not os.path.exists(folder): os.mkdir(folder)
        if not os.path.exists(folder_experiments): os.mkdir(folder_experiments)
        if not os.path.exists(folder_run): os.mkdir(folder_run)

    def _read_last_line(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            return lines[-2],lines[-1]

    def log(self, args:dict, filename):
        """Append`the results as last line of a filename"""
        header_filename = os.path.join(self.folder_run, "header.txt")
        if not os.path.exists(header_filename):
            header = [str(a) for a in list(args.keys())]
            with open(header_filename, "w") as f:
                f.write(",".join(header))
        with open(filename, "a") as f:
            f.write("\n")
            f.write('All data;')
            f.write(";".join(['%s:%s' % (str(k), str(v)) for k,v in list(args.items())]))
            f.write('\nSignature;')
            f.write(str(args['run_signature']))
            f.write('\nSeed;')
            f.write(str(args['seed_run_i']))
            f.write('\nTotal_Seeds;')
            f.write(str(args['seed']))

    def get_date(self):
        date = str(datetime.datetime.now()).replace(":","_").replace(" ","_").replace("-","_")
        date = date[:date.index('.')]
        return date
    

    def exists_experiment(self, args:dict):
        
        if not os.path.exists(os.path.join(self.folder_experiments,'experiments.csv')):
            return False
        # if there are no csv files starting with experiments, return False
        experiments_files = [f for f in os.listdir(self.folder_experiments) if f.startswith('experiments')]
        if len(experiments_files) == 0:
            return False

        # open the experiments file. for every line, check if the signature is in the line
        headers = None
        for file in experiments_files:
            # print('\nfile',file)
            with open(os.path.join(self.folder_experiments,file), 'r') as f:
                lines = f.readlines()
                for j,line in enumerate(lines):
                    # Take the index of the column 'run_signature', to look for the signature
                    if j == 0:
                        # if line starts with 'sep', continue
                        if line.startswith('sep'):
                            continue
                        else:
                            headers = line.split(';')
                            pos_1 = headers.index('run_signature')
                        continue
                    if j == 1:
                        if headers is None:
                            headers = line.split(';')
                            pos_1 = headers.index('run_signature')
                        continue
                    else: # if the line is not empty
                        if line == '\n' or line == '' or line.startswith(';;;;;') or line.startswith('<')  or line.startswith('='):
                            continue 
                        # look for the signature
                        file_signature = line.split(';')[pos_1]
                        if file_signature in args['run_signature']:
                            return True
        return False

    def exists_run(self, args:dict,log_filename_tmp,seed):
        # Check if the training has already been done for this seed
        # in log_filename_tmp,to not take into account the time, split by ';'  and take up to the last two elements 
        sub_signature = log_filename_tmp.split('-')[1:-3]
        # addd the seed
        sub_signature.append(str('seed_'+str(seed)))
        # read all the files in the folder_run
        all_files = os.listdir(self.folder_run) 
        for file in all_files: 
            # if the file contains the sub_signature, then the training has been done
            if all(sub in file for sub in sub_signature):
                return True
        return False

    def write_to_csv(self, to_write):
        lines = []
        for filename in os.listdir(self.folder):
            if filename.startswith("log"):
                last_line = self._read_last_line(os.path.join(self.folder,filename))
                lines.append(last_line)
            if filename.startswith("header"):
                header = self._read_last_line(os.path.join(self.folder,filename))
        with open(os.path.join(self.folder,to_write), "w") as f:
            f.write(header + "\n")
            for line in lines:
                f.write(line+"\n")

    def get_avg_results(self, run_signature, seeds):
        """
        Calculate the average results from multiple experiment runs with different seeds.

        Args:
            run_signature (str): Unique identifier present in the filenames of experiment result files.
            seeds (list): List of seeds used in the experiments.

        Returns:
            tuple: A tuple containing average results dictionary and list of metric names.
                - The average results dictionary contains the average and standard deviation of each metric.
                - The list of metric names indicates the names of metrics used by the model.
        """
        # Get all files in the run folder
        all_files = os.listdir(self.folder_run)
        
        # Filter files based on run_signature
        run_files = [file for file in all_files if run_signature in file] 
        
        # Check if the number of files matches the number of seeds
        n_files = len(run_files) 
        if n_files < len(seeds):
            print('The number of files', n_files, 'found in the experiments is different from the number of seeds', len(seeds), '(', seeds, ')!!!!!!!')
            return None, None
        
        # Initialize dictionaries and lists to store results
        info_results = {}  # Dictionary with the info of the metrics of the run
        metrics_names = []  # Metrics used by the model
        seeds_found = set()  # Set to keep track of found seeds
        
        # Iterate through each file
        for file in run_files:
            with open(os.path.join(self.folder_run, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # Find lines starting with 'All data'
                    if line.startswith('All data'):
                        d = line.split(';')[1:]
                        d[-1] = d[-1][:-1]  # Remove newline character from the last element
                        
                        # Extract info about accuracy and time
                        info_run = {el.split(':')[0]: ast.literal_eval(el.split(':')[1]) for el in d if el.split(':')[0] in ['train_acc', 'valid_acc', 'test_acc', 'time_run']}
                        info_run['time'] = [float(el.split(':')[1]) for el in d if el.split(':')[0] in ['time_train', 'time_inference', 'time_ground_train', 'time_ground_valid', 'time_ground_test']]
                        
                        # Get the names of the metrics
                        metrics_names = [ast.literal_eval(el.split(':')[1]) for el in d if el.split(':')[0] == 'metrics'][0]
                        
                        # Get the seed used in the run
                        seed_found = [ast.literal_eval(el.split(':')[1]) for el in d if el.split(':')[0] == 'seed_run_i'][0]
                        seeds_found.add(seed_found)
                        
                        # Append results to info_results if seed matches
                        if seed_found in seeds:
                            for key in info_run.keys():
                                if key in info_results:
                                    info_results[key].append(np.round(info_run[key], 3))
                                else:
                                    info_results[key] = [np.round(info_run[key], 3)]
        
        # Check if all seeds were found
        if len(seeds_found) != len(seeds):
            print('The number of seeds', seeds_found, 'found in the experiments is different from the number of seeds', seeds, '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        
        assert len(seeds_found) == len(seeds), 'The number of seeds found in the experiments folder is different from the number of seeds you set in the code'
        
        # Calculate average and standard deviation for each metric
        for key in info_results.keys():
            avg = np.mean(info_results[key], axis=0)
            std = np.std(info_results[key], axis=0)
            info_results[key] = [list(avg), list(std)] 
        
        return info_results, metrics_names

    def write_avg_results(self, args_dict, info_results, metrics_name):
        """
        Write average results to a CSV file along with experiment parameters.

        Args:
            args_dict (dict): Dictionary containing experiment parameters.
            info_results (dict): Dictionary containing average results and standard deviations.
            metrics_name (list): List of metric names used in the experiment.

        Returns:
            None
        """
        file_csv = os.path.join(self.folder_experiments, 'experiments.csv')
        
        # Remove 'contrastive_loss' from metric names if present
        if 'contrastive_loss' in metrics_name:
            metrics_name.remove('contrastive_loss')

        # Construct column names for CSV file
        names_metrics = [str(metric) for metric in list(metrics_name) if not metric.startswith('val')]
        metrics = [str(element) + '_' + str(metric) for element in ['train', 'val', 'test'] for metric in names_metrics]
        metrics += ['time_ground_train', 'time_ground_valid', 'time_train', 'time_ground_test', 'time_inference']
        combined_names = ';'.join(list(args_dict.keys()) + [str(metric) for metric in metrics])

        # Combine parameter values and average results
        values_args = [str(v) for k, v in args_dict.items()]
        values_metrics = []
        for k, v in info_results.items():
            for i in range(len(v[0])):
                values_metrics.append(str([np.round(v[0][i], 3), np.round(v[1][i], 3)]))
        combined_results = ';'.join(values_args + values_metrics)

        print("Writing results to", file_csv)

        # Write combined results to CSV file
        with open(file_csv, 'a') as f:
            empty = os.stat(file_csv).st_size == 0
            if empty:
                f.write('sep=;\n')
                f.write(combined_names)
            f.write('\n')
            f.write(combined_results)

        # # Write column names to header.txt if not already present
        # header_file = os.path.join(self.folder_experiments, 'header.txt')
        # if not os.path.exists(header_file):
        #     with open(header_file, 'w') as f:
        #         f.write('sep=;\n')
        #         f.write(combined_names)
        # else:
        #     with open(header_file, 'r') as f:
        #         lines = f.readlines()
        #         if combined_names not in lines:
        #             with open(header_file, 'a') as f:
        #                 f.write('\n')
        #                 f.write(combined_names)
                        
        # # Write run signature to signature.txt
        # with open(os.path.join(self.folder_experiments, 'signature.txt'), 'a') as f:
        #     f.write(args_dict['run_signature'])
        #     f.write('\n')
        
        return None



class BinaryCrossEntropyRagged(tf.keras.losses.Loss):
    def __init__(self, balance_negatives=False, from_logits=False):
        super().__init__()
        self.balance_negatives = balance_negatives
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if (isinstance(y_true, tf.RaggedTensor) or
            isinstance(y_pred, tf.RaggedTensor)):
            y_true = y_true.to_tensor()
            y_pred = y_pred.to_tensor()
        loss = tf.keras.losses.binary_crossentropy(
            y_true, y_pred, from_logits=self.from_logits)

        if self.balance_negatives:
            num_positives = tf.reduce_sum(tf.where(y_true == 1, 1.0, 0.0),
                                          axis=-1, keepdims=True)
            num_negatives = tf.reduce_sum(tf.where(y_true == 0, 1.0, 0.0),
                                          axis=-1, keepdims=True)
            loss_positive = tf.math.divide_no_nan(
                tf.where(y_true == 1, loss, 0.0),
                tf.expand_dims(num_positives, axis=-1))
            loss_negative = tf.math.divide_no_nan(
                tf.where(y_true == 0, loss, 0.0),
                tf.expand_dims(num_negatives, axis=-1))
            loss = loss_positive + loss_negative
        return loss

class PairwiseCrossEntropyRagged(tf.keras.losses.Loss):
    def __init__(self, balance_negatives=False, from_logits=False):
        super().__init__()
        self.balance_negatives = balance_negatives
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if (isinstance(y_true, tf.RaggedTensor) or
            isinstance(y_pred, tf.RaggedTensor)):
            y_true = y_true.to_tensor()
            y_pred = y_pred.to_tensor()
        pos_loss = -tf.reduce_sum(tf.where(y_true == 1, tf.math.log(1e-7 + tf.nn.sigmoid(y_pred)), 0.0),
                                  axis=-1, keepdims=True)
        neg_loss = -tf.reduce_sum(tf.where(y_true == 0, tf.math.log(1e-7 + tf.nn.sigmoid(-y_pred)), 0.0),
                                  axis=-1, keepdims=True)
        if self.balance_negatives:
            num_positives = tf.reduce_sum(tf.where(y_true == 1, 1.0, 0.0),
                                          axis=-1, keepdims=True)
            num_negatives = tf.reduce_sum(tf.where(y_true == 0, 1.0, 0.0),
                                          axis=-1, keepdims=True)

        loss = tf.squeeze(pos_loss + neg_loss)
        return loss


class CategoricalCrossEntropyRagged(tf.keras.losses.Loss):
    def __init__(self, from_logits=False):
        super().__init__()
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if (isinstance(y_true, tf.RaggedTensor) or
            isinstance(y_pred, tf.RaggedTensor)):
            y_true = y_true.to_tensor()
            y_pred = y_pred.to_tensor()
        return tf.keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits)

######################################
# KGE utils and metrics, this is application dependent, TODO move to kge.py?
def KgeLossFactory(name: str) -> tf.keras.losses.Loss:
    if name == 'hinge':
        return HingeLossRagged(gamma=1.0)
    elif name == 'l2':
        return L2LossRagged()
    elif name == 'categorical_crossentropy':
        return CategoricalCrossEntropyRagged()
    elif name == 'binary_crossentropy':
        return BinaryCrossEntropyRagged()
    elif name == 'balanced_binary_crossentropy':
        return BinaryCrossEntropyRagged(balance_negatives=True)
    elif name == 'balanced_pairwise_crossentropy':
        return PairwiseCrossEntropyRagged(balance_negatives=True)
    else:
        assert False, 'Unknown loss %s'% name

class MRRMetric(tf.keras.metrics.Metric):
  """Implements mean reciprocal rank (MRR). It uses the same implementation
     of tensorflow_ranking MRRMetric but with an online check for ragged
     tensors."""
  def __init__(self, name='mrr', dtype=None, **kwargs):
      super().__init__(name, dtype, **kwargs)
      self.mrr = self.add_weight("total", initializer="zeros")
      self._count = self.add_weight("count", initializer="zeros")
      self.reset_state()

  def reset_state(self):
      self.mrr.assign(0.)
      self._count.assign(0.)

  def result(self):
      return tf.math.divide_no_nan(self.mrr, self._count)

  def update_state(self, y_true, y_pred, sample_weight=None):
    mrrs = self._compute(y_true, y_pred)
    self.mrr.assign_add(tf.reduce_sum(mrrs))
    self._count.assign_add(tf.reduce_sum(tf.ones_like(mrrs)))

  def _compute(self, labels, predictions):
    if any(isinstance(tensor, tf.RaggedTensor)
           for tensor in [labels, predictions]):
      labels, predictions, _, _ = ragged_to_dense(labels, predictions, None)

    topn = tf.shape(predictions)[1]
    sorted_labels, = sort_by_scores(predictions, [labels], topn=topn, mask=None)
    sorted_list_size = tf.shape(input=sorted_labels)[1]
    # Relevance = 1.0 when labels >= 1.0 to accommodate graded relevance.
    relevance = tf.cast(tf.greater_equal(sorted_labels, 1.0), dtype=tf.float32)
    reciprocal_rank = 1.0 / tf.cast(
        tf.range(1, sorted_list_size + 1), dtype=tf.float32)
    # MRR has a shape of [batch_size, 1].
    mrr = tf.reduce_max(
        input_tensor=relevance * reciprocal_rank, axis=1, keepdims=True)
    return mrr

# Wrapper around tf.keras.metrics.AUC adding the convertion to ragged tensors.
class AUCPRMetric(tf.keras.metrics.AUC):
  """Implements mean reciprocal rank (MRR). It uses the same implementation
     of tensorflow_ranking MRRMetric but with an online check for ragged
     tensors."""
  def __init__(self, name='auc-pr', dtype=None, **kwargs):
      super().__init__(curve='PR', name=name, dtype=dtype)

  def _compute(self, labels, predictions):
    if any(isinstance(tensor, tf.RaggedTensor)
           for tensor in [labels, predictions]):
      labels, predictions, _, _ = ragged_to_dense(labels, predictions, None)
    return super()._compute(labels, predictions)

class HitsMetric(tf.keras.metrics.Metric):
  """Implements the HITS@N metric. It uses the same implementation
     of tensorflow_ranking MRRMetric but with an online check for ragged
     tensors."""
  def __init__(self, n, name='hits', dtype=None, **kwargs):
      super().__init__('%s@%d' % (name, n), dtype, **kwargs)
      self._n = n
      self.hits = self.add_weight("total", initializer="zeros")
      self._count = self.add_weight("count", initializer="zeros")
      self.reset_state()

  def reset_state(self):
      self.hits.assign(0.)
      self._count.assign(0.)

  def result(self):
      return tf.math.divide_no_nan(self.hits, self._count)

  def update_state(self, y_true, y_pred, sample_weight=None):
    hits = self._compute(y_true, y_pred)
    self.hits.assign_add(tf.reduce_sum(hits))
    self._count.assign_add(tf.reduce_sum(tf.ones_like(hits)))

  def _compute(self, labels, predictions):
    if any(isinstance(tensor, tf.RaggedTensor)
           for tensor in [labels, predictions]):
      labels, predictions, _, _ = ragged_to_dense(labels, predictions, None)

    topn = tf.shape(predictions)[1]
    sorted_labels, = sort_by_scores(predictions, [labels], topn=topn, mask=None)
    sorted_list_size = tf.shape(input=sorted_labels)[1]
    # Relevance = 1.0 when labels >= 1.0 to accommodate graded relevance.
    relevance = tf.cast(tf.greater_equal(sorted_labels, 1.0), dtype=tf.float32)
    top_relevance = relevance[:, :self._n]
    hits = tf.reduce_sum(top_relevance, axis=1, keepdims=True)
    return hits

  def get_config(self):
      base_config = super().get_config()
      config = { 'n': self._n, }
      return {**base_config, **config}


def get_model_memory_usage(batch_size, model):
    import numpy as np
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


#class NTPMRR():
#    def __call__(self, y_true, y_pred, *args, **kwargs):
#        rank_l = 1. + tf.cast(tf.argsort(tf.argsort(- y_pred))[:, 0], tf.float32)
#        mrr =  1.0 / rank_l
#        return np.mean(mrr, axis=0)
