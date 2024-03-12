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
from tensorflow_ranking.python.utils import sort_by_scores, ragged_to_dense
from keras_ns.logic.commons import Atom, Domain, Rule, RuleLoader

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

def add_if_not_in(to_insert: list, full_list:list, reference_set: set):
    for a in to_insert:
        if a not in reference_set:
            reference_set.add(a)
            full_list.append(a)

@tf.custom_gradient
def differentiable_sign(x):
    sign = tf.sign(x)
    def grad(x):
        return tf.gradients(tf.sigmoid(x), x)
    return sign, grad

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

def parse_atom(atom):
    spls = atom.split("(")
    atom_str = spls[0].strip()
    constant_str = spls[1].split(")")[0].split(",")

    return [atom_str] + [c.strip() for c in constant_str]

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
            # Append the elemengt to the list
            flatList.append(elem)
    return flatList


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
          # remove previous checkpoints. Remove all the files in self._filepath
          print('path',self._filepath)
          if os.path.exists(self._filepath):
              for file in os.listdir(self._filepath):
                  os.remove(os.path.join(self._filepath, file))
          filename = '%s\epoch%d.ckpt' % (self._filepath, epoch)
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


def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)


import sys
import select

def heardEnter():
    i,o,e = select.select([sys.stdin],[],[],0.0001)
    for s in i:
        if s == sys.stdin:
            input = sys.stdin.readline()
            return True
    return False

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


from functools import wraps
import pickle


def list2tuple(t):
    if type(t) == list:
        return tuple([list2tuple(f) for f in t])
    return t



def cached(func):
    cache_filename = "global_cache.dat"
    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {}
    @wraps(func)
    def wrapper(*args):
        try:
            return cache[args]
        except KeyError:
            cache[args] = result = func(*args)
            with open(cache_filename, 'wb') as f:
                pickle.dump(cache, f)
            return result
    return wrapper

def persist_to_file(file_name):

    def decorator(original_func):

        try:
            with open(file_name, 'rb') as f:
                cache = pickle.load(f)
        except (IOError, ValueError):
            cache = {}

        def new_func(*param):
            if param not in cache:
                cache[param] = original_func(*param)
                pickle.dump(cache, open(file_name, 'wb'))
            return cache[param]

        return new_func

    return decorator



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

    def __init__(self, folder,folder_experiments,folder_run):
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
            with open(os.path.join(self.folder_experiments,file), 'r') as f:
                lines = f.readlines()
                for j,line in enumerate(lines):
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
                    else:# if the line is not empty
                        if line == '\n' or line == '' or line.startswith(';;;;;'):
                            continue 
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
        # read all the files
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

    def get_avg_results(self,run_signature,seeds):
        # For every file with a different seed, read the results and take the average
        all_files = os.listdir(self.folder_run)
        # get the files that contain the run_signature
        run_files = [file for file in all_files if run_signature in file]
        # get the number of files
        # print('run_signature',run_signature)
        # print('all files',all_files)
        n_files = len(run_files)
        # print('n_files',n_files,'seeds',seeds)
        # print('run_files',run_files)
        if n_files < len(seeds):
            print('The number of files',n_files,' found in the experiments is different from the number of seeds',len(seeds),'(',seeds,')!!!!!!!')
            return None,None
        # for every file, read all the lines and if the line starts with 'all_data', take the values and add them to the array
        info = {}   
        metrics_names = []
        seeds_found = set()
        for file in run_files:
            with open(os.path.join(self.folder_run,file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('All data'):
                        d = line.split(';')[1:]
                        # remove the \n from the last element
                        d[-1] = d[-1][:-1]
                        info_exp = {el.split(':')[0] : ast.literal_eval(el.split(':')[1]) for el in d if el.split(':')[0] in ['train_acc', 'valid_acc', 'test_acc','time_run']}
                        # add also time_run
                        info_exp['time'] = [float(el.split(':')[1]) for el in d if el.split(':')[0] in ['time_train','time_inference','time_ground_train','time_ground_valid','time_ground_test']]
                        # print('info_exp',info_exp)
                        # get the names of the metrics from the element 'metrics'
                        metrics_names = [ast.literal_eval(el.split(':')[1]) for el in d if el.split(':')[0] == 'metrics'][0]
                        seed_found = [ast.literal_eval(el.split(':')[1]) for el in d if el.split(':')[0] == 'seed_run_i'][0]
                        # print ('seed_found',seed_found,'metrics_names',metrics_names)
                        seeds_found.add(seed_found)
                        # append the key,values to the dictionary
                        if seed_found in seeds:
                            for key in info_exp.keys():
                                if key in info:
                                    info[key].append(np.round(info_exp[key],3))
                                else:
                                    info[key] = [np.round(info_exp[key],3)]
        for key in info_exp.keys():
            print('key',key,'info_exp[key]',info[key])
        #print a message also
        assert len(seeds_found) == len(seeds), 'The number of seeds found in the experiments folder is different from the number of seeds you set in the code'
        if len(seeds_found) != len(seeds):
            print('The number of seeds found in the experiments is different from the number of seeds!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # for every key in the dictionary, take the average and the std
        for key in info.keys():
            avg = np.mean(info[key],axis=0)
            std = np.std(info[key],axis=0)
            info[key] = [list(avg),list(std)]
        for key in info.keys():
            print('names',metrics_names,'\nkey',key,'info[key]',info[key])
        return info,metrics_names

    def write_avg_results(self,args_dict,info_metrics,metrics_name):
        file_csv = os.path.join(self.folder_experiments,'experiments.csv')
        if 'contrastive_loss' in metrics_name: # delete it from the list metrics_name
            metrics_name.remove('contrastive_loss')
        # join the args_dict with the info_metrics
        # take the metric elements until one element starts with 'val'
        names_metrics = [str(metric) for metric in list(metrics_name) if not metric.startswith('val')]
        metrics =  [str(element)+'_'+str(metric) for element in ['train','val','test'] for metric in names_metrics]
        metrics += ['time_ground_train','time_ground_valid','time_train','time_ground_test','time_inference']
        # metrics = [str(element)+'_'+str(metric) for element in ['train','val','test'] for metric in list(metrics_name)]
        combined_names = ';'.join(list(args_dict.keys()) + [str(metric) for metric in metrics] )
        values_args = [str(v) for k,v in args_dict.items()] 
        values_metrics = []
        for k,v in info_metrics.items():
            print('k',k,'v',v)
            for i in range(len(v[0])):
                values_metrics.append(str([ np.round(v[0][i],3) , np.round(v[1][i],3) ]))
        combined_results = ';'.join(values_args + values_metrics)
        # print('args_dict',list(args_dict.keys()))

        # print('metrics',metrics)
        # print('values_metrics',values_metrics)
        # Create a file for my results 
        print("Writing results to", file_csv)   
    
        # Check if the file folder+'header.txt' exists, otherwise create it
        if not os.path.exists(os.path.join(self.folder_experiments,'header.txt')):
            with open(os.path.join(self.folder_experiments,'header.txt'), 'w') as f:
                f.write('sep=;\n')
                f.write(combined_names)
        else: 
            with open(os.path.join(self.folder_experiments,'header.txt'), 'r') as f:
                lines = f.readlines()
                if combined_names not in lines:
                    with open(os.path.join(self.folder_experiments,'header.txt'), 'a') as f:
                        f.write('\n')
                        f.write(combined_names)
        with open(file_csv, 'a') as f: 
            empty = os.stat(file_csv).st_size == 0
            if empty:
                f.write('sep=;\n')
                f.write(combined_names)
            f.write('\n')
            f.write(combined_results)
        # write also the results in signature.txt
        with open(os.path.join(self.folder_experiments,'signature.txt'), 'a') as f:
            f.write(args_dict['run_signature'])
            f.write('\n')



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


class NTPMRR():
   def __call__(self, y_true, y_pred, *args, **kwargs):
       rank_l = 1. + tf.cast(tf.argsort(tf.argsort(- y_pred))[:, 0], tf.float32)
       mrr =  1.0 / rank_l
       return np.mean(mrr, axis=0)