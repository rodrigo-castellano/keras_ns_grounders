import os.path
import sys

import pandas as pd
import tensorflow as tf
import argparse
import numpy as np
from typing import Dict
import datetime
from tensorflow_ranking.python.utils import sort_by_scores, ragged_to_dense


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





class MMapModelCheckpoint(tf.keras.callbacks.Callback):
  """Save models to Memory as a Keras callback."""

  def __init__(self, model: tf.keras.Model,
               monitor: str='val_loss',
               maximize: bool=True,
               verbose: bool=True,
               frequency: int = 1):

    self.model = model
    self.best_val = -sys.float_info.max if maximize else sys.float_info.max
    self.monitor = monitor
    self.best_weights = None
    self.best_epoch = None
    self.maximize = maximize
    self.verbose = verbose
    self.frequency = frequency

  def restore_weights(self):
    if self.best_weights is None:
        print('Can not restore the weights as they have not been saved yet')
        return

    if self.verbose:
        print('Restoring weights from epoch', self.best_epoch)

    assert self.model is not None
    self.model.set_weights(self.best_weights)

  def on_epoch_end(self, epoch, logs):
    if (epoch+1) % self.frequency != 0:
        return

    assert self.monitor  in logs, 'Unknown metric %s at epoch %d. Use the MMapModelCheckpoint.frequency if you are not validating at each step' % (self.monitor,epoch)
    val = logs[self.monitor]
    if (self.maximize and val >= self.best_val) or (
        not self.maximize and val <= self.best_val):
      self.best_val = val
      self.best_weights = self.model.get_weights()
      self.best_epoch = epoch
      if self.verbose:
        print('\n%s New best val (%.3f)' % (self.monitor, val), flush=True)


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


def generate_countries_data(path_kb):
    regions = ['europe', 'asia', 'africa', 'americas', 'oceania']

    subregions = ["northern_america",
                  "eastern_europe",
                  "australia_and_new_zealand",
                  "melanesia",
                  "micronesia",
                  "eastern_africa",
                  "southern_asia",
                  "eastern_asia",
                  "south_america",
                  "central_europe",
                  "western_asia",
                  "northern_africa",
                  "western_africa",
                  "northern_europe",
                  "middle_africa",
                  "caribbean",
                  "polynesia",
                  "western_europe",
                  "southern_europe",
                  "central_america",
                  "southern_africa",
                  "central_asia",
                  "south_eastern_asia"
                  ]

    states = {'moldova', 'serbia', 'bulgaria', 'zimbabwe', 'madagascar', 'southern_africa', 'portugal',
              'united_states_minor_outlying_islands', 'northern_america', 'kyrgyzstan', 'somalia', 'samoa',
              'canada', 'guam', 'indonesia', 'venezuela', 'paraguay', 'croatia', 'syria', 'andorra',
              'saint_martin', 'japan', 'greenland', 'lithuania', 'czechia', 'guadeloupe', 'iceland', 'italy',
              'cuba', 'marshall_islands', 'australia', 'mayotte', 'svalbard_and_jan_mayen', 'nauru',
              'guatemala', 'panama', 'uruguay', 'liberia', 'iran', 'south_korea', 'north_korea',
              'british_indian_ocean_territory', 'nepal', 'saint_vincent_and_the_grenadines', 'seychelles',
              'slovakia', 'south_georgia', 'libya', 'cameroon', 'uganda', 'belarus', 'aland_islands', 'chad',
              'oman', 'eritrea', 'botswana', 'mexico', 'saint_barthelemy', 'cambodia', 'turkmenistan',
              'timor_leste', 'saint_pierre_and_miquelon', 'british_virgin_islands', 'martinique', 'slovenia',
              'kenya', 'bahamas', 'fiji', 'guinea', 'zambia', 'hong_kong', 'angola', 'honduras', 'namibia',
              'middle_africa', 'curacao', 'pakistan', 'northern_africa', 'greece', 'sudan', 'jamaica',
              'dominican_republic', 'armenia', 'el_salvador', 'yemen', 'turks_and_caicos_islands', 'hungary',
              'qatar', 'saudi_arabia', 'jersey', 'thailand', 'tonga', 'comoros', 'rwanda', 'liechtenstein',
              'luxembourg', 'argentina', 'singapore', 'sint_maarten', 'ethiopia', 'djibouti', 'finland',
              'caribbean', 'south_america', 'kiribati', 'cocos_keeling_islands', 'bangladesh', 'chile', 'iraq',
              'africa', 'kazakhstan', 'micronesia', 'macedonia', 'eastern_europe', 'tanzania', 'maldives',
              'southern_europe', 'barbados', 'south_africa', 'mauritania', 'uzbekistan', 'san_marino',
              'malawi', 'french_polynesia', 'sweden', 'macau', 'grenada', 'western_africa', 'cook_islands',
              'christmas_island', 'myanmar', 'colombia', 'belgium', 'united_arab_emirates', 'montenegro',
              'poland', 'sierra_leone', 'romania', 'bermuda', 'tajikistan', 'montserrat', 'south_eastern_asia',
              'guinea_bissau', 'central_america', 'swaziland', 'netherlands', 'new_caledonia',
              'solomon_islands', 'taiwan', 'anguilla', 'haiti', 'saint_lucia', 'australia_and_new_zealand',
              'nigeria', 'antigua_and_barbuda', 'austria', 'nicaragua', 'vatican_city', 'united_kingdom',
              'costa_rica', 'palau', 'india', 'niger', 'afghanistan', 'isle_of_man', 'guernsey', 'mongolia',
              'togo', 'dominica', 'kuwait', 'norfolk_island', 'cayman_islands', 'denmark', 'reunion',
              'faroe_islands', 'mozambique', 'papua_new_guinea', 'central_europe', 'tokelau',
              'equatorial_guinea', 'switzerland', 'norway', 'sao_tome_and_principe', 'tuvalu', 'lebanon',
              'burkina_faso', 'new_zealand', 'latvia', 'suriname', 'saint_kitts_and_nevis', 'benin',
              'wallis_and_futuna', 'brunei', 'northern_mariana_islands', 'georgia', 'bolivia', 'ireland',
              'french_guiana', 'niue', 'american_samoa', 'ivory_coast', 'burundi', 'cyprus',
              'bosnia_and_herzegovina', 'northern_europe', 'south_sudan', 'france', 'western_asia', 'europe',
              'mali', 'china', 'eastern_africa', 'puerto_rico', 'estonia', 'vanuatu', 'asia', 'philippines',
              'western_sahara', 'algeria', 'americas', 'russia', 'jordan', 'palestine', 'vietnam',
              'trinidad_and_tobago', 'lesotho', 'falkland_islands', 'gabon', 'morocco', 'israel', 'guyana',
              'laos', 'turkey', 'malta', 'monaco', 'polynesia', 'republic_of_the_congo', 'dr_congo', 'egypt',
              'germany', 'pitcairn_islands', 'united_states_virgin_islands', 'kosovo', 'ecuador', 'azerbaijan',
              'gambia', 'melanesia', 'central_asia', 'oceania', 'bhutan', 'spain', 'ghana', 'malaysia',
              'sri_lanka', 'belize', 'cape_verde', 'ukraine', 'united_states', 'eastern_asia', 'mauritius',
              'central_african_republic', 'albania', 'southern_asia', 'gibraltar', 'aruba', 'brazil',
              'western_europe', 'bahrain', 'peru', 'tunisia',
              'senegal'}.difference(subregions).difference(regions)
    states = list(states)

    atoms = read_file_as_lines(path_kb)
    atoms = set([a.replace(".", "") for a in atoms])

    return states, subregions, regions, atoms

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

    def __init__(self, folder):
        self.folder = folder
        if not os.path.exists(folder): os.mkdir(folder)

    def _read_last_line(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            return lines[-2],lines[-1]

    def log(self, args:dict, filename):
        """Append`the results as last line of a filename"""
        header_filename = os.path.join(self.folder, "header.txt")
        if not os.path.exists(header_filename):
            header = [str(a) for a in list(args.keys())]
            with open(header_filename, "w") as f:
                f.write(",".join(header))
        with open(filename, "a") as f:
            f.write("\n")
            f.write('All data-')
            f.write("-".join(['%s:%s' % (str(k), str(v)) for k,v in list(args.items())]))
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
    

    def exists(self, args:dict,signature=None):
        values = [str(a) for a in list(args.values())]
        keys = [str(a) for a in list(args.keys())]
        string_args = ",".join(['%s:%s' % (str(k), str(v)) for k,v in list(args.items())])
        for filename in os.listdir(self.folder):
            if filename.startswith("log"):
                
                last_lines = self._read_last_line(os.path.join(self.folder,filename))
                last_line = "".join(last_lines)

                last_line = last_line.replace(' ', '')
                last_line = last_line.replace('\n', '')

                string_args = string_args.replace(' ', '')
                string_args = string_args.replace('\n', '')
                # print('\nlast_line\n', last_line)
                # print('string_args\n', string_args)
                # look for all the keys in the last line 
                # for k in keys:
                #     if k not in last_line:
                #         # take them out of the signature, as well as the next word splited by "_"
                #         string_args_split = string_args.split("_")
                #         pos = string_args_split.index(k)
                #         # join all except for pos and pos+1
                #         string_args = "_".join(string_args_split[:pos] + string_args_split[pos+2:])
                # print('2string_args\n', string_args)
                if string_args in last_line:
                    # print('string_args in last_line')
                    return True 

                # file_values = [str(a) for a in filename.split("_")]
                # file_values[0] = file_values[0].replace('log', '')
                # # remove the date
                # file_values = file_values[:-1]
                # print('file_values', file_values)
                # print(all ([v in signature for v in file_values]))
                # if all ([v in signature for v in file_values]):
                #     print('all ([v in file_values for v in values])!!!!!!!!!!!!!!!!!!!!!!\n\n\n')
                #     return True

                # # check if all the file_values are in string_args
                # file_values = [str(a) for a in last_line.split(",")]
                # print('file_values', file_values)
                # print('string_args\n', string_args)
                # print([v in string_args for v in file_values])
                # if all([v in string_args for v in file_values]):
                #     print('all([v in string_args for v in file_values])')
                #     print('last_line\n', last_line)
                #     print('string_args\n', string_args)

                # current_values = [str(a) for a in last_line.split(",")]
                # if current_values[:len(values)] == values:
                #     return True
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
    #   super().__init__(name, dtype, **kwargs)
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
    config = {
        'n': self._n,  # Save the value of 'n'
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))



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
