# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions to support building models for StreetView text transcription."""

import tensorflow as tf
from tensorflow.contrib import slim
import re, logging

def logits_to_log_prob(logits):
  """Computes log probabilities using numerically stable trick.

  This uses two numerical stability tricks:
  1) softmax(x) = softmax(x - c) where c is a constant applied to all
  arguments. If we set c = max(x) then the softmax is more numerically
  stable.
  2) log softmax(x) is not numerically stable, but we can stabilize it
  by using the identity log softmax(x) = x - log sum exp(x)

  Args:
    logits: Tensor of arbitrary shape whose last dimension contains logits.

  Returns:
    A tensor of the same shape as the input, but with corresponding log
    probabilities.
  """

  with tf.variable_scope('log_probabilities'):
    reduction_indices = len(logits.shape.as_list()) - 1
    #keep_dims: Deprecated alias for `keepdims`.
    #reduction_indices: The old (deprecated) name for axis.
    max_logits = tf.reduce_max(
        logits, axis=reduction_indices, keepdims=True)
    safe_logits = tf.subtract(logits, max_logits)
    sum_exp = tf.reduce_sum(
        tf.exp(safe_logits),
        axis=reduction_indices,
        keepdims=True)
    log_probs = tf.subtract(safe_logits, tf.log(sum_exp))
  return log_probs


def variables_to_restore(scope=None, strip_scope=False):
  """Returns a list of variables to restore for the specified list of methods.

  It is supposed that variable name starts with the method's scope (a prefix
  returned by _method_scope function).

  Args:
    methods_names: a list of names of configurable methods.
    strip_scope: if True will return variable names without method's scope.
      If methods_names is None will return names unchanged.
    model_scope: a scope for a whole model.

  Returns:
    a dictionary mapping variable names to variables for restore.
  """
  if scope:
    variable_map = {}
    method_variables = slim.get_variables_to_restore(include=[scope])
    for var in method_variables:
      if strip_scope:
        var_name = var.op.name[len(scope) + 1:]
      else:
        var_name = var.op.name
      variable_map[var_name] = var

    return variable_map
  else:
    return {v.op.name: v for v in slim.get_variables_to_restore()}



def reverse_dict(m_dict):
    return dict(zip(m_dict.values(), m_dict.keys()))

def read_dict(filename, null_character=u'\u2591'):
    pattern = re.compile(r'(\d+)\t(.+)')
    charset = {}
    with tf.gfile.GFile(filename) as f:
        for i, line in enumerate(f):
            m = pattern.match(line)
            if m is None:
                charset[" "] = 0
                logging.warning('incorrect charset file. line #%d: %s', i, line)
                continue
            code = int(m.group(1))
            char = m.group(2)  # .decode('utf-8')
            if char == '<nul>':
                char = null_character
            # charset[code] = char
            charset[char] = code
    return charset

def decode_code(code):
    if type(code) == bytes:
        return code.decode("utf-8")
    #str(code, encoding='utf-8')
    return code

def encode_code(code):
    if type(code) == str:
        return code.encode("utf-8")
    #bytes(code, 'utf-8')
    return code


def read_charset(filename, null_character=u'\u2591'):
    pattern = re.compile(r'(\d+)\t(.+)')
    charset = {}
    with tf.gfile.GFile(filename) as f:
        for i, line in enumerate(f):
            m = pattern.match(line)
            if m is None:
                #charset[0] = " "
                logging.warning('incorrect charset file. line #%d: %s', i, line)
                continue
            code = int(m.group(1))
            char = m.group(2)  # .decode('utf-8')
            if char == '<nul>':
                char = null_character
            charset[code] = char
    return charset

def _dict_to_array(id_to_char, default_character):
  num_char_classes = max(id_to_char.keys()) + 1
  array = [default_character] * num_char_classes
  for k, v in id_to_char.items():
    array[k] = v
  return array

class CharsetMapper(object):
  """A simple class to map tensor ids into strings.

    It works only when the character set is 1:1 mapping between individual
    characters and individual ids.

    Make sure you call tf.tables_initializer().run() as part of the init op.
    """

  def __init__(self, charset, default_character='?'):
    """Creates a lookup table.

    Args:
      charset: a dictionary with id-to-character mapping.
    """
    mapping_strings = tf.constant(_dict_to_array(charset, default_character))
    self.table = tf.contrib.lookup.index_to_string_table_from_tensor(
      mapping=mapping_strings, default_value=default_character)

  def get_text(self, ids):
    """Returns a string corresponding to a sequence of character ids.

        Args:
          ids: a tensor with shape [batch_size, max_sequence_length]
        """
    return tf.reduce_join(
      self.table.lookup(tf.to_int64(ids)), reduction_indices=1)
