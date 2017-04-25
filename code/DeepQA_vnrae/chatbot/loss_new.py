# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Seq2seq loss operations for use in sequence models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops

__all__ = ["sequence_loss"]


def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):

    if len(logits.get_shape()) != 3:
        raise ValueError("Logits must be a "
                         "[batch_size x sequence_length x logits] tensor")
    if len(targets.get_shape()) != 2:
        raise ValueError("Targets must be a [batch_size x sequence_length] "
                         "tensor")
    if len(weights.get_shape()) != 2:
        raise ValueError("Weights must be a [batch_size x sequence_length] "
                         "tensor")
    with ops.name_scope(name, "sequence_loss", [logits, targets, weights]):
        num_classes = logits.get_shape()[2].value
        probs_flat = array_ops.reshape(logits, [-1, num_classes])
        targets = array_ops.reshape(targets, [-1])
        if softmax_loss_function is None:
            crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                labels=targets, logits=probs_flat)
        else:
            crossent = softmax_loss_function(probs_flat, targets)
        crossent = crossent * array_ops.reshape(weights, [-1])
        if average_across_timesteps and average_across_batch:
            crossent = math_ops.reduce_sum(crossent)
            total_size = math_ops.reduce_sum(weights)
            total_size += 1e-12 # to avoid division by 0 for all-0 weights
            crossent /= total_size
        else:
            batch_size = array_ops.shape(logits)[0]
            sequence_length = array_ops.shape(logits)[1]
            crossent = array_ops.reshape(crossent, [batch_size, sequence_length])
        if average_across_timesteps and not average_across_batch:
            crossent = math_ops.reduce_sum(crossent, axis=[1])
            total_size = math_ops.reduce_sum(weights, axis=[1])
            total_size += 1e-12 # to avoid division by 0 for all-0 weights
            crossent /= total_size
        if not average_across_timesteps and average_across_batch:
            crossent = math_ops.reduce_sum(crossent, axis=[0])
            total_size = math_ops.reduce_sum(weights, axis=[0])
            total_size += 1e-12 # to avoid division by 0 for all-0 weights
            crossent /= total_size
        return crossent