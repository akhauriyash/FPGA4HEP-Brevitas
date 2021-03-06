# MIT License

# Copyright (c) 2019 Xilinx

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.stats import StatsOp


QUANT_TYPE = QuantType.INT
SCALING_MIN_VAL = 2e-16

ACT_SCALING_IMPL_TYPE = ScalingImplType.PARAMETER
ACT_SCALING_PER_CHANNEL = False
ACT_SCALING_RESTRICT_SCALING_TYPE = RestrictValueType.LOG_FP
ACT_MAX_VAL = 1.61
ACT_RETURN_QUANT_TENSOR = False
ACT_PER_CHANNEL_BROADCASTABLE_SHAPE = None
HARD_TANH_THRESHOLD = 10.0

WEIGHT_SCALING_IMPL_TYPE = ScalingImplType.STATS
WEIGHT_SCALING_PER_OUTPUT_CHANNEL = True
WEIGHT_SCALING_STATS_OP = StatsOp.MAX
WEIGHT_RESTRICT_SCALING_TYPE = RestrictValueType.LOG_FP
WEIGHT_NARROW_RANGE = True

ENABLE_BIAS_QUANT = False

HADAMARD_FIXED_SCALE = False



def make_quant_linear(in_channels,
                      out_channels,
                      bias,
                      bit_width,
                      enable_bias_quant=ENABLE_BIAS_QUANT,
                      weight_quant_type=QUANT_TYPE,
                      weight_scaling_impl_type=WEIGHT_SCALING_IMPL_TYPE,
                      weight_scaling_stats_op=WEIGHT_SCALING_STATS_OP,
                      weight_scaling_per_output_channel=WEIGHT_SCALING_PER_OUTPUT_CHANNEL,
                      weight_restrict_scaling_type=WEIGHT_RESTRICT_SCALING_TYPE,
                      weight_narrow_range=WEIGHT_NARROW_RANGE,
                      weight_scaling_min_val=SCALING_MIN_VAL):
    bias_quant_type = QUANT_TYPE if enable_bias_quant else QuantType.FP
    return qnn.QuantLinear(in_channels, out_channels,
                           bias=bias,
                           bias_quant_type=bias_quant_type,
                           compute_output_bit_width=bias and enable_bias_quant,
                           compute_output_scale=bias and enable_bias_quant,
                           weight_bit_width=bit_width,
                           weight_quant_type=weight_quant_type,
                           weight_scaling_impl_type=weight_scaling_impl_type,
                           weight_scaling_stats_op=weight_scaling_stats_op,
                           weight_scaling_per_output_channel=weight_scaling_per_output_channel,
                           weight_restrict_scaling_type=weight_restrict_scaling_type,
                           weight_narrow_range=weight_narrow_range,
                           weight_scaling_min_val=weight_scaling_min_val)



def make_quant_relu(bit_width,
                    quant_type=QUANT_TYPE,
                    scaling_impl_type=ACT_SCALING_IMPL_TYPE,
                    scaling_per_channel=ACT_SCALING_PER_CHANNEL,
                    restrict_scaling_type=ACT_SCALING_RESTRICT_SCALING_TYPE,
                    scaling_min_val=SCALING_MIN_VAL,
                    max_val=ACT_MAX_VAL,
                    return_quant_tensor=ACT_RETURN_QUANT_TENSOR,
                    per_channel_broadcastable_shape=ACT_PER_CHANNEL_BROADCASTABLE_SHAPE):
    return qnn.QuantReLU(bit_width=bit_width,
                         quant_type=quant_type,
                         scaling_impl_type=scaling_impl_type,
                         scaling_per_channel=scaling_per_channel,
                         restrict_scaling_type=restrict_scaling_type,
                         scaling_min_val=scaling_min_val,
                         max_val=max_val,
                         return_quant_tensor=return_quant_tensor,
                         per_channel_broadcastable_shape=per_channel_broadcastable_shape)



def make_quant_hard_tanh(bit_width,
                         quant_type=QUANT_TYPE,
                         scaling_impl_type=ACT_SCALING_IMPL_TYPE,
                         scaling_per_channel=ACT_SCALING_PER_CHANNEL,
                         restrict_scaling_type=ACT_SCALING_RESTRICT_SCALING_TYPE,
                         scaling_min_val=SCALING_MIN_VAL,
                         threshold=HARD_TANH_THRESHOLD,
                         return_quant_tensor=ACT_RETURN_QUANT_TENSOR,
                         per_channel_broadcastable_shape=ACT_PER_CHANNEL_BROADCASTABLE_SHAPE):
    return qnn.QuantHardTanh(bit_width=bit_width,
                             quant_type=quant_type,
                             scaling_per_channel=scaling_per_channel,
                             scaling_impl_type=scaling_impl_type,
                             restrict_scaling_type=restrict_scaling_type,
                             scaling_min_val=scaling_min_val,
                             max_val=threshold,
                             min_val=-threshold,
                             per_channel_broadcastable_shape=per_channel_broadcastable_shape,
                             return_quant_tensor=return_quant_tensor)


def make_activation(bit_width, type):
  if type=='relu':
    return make_quant_relu(bit_width, quant_type=QuantType.INT) if bit_width!=1 else make_quant_hard_tanh(bit_width, quant_type=QuantType.BINARY)
  elif type=='hardtanh':
    return make_quant_hard_tanh(bit_width, quant_type=QuantType.BINARY) if bit_width==1 else make_quant_hard_tanh(bit_width, quant_type=QuantType.INT)
  else:
    raise NotImplementedError

