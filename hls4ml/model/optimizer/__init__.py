from __future__ import absolute_import

from hls4ml.model.optimizer.optimizer import OptimizerPass, register_pass, get_optimizer, optimize_model, get_available_passes


from hls4ml.model.optimizer.passes.nop import EliminateLinearActivation
from hls4ml.model.optimizer.passes.bn_quant import MergeBatchNormAndQuantizedTanh, QuantizeDenseOutput
from hls4ml.model.optimizer.passes.dense_bn_fuse import FuseDenseAndBatchNormalization
from hls4ml.model.optimizer.passes.fuse_biasadd import FuseBiasAdd
from hls4ml.model.optimizer.passes.fuse_zeropad import FuseZeroPad1D, FuseZeroPad2D
from hls4ml.model.optimizer.passes.check_nonsequential import CheckNonSequential
try:
    from hls4ml.model.optimizer.passes.qkeras import OutputRoundingSaturationMode
    from hls4ml.model.optimizer.passes.qkeras import QKerasFactorizeAlpha
    from hls4ml.model.optimizer.passes.qkeras import FuseConsecutiveBatchNormalization
    __qkeras_optimizers__ = True
except ImportError:
    __qkeras_optimizers__ = False

register_pass('eliminate_linear_activation', EliminateLinearActivation)
register_pass('merge_batch_norm_quantized_tanh', MergeBatchNormAndQuantizedTanh)
register_pass('quantize_dense_output', QuantizeDenseOutput)
register_pass('fuse_dense_batch_norm', FuseDenseAndBatchNormalization)
register_pass('fuse_biasadd', FuseBiasAdd)
register_pass('fuse_zeropad1d', FuseZeroPad1D)
register_pass('fuse_zeropad2d', FuseZeroPad2D)
register_pass('check_nonsequential', CheckNonSequential)
if __qkeras_optimizers__:
    register_pass('output_rounding_saturation_mode', OutputRoundingSaturationMode)
    register_pass('qkeras_factorize_alpha', QKerasFactorizeAlpha)
    register_pass('fuse_consecutive_batch_normalization', FuseConsecutiveBatchNormalization) 

