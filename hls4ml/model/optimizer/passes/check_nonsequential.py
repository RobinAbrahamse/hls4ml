from hls4ml.model.optimizer import OptimizerPass

from hls4ml.model import hls_model

class CheckNonSequential(OptimizerPass):
    ''' Checks whether the model is non-sequential 
    (original usage for OneAPI memory reuse optimization in sequential models). '''
    def match(self, node):
        return node.model.sequential and len(node.inputs) > 1

    def transform(self, model, node):
        model.sequential = False
        return True