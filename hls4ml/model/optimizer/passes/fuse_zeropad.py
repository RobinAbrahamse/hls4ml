from hls4ml.model.optimizer import OptimizerPass

from hls4ml.model import hls_model

class FuseZeroPad1D(OptimizerPass):
    ''' Fuses ZeroPadding1D into Pooling/Conv layer (original usage for OneAPI support). '''
    def match(self, node):
        is_match = node.get_input_node().__class__.__name__ == 'ZeroPadding1D' and \
            (node.__class__.__name__ == 'Conv1D' or \
            node.__class__.__name__ == 'SeparableConv1D' or \
            node.__class__.__name__ == 'Pooling1D')
        return is_match

    def transform(self, model, node):
        zeropad_node = node.get_input_node()
        node.set_attr('pad_left', node.get_attr('pad_left') + zeropad_node.get_attr('pad_left'))
        node.set_attr('pad_right', node.get_attr('pad_right') + zeropad_node.get_attr('pad_right'))

        model.remove_node(zeropad_node, rewire=True)

        return True

class FuseZeroPad2D(OptimizerPass):
    ''' Fuses ZeroPadding2D into Pooling/Conv layer (original usage for OneAPI support). '''
    def match(self, node):
        is_match = node.get_input_node().__class__.__name__ == 'ZeroPadding2D' and \
            (node.__class__.__name__ == 'Conv2D' or \
            node.__class__.__name__ == 'SeparableConv2D' or \
            node.__class__.__name__ == 'DepthwiseConv2D' or \
            node.__class__.__name__ == 'Pooling2D')
        return is_match

    def transform(self, model, node):
        zeropad_node = node.get_input_node()
        node.set_attr('pad_top', node.get_attr('pad_top') + zeropad_node.get_attr('pad_top'))
        node.set_attr('pad_bottom', node.get_attr('pad_bottom') + zeropad_node.get_attr('pad_bottom'))
        node.set_attr('pad_left', node.get_attr('pad_left') + zeropad_node.get_attr('pad_left'))
        node.set_attr('pad_right', node.get_attr('pad_right') + zeropad_node.get_attr('pad_right'))

        model.remove_node(zeropad_node, rewire=True)

        return True
