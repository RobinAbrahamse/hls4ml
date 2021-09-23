from shutil import copyfile, copytree, rmtree
import numpy as np
import os
import re
import glob
from collections import OrderedDict

from hls4ml.writer.writers import Writer


oneapi_data_types_map_to_cpp = {
    "f32": "float",
    "b16": "float",
    "s8": "signed short",
    "u8": "unsigned short"
}

weights_names_map = {
    "a": "alpha",
    "b": "bias",
    "d": "depthwise_weights",
    "p": "pointwise_weights",
    "s": "scale",
    "w": "weights",
    "z": "zero_bias"
}

class OneApiWriter(Writer):
    def save_weights_to_file(self, array, odir, write_txt_file=True):
        h_file = open("{}/firmware/weights/{}.h".format(odir, array.name),"w")
        if write_txt_file:
            txt_file = open("{}/firmware/weights/{}.txt".format(odir, array.name),"w")

        #meta data
        h_file.write("//Numpy array shape {}\n".format(array.shape))
        h_file.write("//Min {:.12f}\n".format(np.min(array.min)))
        h_file.write("//Max {:.12f}\n".format(np.max(array.max)))
        h_file.write("//Number of zeros {}\n".format(array.nzeros))
        h_file.write("\n")

        h_file.write("#ifndef {}_H_\n".format(array.name.upper()))
        h_file.write("#define {}_H_\n".format(array.name.upper()))
        h_file.write("\n")

        if write_txt_file:
            h_file.write("#ifndef __SYNTHESIS__\n")
            h_file.write(array.definition_cpp() + ";\n")
            h_file.write("#else\n")

        h_file.write(array.definition_cpp() + " = {")

        #fill c++ array.
        #not including internal brackets for multidimensional case
        txt = ", ".join(array)
        h_file.write(txt)
        if write_txt_file:
            txt_file.write(txt)
        h_file.write("};\n")
        if write_txt_file:
            h_file.write("#endif\n")
            txt_file.close()
        h_file.write("\n#endif\n")
        h_file.close()

    def write_project_dir(self, model):
        if not os.path.isdir("{}/firmware/weights".format(model.config.get_output_dir())):
            os.makedirs("{}/firmware/weights".format(model.config.get_output_dir()))

    def write_project_cpp(self, model):
        """ Writes main function for the project """
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir,'../templates/oneapi/firmware/myproject.cpp'),'r')
        fout = open('{}/firmware/{}.cpp'.format(model.config.get_output_dir(), model.config.get_project_name()),'w')

        indent = '    '
        for line in f.readlines():
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())
            elif '//hls4ml init engine' in line:
                newline = line
                newline += f"dnnl::engine eng(dnnl::engine::kind::{model.config.device}, 0);\n"
            elif '//hls4ml insert layers' in line:
                newline = ''
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        data_type = oneapi_data_types_map_to_cpp[w.type.precision]
                        weights_type = weights_names_map.get(w.name[0])
                        buffer_name = f'{layer.name}_{weights_type}_buffer'
                        if w.__class__.__name__ == 'CompressedWeightVariable':
                            create_buffer = f'std::vector<{data_type}> {buffer_name}({w.nonzeros});\n'
                            load_weights = f'nnet::load_compressed_weights_from_txt<{data_type}, {w.nonzeros}>({buffer_name}.data(), "{w.name}.txt");\n'
                        else:
                            create_buffer = f'std::vector<{data_type}> {buffer_name}({w.data_length});\n'
                            load_weights = f'nnet::load_weights_from_txt<{data_type}, {w.data_length}>({buffer_name}.data(), "{w.name}.txt");\n'
                        newline += indent + create_buffer
                        newline += indent + load_weights
                    dcpp_definition = layer.definition_dpcpp()
                    newline += indent + dcpp_definition + "\n"
                output_memory = f"{model.outputs[0]}_memory"
                if model.sequential: # memory reuse optimization
                    output_layer = model.graph.get(model.outputs[0])
                    if not output_layer.memory_descriptor:
                        output_memory = f"{output_layer.get_input_node_with_mem_desc(output_layer).name}_memory"
                newline += f"{indent}output_data_memory = {output_memory};\n"
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_project_header(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir,'../templates/oneapi/firmware/myproject.h'),'r')
        fout = open('{}/firmware/{}.h'.format(model.config.get_output_dir(), model.config.get_project_name()),'w')

        indent = '    '
        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT',format(model.config.get_project_name().upper()))
            elif 'void myproject(' in line:
                newline = 'void {}(\n'.format(model.config.get_project_name())
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_weights(self, model):
        for layer in model.get_layers():
            for weights in layer.get_weights():
                self.save_weights_to_file(weights, model.config.get_output_dir())

    def write_utils(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        srcpath = os.path.join(filedir,'../templates/oneapi/utils/')
        dstpath = '{}/firmware/utils/'.format(model.config.get_output_dir())
        if os.path.exists(dstpath):
            rmtree(dstpath)
        copytree(srcpath, dstpath)

    def write_build_script(self, model):
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir,'../templates/oneapi/build_lib.sh'),'r')
        fout = open('{}/build_lib.sh'.format(model.config.get_output_dir()),'w')

        for line in f.readlines():
            if "PROJECT" in line:
                line = line.replace('myproject', model.config.get_project_name())
            fout.write(line)
        f.close()
        fout.close()
    
    def write_hls(self, model):
        print("Writing HLS4ML OneAPI project")
        self.write_project_dir(model)
        self.write_project_cpp(model)
        self.write_project_header(model)
        self.write_utils(model)
        self.write_weights(model)
        self.write_build_script(model)
        print("Done")