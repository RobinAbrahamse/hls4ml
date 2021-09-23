from hls4ml.templates.templates import Backend

input_config_template = """
    dnnl::memory::dims input_data_dims = {{{dims}}};
    input_data_memory = dnnl::memory({{
            {{input_data_dims}},
            dnnl::memory::data_type::{data_type},
            dnnl::memory::format_tag::{format_tag}}},
            eng);

    auto input_data_md = dnnl::memory::desc({{
            {{input_data_dims}},
            dnnl::memory::data_type::{data_type},
            dnnl::memory::format_tag::any}});\n"""

memory_config_template = """
    dnnl::memory::dims {layer_name}_{memory_object}_dims = {{{dims}}};
    {memory_object_type} {layer_name}_{memory_object}_{placeholder}memory = dnnl::memory({{
            {{{layer_name}_{memory_object}_dims}},
            dnnl::memory::data_type::{data_type},
            dnnl::memory::format_tag::{format_tag}}},
            eng);

    auto {layer_name}_{memory_object}_md = dnnl::memory::desc({{
            {{{layer_name}_{memory_object}_dims}},
            dnnl::memory::data_type::{data_type},
            dnnl::memory::format_tag::any}});\n"""

dense_config_template = """
    // Dense layer
    dnnl::memory::dims {layer_name}_output_dims = {output_dims};
    auto {layer_name}_output_md = dnnl::memory::desc({{
            {{{layer_name}_output_dims}},
            dnnl::memory::data_type::{data_type},
            dnnl::memory::format_tag::any}});

    auto {layer_name}_desc = dnnl::inner_product_forward::desc(
            dnnl::prop_kind::forward_inference,
            {input_desc}, {layer_name}_weights_md, {layer_name}_bias_md, {layer_name}_output_md);

    auto {layer_name}_prim_desc = dnnl::inner_product_forward::primitive_desc({layer_name}_desc, eng);

    write_to_dnnl_memory({layer_name}_weights_buffer.data(), {layer_name}_weights_placeholder_memory);
    write_to_dnnl_memory({layer_name}_bias_buffer.data(), {layer_name}_bias_memory);

    auto {layer_name}_weights_memory = {layer_name}_weights_placeholder_memory;
    if ({layer_name}_prim_desc.weights_desc() != {layer_name}_weights_placeholder_memory.get_desc()) {{
        {layer_name}_weights_memory = dnnl::memory({layer_name}_prim_desc.weights_desc(), eng);
        dnnl::reorder({layer_name}_weights_placeholder_memory, {layer_name}_weights_memory).execute(
                engine_stream, {layer_name}_weights_placeholder_memory, {layer_name}_weights_memory);
    }}
    auto {output_memory} = dnnl::memory({layer_name}_prim_desc.dst_desc(), eng);

    net.push_back(dnnl::inner_product_forward({layer_name}_prim_desc));
    net_args.push_back({{{{DNNL_ARG_SRC, {input_memory}}},
            {{DNNL_ARG_WEIGHTS, {layer_name}_weights_memory}},
            {{DNNL_ARG_BIAS, {layer_name}_bias_memory}},
            {{DNNL_ARG_DST, {output_memory}}}}});
    //\n"""

conv_config_template = """
    // Convolution layer
    dnnl::memory::dims {layer_name}_output_dims = {output_dims};
    dnnl::memory::dims {layer_name}_strides = {strides};
    dnnl::memory::dims {layer_name}_padding_l = {padding_l};
    dnnl::memory::dims {layer_name}_padding_r = {padding_r};

    auto {layer_name}_output_md = dnnl::memory::desc({{
            {{{layer_name}_output_dims}},
            dnnl::memory::data_type::{data_type},
            dnnl::memory::format_tag::any}});

    auto {layer_name}_desc = dnnl::convolution_forward::desc(
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct, {input_desc}, {layer_name}_weights_md,
            {layer_name}_bias_md, {layer_name}_output_md, {layer_name}_strides, 
            {layer_name}_padding_l, {layer_name}_padding_r);

    auto {layer_name}_prim_desc = dnnl::convolution_forward::primitive_desc({layer_name}_desc, eng);

    write_to_dnnl_memory({layer_name}_weights_buffer.data(), {layer_name}_weights_placeholder_memory);
    write_to_dnnl_memory({layer_name}_bias_buffer.data(), {layer_name}_bias_memory);

    auto {layer_name}_weights_memory = {layer_name}_weights_placeholder_memory;
    if ({layer_name}_prim_desc.weights_desc() != {layer_name}_weights_placeholder_memory.get_desc()) {{
        {layer_name}_weights_memory = dnnl::memory({layer_name}_prim_desc.weights_desc(), eng);
        dnnl::reorder({layer_name}_weights_placeholder_memory, {layer_name}_weights_memory).execute(
                engine_stream, {layer_name}_weights_placeholder_memory, {layer_name}_weights_memory);
    }}

    auto {output_memory} = dnnl::memory({layer_name}_prim_desc.dst_desc(), eng);

    net.push_back(dnnl::convolution_forward({layer_name}_prim_desc));
    net_args.push_back({{{{DNNL_ARG_SRC, {input_memory}}},
            {{DNNL_ARG_WEIGHTS, {layer_name}_weights_memory}},
            {{DNNL_ARG_BIAS, {layer_name}_bias_memory}},
            {{DNNL_ARG_DST, {output_memory}}}}});
    //\n"""

pooling_config_template = """
    // Pooling layer
    dnnl::memory::dims {layer_name}_output_dims = {output_dims};
    dnnl::memory::dims {layer_name}_kernel = {kernel};
    dnnl::memory::dims {layer_name}_strides = {strides};
    dnnl::memory::dims {layer_name}_padding_l = {padding_l};
    dnnl::memory::dims {layer_name}_padding_r = {padding_r};

    auto {layer_name}_output_md = dnnl::memory::desc({{
            {{{layer_name}_output_dims}},
            {input_desc}.data_type(),
            dnnl::memory::format_tag::any}});

    auto {layer_name}_desc = dnnl::pooling_forward::desc(
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::pooling_{pool_op}, {input_desc}, {layer_name}_output_md,
            {layer_name}_strides, {layer_name}_kernel, 
            {layer_name}_padding_l, {layer_name}_padding_r);

    auto {layer_name}_prim_desc = dnnl::pooling_forward::primitive_desc({layer_name}_desc, eng);

    auto {output_memory} = dnnl::memory({layer_name}_prim_desc.dst_desc(), eng);

    net.push_back(dnnl::pooling_forward({layer_name}_prim_desc));
    net_args.push_back({{{{DNNL_ARG_SRC, {input_memory}}},
            {{DNNL_ARG_DST, {output_memory}}}}});
    //\n"""

addsub_config_template = """
    // Add/Subtract layer
    auto {layer_name}_prim_desc = dnnl::sum::primitive_desc({scales}, {input_descs}, eng);

    auto {output_memory} = dnnl::memory({layer_name}_prim_desc.dst_desc(), eng);

    net.push_back(dnnl::sum({layer_name}_prim_desc));
    net_args.push_back({{{input_args},
            {{DNNL_ARG_DST, {output_memory}}}}});
    //\n"""

eltwise_config_template = """
    // {type} activation layer
    auto {layer_name}_desc = dnnl::eltwise_forward::desc(
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::eltwise_{type}, {input_desc},
            {alpha});

    auto {layer_name}_prim_desc = dnnl::eltwise_forward::primitive_desc({layer_name}_desc, eng);
    {init_memory}
    net.push_back(dnnl::eltwise_forward({layer_name}_prim_desc));
    net_args.push_back({{{{DNNL_ARG_SRC, {input_memory}}},
            {{DNNL_ARG_DST, {output_memory}}}}});
    //\n"""

softmax_config_template = """
    // Softmax activation layer
    auto {layer_name}_desc = dnnl::softmax_forward::desc(dnnl::prop_kind::forward_inference,
            {input_desc}, {axis});

    auto {layer_name}_prim_desc = dnnl::softmax_forward::primitive_desc({layer_name}_desc, eng);
    {init_memory}
    net.push_back(dnnl::softmax_forward({layer_name}_prim_desc));
    net_args.push_back({{{{DNNL_ARG_SRC, {input_memory}}},
            {{DNNL_ARG_DST, {output_memory}}}}});
    //\n"""

batchnormalization_config_template = """
    // Batch Normalization layer
    auto {layer_name}_desc = dnnl::batch_normalization_forward::desc(
            dnnl::prop_kind::forward_inference,
            {input_desc},
            {epsilon}, dnnl::normalization_flags::use_scale_shift);

    auto {layer_name}_prim_desc = dnnl::batch_normalization_forward::primitive_desc({layer_name}_desc, eng);

    write_to_dnnl_memory({layer_name}_scale_buffer.data(), {layer_name}_scale_memory);
    write_to_dnnl_memory({layer_name}_bias_buffer.data(), {layer_name}_bias_memory);
    {init_memory}
    net.push_back(dnnl::batch_normalization_forward({layer_name}_prim_desc));
    net_args.push_back({{{{DNNL_ARG_SRC, {input_memory}}},
            {{DNNL_ARG_SCALE_SHIFT, {layer_name}_scale_memory}},
            {{DNNL_ARG_SCALE_SHIFT + 1, {layer_name}_bias_memory}},
            {{DNNL_ARG_DST, {output_memory}}}}});
    //\n"""

class OneAPI(Backend):
    def __init__(self):
        super(OneAPI, self).__init__('oneAPI')
        self.register_config_template('Input', input_config_template)
        self.register_config_template('Memory', memory_config_template)
        self.register_config_template('Dense', dense_config_template)
        self.register_config_template('Merge', addsub_config_template)
        self.register_config_template('Activation', eltwise_config_template)
        self.register_config_template('Softmax', softmax_config_template)
        self.register_config_template('BatchNormalization', batchnormalization_config_template)
        self.register_config_template('Conv1D', conv_config_template)
        self.register_config_template('SeparableConv1D', conv_config_template)
        self.register_config_template('Conv2D', conv_config_template)
        self.register_config_template('SeparableConv2D', conv_config_template)
        self.register_config_template('DepthwiseConv2D', conv_config_template)
        self.register_config_template('Pooling2D', pooling_config_template)
        self.register_config_template('Pooling1D', pooling_config_template)
        self.register_config_template('GlobalPooling2D', pooling_config_template)
        self.register_config_template('GlobalPooling1D', pooling_config_template)

    def register_config_template(self, name, config_template):
        self.config_templates[name] = config_template
