# Copyright 2023 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DSLX module example

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "PROC_WRAPPER_TYPE",
    "cc_xls_ir_jit_wrapper",
    "xls_dslx_ir",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_test",
    "xls_ir_verilog",
)

package(
    default_applicable_licenses = ["//:license"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],
)

xls_dslx_library(
    name = "capitalize_module",
    srcs = [
        "capitalize.x",
        "some_caps.x",
        "some_caps_streaming.x",
    ],
)

xls_dslx_test(
    name = "capitalize_tests",
    dslx_test_args = {"compare": "jit"},
    library = ":capitalize_module",
)

filegroup(
    name = "x_files",
    srcs = glob(["*.x"]),
    visibility = ["//visibility:private"],
)

xls_dslx_library(
    name = "some_caps_streaming_configured",
    srcs = ["some_caps_streaming_configured.x"],
    deps = [":capitalize_module"],
)

xls_dslx_ir(
    name = "some_caps_streaming_configured_ir",
    dslx_top = "some_caps_specialized",
    ir_conv_args = {
        # Set fifo config for multi-proc codegen.
        # It needs to know what configuration to use for the fifo.
        "default_fifo_config": "depth: 4, bypass: true, " +
                               "register_push_outputs: false, register_pop_outputs: false",
    },
    ir_file = "some_caps_streaming_configured.ir",
    library = ":some_caps_streaming_configured",
)

cc_xls_ir_jit_wrapper(
    name = "some_caps_jit_wrapper",
    src = ":some_caps_streaming_configured.ir",
    jit_wrapper_args = {
        "class_name": "SomeCaps",
        "namespace": "xls::examples",
    },
    visibility = ["//xls:xls_internal"],
    wrapper_type = PROC_WRAPPER_TYPE,
)

xls_dslx_opt_ir(
    name = "manaul_chan_caps_streaming_configured_opt_ir",
    # Note: The optimized ir has different top since the channels are used to determine proc
    # liveness instead of spawn tree. This is done to avoid having to deal with mangled names.
    dslx_top = "manual_chan_caps_specialized",
    ir_conv_args = {
        # Set fifo config for multi-proc codegen.
        # It needs to know what configuration to use for the fifo.
        "default_fifo_config": "depth: 4, bypass: true, " +
                               "register_push_outputs: false, register_pop_outputs: false",
    },
    ir_file = "manual_chan_caps_streaming_configured.ir",
    library = ":some_caps_streaming_configured",
    visibility = ["//xls:xls_internal"],
)

xls_ir_verilog(
    name = "manual_chan_caps_streaming_configured_multiproc_verilog",
    src = ":manaul_chan_caps_streaming_configured_opt_ir",
    block_ir_file = "manual_chan_caps_streaming_configured_multiproc.block.ir",
    codegen_args = {
        "module_name": "manual_chan_caps_streaming",
        "generator": "pipeline",
        "pipeline_stages": "4",
        "delay_model": "unit",
        "reset": "rst",
        "reset_data_path": "false",
        "reset_active_low": "false",
        "reset_asynchronous": "false",
        "flop_inputs": "false",
        "flop_single_value_channels": "false",
        "flop_outputs": "false",
        "add_idle_output": "false",
        "streaming_channel_data_suffix": "_data",
        "streaming_channel_ready_suffix": "_ready",
        "streaming_channel_valid_suffix": "_valid",
        "use_system_verilog": "true",
        "assert_format": "\\;",
        "multi_proc": "true",
    },
    module_sig_file = "manual_chan_caps_streaming_configured_multiproc.sig.textproto",
    verilog_file = "manual_chan_caps_streaming_configured_multiproc.sv",
    visibility = ["//xls:xls_internal"],
)

cc_xls_ir_jit_wrapper(
    name = "some_caps_opt_jit_wrapper",
    src = ":manaul_chan_caps_streaming_configured_opt_ir",
    jit_wrapper_args = {
        "class_name": "SomeCapsOpt",
        "namespace": "xls::examples",
    },
    visibility = ["//xls:xls_internal"],
    wrapper_type = PROC_WRAPPER_TYPE,
)
