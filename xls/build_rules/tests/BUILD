# Copyright 2020 The XLS Authors
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

# @rules_cc//cc:cc_test.bzl
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "cc_xls_ir_jit_wrapper",
    "check_sha256sum_frozen",
    "check_sha256sum_test",
    "get_mangled_ir_symbol",
    "xls_benchmark_ir",
    "xls_dslx_cpp_type_library",
    "xls_dslx_ir",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_opt_ir_test",
    "xls_dslx_test",
    "xls_dslx_verilog",
    "xls_eval_ir_test",
    "xls_ir_cc_library",
    "xls_ir_equivalence_test",
    "xls_ir_opt_ir",
    "xls_ir_verilog",
)
load("//xls/build_rules:xls_cc_embed_data_rules.bzl", "xls_cc_embed_data")
load("//xls/build_rules:xls_internal_aot_rules.bzl", "xls_aot_generate")
load(
    "//xls/build_rules:xls_rules_build_defs.bzl",
    "FUNCTION_WRAPPER_TYPE",
    "xls_ir_jit_wrapper_macro",
    xls_dslx_ir_native_rule = "xls_dslx_ir",
    xls_dslx_opt_ir_native_rule = "xls_dslx_opt_ir",
    xls_dslx_verilog_native_rule = "xls_dslx_verilog",
    xls_ir_jit_wrapper_native_rule = "xls_ir_jit_wrapper",
    xls_ir_opt_ir_native_rule = "xls_ir_opt_ir",
    xls_ir_verilog_native_rule = "xls_ir_verilog",
)
load(
    "//xls/build_rules/tests:test_build_defs.bzl",
    "codegen_args_proto",
    "scheduling_args_proto",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# Ensure that we can take a dep on a genrule .x file and import it.
genrule(
    name = "generated_x",
    outs = ["generated.x"],
    cmd = "echo 'pub const VALUE = u32:42;' > $(OUTS) && " +
          "echo 'fn main() -> u32 { VALUE }' >> $(OUTS) && " +
          "echo '#[test]' >> $(OUTS) && " +
          "echo 'fn test_value() { assert_eq(VALUE, u32:42) }' >> $(OUTS)",
)

# Test with a checksum file generated with a genrule.
check_sha256sum_test(
    name = "generated_x_sha256sum_test",
    src = ":generated.x",
    sha256sum = "cb04ccd886a4191d5ed6ddb3ac213b9d4eb9144b6d52880544d72ae6f055e432",
)

# Build a frozen file generated with a genrule.
check_sha256sum_frozen(
    name = "generated_x_sha256sum_frozen",
    src = ":generated.x",
    frozen_file = "generated.frozen.x",
    sha256sum = "cb04ccd886a4191d5ed6ddb3ac213b9d4eb9144b6d52880544d72ae6f055e432",
)

# Test with a frozen checksum file generated with a check_sha256sum_frozen rule.
check_sha256sum_test(
    name = "generated_frozen_x_sha256sum_test",
    src = ":generated.frozen.x",
    sha256sum = "cb04ccd886a4191d5ed6ddb3ac213b9d4eb9144b6d52880544d72ae6f055e432",
)

# Check that we can generate a block a couple different ways.
xls_dslx_verilog(
    name = "add_one_pipeline_rtl",
    srcs = ["add_one.x"],
    codegen_args = {
        "pipeline_stages": "2",
        "reset_data_path": "false",
        "delay_model": "unit",
    },
    dslx_top = "main",
    verilog_file = "add_one_pipeline_rtl.sv",
)

codegen_args_proto(
    name = "add_one_pipeline_rtl_proto_codegen_proto",
    codegen_options_proto = "add_one_pipeline_rtl_codegen.txtproto",
    is_pipelined = True,
    module_name = "main",
    reset_data_path = False,
    top = "__add_one__main",
)

scheduling_args_proto(
    name = "add_one_pipeline_rtl_proto_scheduling_proto",
    delay_model = "unit",
    pipeline_stages = 2,
    scheduling_options_proto = "add_one_pipeline_rtl_scheduling.txtproto",
)

xls_dslx_verilog(
    name = "add_one_pipeline_rtl_proto",
    srcs = ["add_one.x"],
    codegen_options_proto = ":add_one_pipeline_rtl_codegen.txtproto",
    dslx_top = "main",
    scheduling_options_proto = ":add_one_pipeline_rtl_scheduling.txtproto",
    verilog_file = "add_one_pipeline_rtl_proto.sv",
)

xls_dslx_verilog(
    name = "add_one_pipeline_rtl_proto_fail",
    srcs = ["add_one.x"],
    codegen_args = {
        "clock_period_ps": "42",
    },
    codegen_options_proto = ":add_one_pipeline_rtl_codegen.txtproto",
    dslx_top = "main",
    # uncomment the following line to test that passing both scheduling options and args will crash.
    # scheduling_options_proto = ":add_one_pipeline_rtl_scheduling.txtproto",
    verilog_file = "add_one_pipeline_rtl_proto_fail.sv",
)

# Test a single file when many are generated.
check_sha256sum_test(
    name = "add_one_pipeline_rtl_v_sha256sum_test",
    src = ":add_one_pipeline_rtl.sv",
    sha256sum = "7230178ff270fa174805ff808bbe927c19a13961fc4e957e59aa434569e36caf",
)

# Build a frozen file generated with a build rule.
check_sha256sum_frozen(
    name = "add_one_pipeline_rtl_v_sha256sum_frozen",
    src = ":add_one_pipeline_rtl.sv",
    frozen_file = ":add_one_pipeline_rtl.frozen.v",
    sha256sum = "7230178ff270fa174805ff808bbe927c19a13961fc4e957e59aa434569e36caf",
)

# Test with a frozen checksum file generated with a check_sha256sum_frozen rule.
check_sha256sum_test(
    name = "add_one_pipeline_rtl_frozen_v_sha256sum_test",
    src = ":add_one_pipeline_rtl.frozen.v",
    sha256sum = "7230178ff270fa174805ff808bbe927c19a13961fc4e957e59aa434569e36caf",
)

sh_test(
    name = "add_one_pipeline_rtl_has_clk_test",
    srcs = ["has_clk.sh"],
    args = ["$(rootpath :add_one_pipeline_rtl.sv)"],
    data = [":add_one_pipeline_rtl.sv"],
)

sh_test(
    name = "add_one_pipeline_rtl_proto_has_clk_test",
    srcs = ["has_clk.sh"],
    args = ["$(rootpath :add_one_pipeline_rtl_proto.sv)"],
    data = [":add_one_pipeline_rtl_proto.sv"],
)

xls_dslx_verilog(
    name = "add_one_combinational_rtl",
    srcs = ["add_one.x"],
    codegen_args = {
        "generator": "combinational",
    },
    dslx_top = "main",
    verilog_file = "add_one_combinational_rtl.sv",
)

sh_test(
    name = "add_one_combinational_rtl_has_clk_test",
    srcs = ["not_has_clk.sh"],
    args = ["$(rootpath :add_one_combinational_rtl.sv)"],
    data = [":add_one_combinational_rtl.sv"],
)

sh_test(
    name = "add_one_combinational_rtl_block_ir_test",
    srcs = ["is_block_ir.sh"],
    args = ["$(rootpath :add_one_combinational_rtl.block.ir)"],
    data = [":add_one_combinational_rtl.block.ir"],
)

# Run the test inside of the generated file.
xls_dslx_test(
    name = "generated_file_as_input_test",
    srcs = ["generated.x"],
    dslx_test_args = {"compare": "jit"},
)

# Encapsulate the generated file as a library.
xls_dslx_library(
    name = "generated_dslx",
    srcs = ["generated.x"],
)

xls_dslx_library(
    name = "importer_dslx",
    srcs = ["importer.x"],
    deps = [
        ":generated_dslx",
    ],
)

# Reference the library with the generated file inside of a xls_dslx_test rule.
xls_dslx_test(
    name = "dependent_on_library_with_generated_file_test",
    dslx_test_args = {"compare": "jit"},
    library = ":importer_dslx",
)

xls_dslx_library(
    name = "a_dslx",
    srcs = [
        "a.x",
    ],
)

# xls_dslx_library target depends on xls_dslx_library target 'a_dslx'.
xls_dslx_library(
    name = "bc_dslx",
    srcs = [
        "b.x",
        "c.x",
    ],
    deps = [
        ":a_dslx",
    ],
)

# xls_dslx_library with a transitive dependency.
xls_dslx_library(
    name = "d_dslx",
    srcs = ["d.x"],
    deps = [
        ":bc_dslx",
    ],
)

# xls_dslx_test on a xls_dslx_library with a transitive dependency.
xls_dslx_test(
    name = "d_dslx_test",
    dslx_test_args = {"compare": "jit"},
    library = ":d_dslx",
)

# xls_dslx_library with a transitive dependency on a
# xls_dslx_library.
xls_dslx_library(
    name = "e_dslx",
    srcs = ["e.x"],
    deps = [
        ":d_dslx",
    ],
)

# xls_dslx_test on a xls_dslx_library with a transitive dependency.
xls_dslx_test(
    name = "e_dslx_test",
    dslx_test_args = {"compare": "jit"},
    library = ":e_dslx",
)

################################################################################
# Build rules using dslx_library as input.
################################################################################
xls_dslx_library(
    name = "add_one_dslx",
    srcs = ["add_one.x"],
)

xls_dslx_ir(
    name = "add_one_dslx_ir",
    dslx_top = "main",
    library = ":add_one_dslx",
)

xls_dslx_opt_ir(
    name = "add_one_dslx_opt_ir",
    dslx_top = "main",
    library = ":add_one_dslx",
)

xls_dslx_verilog(
    name = "add_one_dslx_verilog",
    codegen_args = {
        "generator": "combinational",
    },
    dslx_top = "main",
    library = ":add_one_dslx",
    verilog_file = "add_one_dslx_verilog.sv",
)

################################################################################
# Simple example with rule chaining.
#
# Below is a diagram illustrating the rule chaining example. The build flow
# is on the left and the test flow is on the right. The flow is from top to
# down. The target names of the rules and the name of the input file are
# displayed.
#
# Overview:
# The 'chaining_example.x' file is the input to the build and test flow.
#
# In the build flow, the 'chaining_example.x' file is the input to a
# xls_dslx_ir rule named 'chaining_example_ir'. The generated IR file of the
# latter target is the input to a xls_ir_opt_ir rule named
# 'chaining_example_opt_ir'. The optimized IR file of the latter target is the
# input to a xls_ir_verilog rule named 'chaining_example_verilog'.
#
# In the test flow, the 'chaining_example.x' file is the input to a
# xls_dslx_test rule named 'chaining_example_test'. The generated IR file of the
# 'chaining_example_ir' target is the input to a xls_eval_ir_test rule named
# 'chaining_example_eval_ir_test', and the input to a xls_ir_equivalence_test
# rule named 'chaining_example_ir_equivalence_test'. The optimized IR file of
# the 'chaining_example_opt_ir' target is the input to a xls_benchmark_ir rule
# named 'chaining_example_benchmark_ir', and the input to a
# xls_ir_equivalence_test rule named 'chaining_example_ir_equivalence_test'.
#
#           Build Flow                  Test/Execution Flow
#           ----------                  -------------------
#
#                     chaining_example.x
#                             |
#               ---------------------------------
#              |                                |
#              |                      chaining_example_test
#              |
#              |               |--> chaining_example_eval_ir_test
#              |               |
#      chaining_example_ir  ---|
#              |               |
#              |               |----------------------------------------------|
#              |                                                              |
#              |                  |--> chaining_example_ir_equivalence_test <-|
#              |                  |
#     chaining_example_opt_ir  ---|
#              |                  |
#    chaining_example_verilog     |--> chaining_example_benchmark_ir
################################################################################

xls_dslx_test(
    name = "chaining_example_test",
    srcs = ["chaining_example.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_ir(
    name = "chaining_example_ir",
    srcs = ["chaining_example.x"],
    dslx_top = "main",
)

xls_ir_opt_ir(
    name = "chaining_example_opt_ir",
    src = ":chaining_example_ir",
    opt_ir_file = "chaining_example_opt_ir.opt.ir",
)

xls_ir_verilog(
    name = "chaining_example_verilog",
    src = ":chaining_example_opt_ir",
    codegen_args = {
        "pipeline_stages": "1",
    },
    verilog_file = "chaining_example_verilog.sv",
)

xls_eval_ir_test(
    name = "chaining_example_eval_ir_test",
    src = ":chaining_example_ir",
)

xls_ir_equivalence_test(
    name = "chaining_example_ir_equivalence_test",
    src_0 = ":chaining_example_ir",
    src_1 = ":chaining_example_opt_ir",
)

xls_benchmark_ir(
    name = "chaining_example_benchmark_ir",
    src = ":chaining_example_opt_ir",
)

################################################################################
# Extending the simple example with rule chaining.
#
# Below is a diagram illustrating an extension to the build flow of the rule
# chaining example. The names of the rules target names and outputs of the
# targets are displayed. The flow is from top to down.
#
# Overview:
# The 'chaining_example_opt_ir.opt.ir' file, the output from the
# 'chaining_example_opt_ir' target from the simple example with chaining, is the
# input to the cc_xls_ir_jit_wrapper macro named
# 'chaining_example_opt_ir_jit_wrapper_main_cc'.
#
# The generated IR file from the 'chaining_example_ir' target is the input to
# two xls_ir_opt_ir rules named 'chaining_example_double_opt_ir' and
# 'chaining_example_triple_run_only_opt_ir'. The optimized IR file from the
# chaining_example_double_opt_ir target is the input to a
# cc_xls_ir_jit_wrapper macro named
# 'chaining_example_opt_ir_jit_wrapper_double_cc'. Note the reuse of the
# generated IR file from the 'chaining_example_ir' target as input to two
# xls_ir_opt_ir rules using different top entities.
#
# The optimized IR file from the 'chaining_example_triple_run_only_opt_ir'
# target is the input to a xls_ir_opt_ir rule named
# 'chaining_example_triple_skip_passes_opt_ir'. Note the chaining of two
# xls_ir_opt_ir rules: 'chaining_example_triple_run_only_opt_ir' and
# 'chaining_example_triple_skip_passes_opt_ir'.
#
#           Build Flow
#           ----------
#  chaining_example_opt_ir.opt.ir (output from chaining_example_opt_ir target -
#               |                  See the simple example with chaining the
#               |                  rules above.)
#               |
#               |------------->  chaining_example_opt_ir_jit_wrapper_main_cc
#
#
#
#  chaining_example_ir (Target from the simple example with chaining the rules
#               |       shown above.)
#               |
#               |------------->  chaining_example_double_opt_ir
#               |           (Optimizes at the 'double' entry point)
#               |                              |
#               |                              |
#               |                              |
#               |        chaining_example_opt_ir_jit_wrapper_double_cc
#               |
#               |
#               |------------->  chaining_example_triple_run_only_opt_ir
#                                (Optimizes at the 'triple' entry point
#                                          with selected passes.)
#                                                  |
#                                                  |
#                                                  |
#                                chaining_example_triple_skip_passes_opt_ir
#                                (Optimizes by skipping selected passes.)
#
# Below is a diagram illustrating the test flow of an extension to the rule
# chaining example. The names of the rules target names and outputs of the
# targets are displayed. The flow is from top to down.
#
# Overview:
# The optimized IR file from the 'chaining_example_triple_run_only_opt_ir'
# target and the optimized IR file from the
# 'chaining_example_triple_skip_passes_opt_ir' are the inputs to an
# xls_ir_equivalence_test rule named
# 'chaining_example_triple_skip_passes_opt_ir'.
#
#           Test Flow
#           ----------
#    chaining_example_triple_run_only_opt_ir.opt.ir (output from
#               |                  chaining_example_triple_run_only_opt_ir
#               |                  target)
#               |
#               |--> chaining_example_triple_ir_equivalence_test <--|
#                                                                   |
#                                                                   |
#              chaining_example_triple_skip_passes_opt_ir.opt.ir----|
#            (output from chaining_example_triple_skip_passes_opt_ir target)
################################################################################

cc_xls_ir_jit_wrapper(
    name = "chaining_example_opt_ir_jit_wrapper_main_cc",
    src = ":chaining_example_opt_ir",
    jit_wrapper_args = {
        "namespace": "not_xls::test",
    },
)

xls_ir_opt_ir(
    name = "chaining_example_double_opt_ir",
    src = ":chaining_example_ir",
    opt_ir_file = "chaining_example_double_opt_ir.opt.ir",
    top = get_mangled_ir_symbol("chaining_example", "double"),
)

cc_xls_ir_jit_wrapper(
    name = "chaining_example_opt_ir_jit_wrapper_double_cc",
    src = ":chaining_example_double_opt_ir",
    jit_wrapper_args = {
        "class_name": "ChainingExample",
        "function": get_mangled_ir_symbol("chaining_example", "double"),
        "namespace": "not_xls::test",
    },
)

xls_ir_opt_ir(
    name = "chaining_example_triple_run_only_opt_ir",
    src = ":chaining_example_ir",
    opt_ir_args = {
        "passes": "arith_simp",
    },
    opt_ir_file = "chaining_example_triple_run_only_opt_ir.opt.ir",
    top = get_mangled_ir_symbol("chaining_example", "triple"),
)

xls_ir_opt_ir(
    name = "chaining_example_triple_skip_passes_opt_ir",
    src = ":chaining_example_triple_run_only_opt_ir",
    opt_ir_args = {
        "skip_passes": "arith_simp",
    },
    opt_ir_file = "chaining_example_triple_skip_passes_opt_ir.opt.ir",
    top = get_mangled_ir_symbol("chaining_example", "triple"),
)

xls_ir_equivalence_test(
    name = "chaining_example_triple_ir_equivalence_test",
    src_0 = ":chaining_example_triple_run_only_opt_ir",
    src_1 = ":chaining_example_triple_skip_passes_opt_ir",
    top = get_mangled_ir_symbol("chaining_example", "triple"),
)

################################################################################
# Simple example hierarchy layout:
#     one        two        three  -> DSLX libraries (compile and parse)
#      \__________|__________/
#                 |
#               four      -> DSLX interpreter test
#                |
#               five      -> synthesis, JIT wrapper and tests
################################################################################

# xls_dslx_library target grouping DSLX files.
xls_dslx_library(
    name = "simple_example_123_dslx",
    srcs = [
        "simple_example_one.x",
        "simple_example_three.x",
        "simple_example_two.x",
    ],
)

# xls_dslx_library target depends on xls_dslx_library target
# 'simple_example_123_dslx'.
xls_dslx_library(
    name = "simple_example_4_dslx",
    srcs = ["simple_example_four.x"],
    deps = [":simple_example_123_dslx"],
)

# Execute the DSLX interpreter.
xls_dslx_test(
    name = "simple_example_4_test",
    dslx_test_args = {"compare": "jit"},
    library = ":simple_example_4_dslx",
)

# Generate verilog from a dslx file.
# The rule performs: DSLX -> IR -> Optimized IR -> Verilog.
xls_dslx_verilog(
    name = "simple_example_5_one_stage",
    srcs = ["simple_example_five.x"],
    codegen_args = {
        "pipeline_stages": "1",
        "reset_data_path": "false",
    },
    dslx_top = "main",
    # Output the Optimized IR file to reference.
    opt_ir_file = "simple_example_5_one_stage.opt.ir",
    verilog_file = "simple_example_5_one_stage.sv",
    deps = [":simple_example_4_dslx"],
)

xls_aot_generate(
    name = "simple_example_5_one_stage_jit_wrapper_aot_gen",
    src = ":simple_example_5_one_stage.opt.ir",
    with_msan = False,
)

# The xls_ir_jit_wrapper rule using an optimized IR file as input.
# The header and source files are outputs of the rule, they can be referenced
# by other rules.
xls_ir_jit_wrapper_macro(
    name = "simple_example_5_one_stage_jit_wrapper",
    src = ":simple_example_5_one_stage.opt.ir",
    aot_info = ":simple_example_5_one_stage_jit_wrapper_aot_gen",
    header_file = "simple_example_5_one_stage_jit_wrapper.h",
    jit_wrapper_args = {
        "namespace": "not_xls::test",
    },
    source_file = "simple_example_5_one_stage_jit_wrapper.cc",
    top = "main",
    wrapper_type = FUNCTION_WRAPPER_TYPE,
)

# cc_xls_ir_jit_wrapper macro with a cc_library target with name
# simple_example_5_one_stage_jit_wrapper_cc created.
# The target can be referenced by other cc_* rules.
#
# For example:
#     cc_library (
#         ...
#         deps = [
#             ...
#             ":simple_example_5_one_stage_jit_wrapper_cc",
#             ...
#         ]
#     )
#
cc_xls_ir_jit_wrapper(
    name = "simple_example_5_one_stage_jit_wrapper_cc",
    src = ":simple_example_5_one_stage.opt.ir",
    jit_wrapper_args = {
        "namespace": "not_xls::test",
    },
)

# The rule performs tests using the DSLX, IR and Optimized IR files from the
# xls_dslx_verilog. See the xls_dslx_opt_ir_test rule description for a list of
# tests.
# Note the rule can depend on a xls_dslx_verilog target because the target
# generates all the data that the xls_dslx_opt_ir_test rule requires for its
# tests.
xls_dslx_opt_ir_test(
    name = "simple_example_5_dslx_opt_ir_test",
    dep = ":simple_example_5_one_stage",
)

################################################################################
# Examples using top attribute.
################################################################################
# The rule performs: DSLX -> IR -> Optimized IR. The top attribute is used at
# the Optimized IR stage.
xls_dslx_opt_ir(
    name = "simple_example_5_top_five",
    srcs = ["simple_example_five.x"],
    dslx_top = "five",
    deps = [":simple_example_4_dslx"],
)

# Test for the 'simple_example_5_top_five' target.
xls_dslx_opt_ir_test(
    name = "simple_example_5_top_five_test",
    dep = ":simple_example_5_top_five",
)

################################################################################
# Test for native rules
################################################################################
# The target is the native rule form of the 'chaining_example_ir' target.
xls_dslx_ir_native_rule(
    name = "chaining_example_ir_native_rule",
    srcs = ["chaining_example.x"],
    dslx_top = "main",
)

# The target is the native rule form of the 'chaining_example_opt_ir' target.
xls_ir_opt_ir_native_rule(
    name = "chaining_example_opt_ir_native_rule",
    src = ":chaining_example_ir",
    top = get_mangled_ir_symbol("chaining_example", "main"),
)

# The target is the native rule form of the 'chaining_example_verilog' target.
xls_ir_verilog_native_rule(
    name = "chaining_example_verilog_native_rule",
    src = ":chaining_example_opt_ir",
    codegen_args = {
        "pipeline_stages": "1",
    },
    verilog_file = "chaining_example_verilog_macro.sv",
)

# The target is the native rule form of the 'simple_example_5_entry_five' target.
xls_dslx_opt_ir_native_rule(
    name = "simple_example_5_entry_five_native_rule",
    srcs = ["simple_example_five.x"],
    dslx_top = "five",
    deps = [":simple_example_4_dslx"],
)

# The target is the native rule form of the 'simple_example_5_one_stage' target.
xls_dslx_verilog_native_rule(
    name = "simple_example_5_one_stage_native_rule",
    srcs = ["simple_example_five.x"],
    codegen_args = {
        "pipeline_stages": "1",
    },
    dslx_top = "main",
    verilog_file = "simple_example_5_one_stage_macro.sv",
    deps = [":simple_example_4_dslx"],
)

# The target is the native rule form of the 'simple_example_5_one_stage_jit_wrapper' target.
xls_ir_jit_wrapper_native_rule(
    name = "simple_example_5_one_stage_jit_wrapper_native_rule",
    src = ":simple_example_5_one_stage.opt.ir",
    aot_info = ":simple_example_5_one_stage_jit_wrapper_aot_gen",
    header_file = "simple_example_5_one_stage_jit_wrapper_native_rule.h",
    jit_wrapper_args = {
        "namespace": "not_xls::test",
    },
    source_file = "simple_example_5_one_stage_jit_wrapper_native_rule.cc",
    wrapper_type = FUNCTION_WRAPPER_TYPE,
)

################################################################################
# Tests for user defined output filenames.
################################################################################
xls_dslx_ir(
    name = "user_defined_output_filename_ir",
    srcs = ["chaining_example.x"],
    dslx_top = "main",
    ir_file = "xls_dslx_ir.ir",
)

xls_ir_opt_ir(
    name = "user_defined_output_filename_opt_ir",
    src = ":xls_dslx_ir.ir",
    opt_ir_file = "xls_ir_opt_ir.opt.ir",
)

xls_ir_opt_ir(
    name = "user_defined_output_filename_std_target_opt_ir",
    src = ":user_defined_output_filename_ir",
    opt_ir_file = "xls_ir_opt_ir_std_target.opt.ir",
)

xls_ir_verilog(
    name = "user_defined_output_filename_verilog",
    src = ":xls_ir_opt_ir.opt.ir",
    codegen_args = {
        "pipeline_stages": "1",
    },
    module_sig_file = "xls_ir_verilog.sig.textproto",
    schedule_file = "xls_ir_verilog.schedule.textproto",
    verilog_file = "xls_ir_verilog.sv",
)

xls_ir_verilog(
    name = "user_defined_output_filename_std_target_verilog",
    src = ":user_defined_output_filename_std_target_opt_ir",
    codegen_args = {
        "pipeline_stages": "1",
    },
    module_sig_file = "xls_ir_verilog_std_target.sig.textproto",
    schedule_file = "xls_ir_verilog_std_target.schedule.textproto",
    verilog_file = "xls_ir_verilog_std_target.sv",
)

xls_dslx_ir_native_rule(
    name = "user_defined_output_filename_native_rule_ir",
    srcs = ["chaining_example.x"],
    dslx_top = "main",
    ir_file = "xls_dslx_ir_native_rule.ir",
)

xls_dslx_opt_ir_native_rule(
    name = "user_defined_output_filename_native_rule_dslx_opt_ir",
    srcs = ["chaining_example.x"],
    dslx_top = "main",
    ir_file = "xls_dslx_opt_ir_native_rule.ir",
    opt_ir_file = "xls_dslx_opt_ir_native_rule.opt.ir",
)

xls_ir_opt_ir_native_rule(
    name = "user_defined_output_filename_native_rule_opt_ir",
    src = ":xls_dslx_ir_native_rule.ir",
    opt_ir_file = "xls_ir_opt_ir_native_rule.opt.ir",
)

xls_ir_verilog_native_rule(
    name = "user_defined_output_filename_native_rule_verilog",
    src = ":xls_ir_opt_ir.opt.ir",
    codegen_args = {
        "pipeline_stages": "1",
    },
    module_sig_file = "xls_ir_verilog_native_rule.sig.textproto",
    schedule_file = "xls_ir_verilog_native_rule.schedule.textproto",
    verilog_file = "xls_ir_verilog_native_rule.sv",
)

xls_dslx_verilog_native_rule(
    name = "user_defined_output_filename_xls_dslx_verilog_native_rule_verilog",
    srcs = ["chaining_example.x"],
    codegen_args = {
        "pipeline_stages": "1",
    },
    dslx_top = "main",
    ir_file = "xls_dslx_verilog_native_rule.ir",
    module_sig_file = "xls_dslx_verilog_native_rule.sig.textproto",
    opt_ir_file = "xls_dslx_verilog_native_rule.opt.ir",
    schedule_file = "xls_dslx_verilog_native_rule.schedule.textproto",
    verilog_file = "xls_dslx_verilog_native_rule.sv",
)

xls_aot_generate(
    name = "user_defined_output_filename_jit_wrapper_aot_gen",
    src = ":xls_ir_opt_ir.opt.ir",
    with_msan = False,
)

xls_ir_jit_wrapper_macro(
    name = "user_defined_output_filename_jit_wrapper",
    src = ":xls_ir_opt_ir.opt.ir",
    aot_info = ":user_defined_output_filename_jit_wrapper_aot_gen",
    header_file = "xls_ir_jit_wrapper.h",
    jit_wrapper_args = {
        "namespace": "not_xls::test",
    },
    source_file = "xls_ir_jit_wrapper.cc",
    top = "main",
    wrapper_type = FUNCTION_WRAPPER_TYPE,
)

xls_ir_jit_wrapper_native_rule(
    name = "user_defined_output_filename_native_rule_jit_wrapper",
    src = ":xls_ir_opt_ir.opt.ir",
    aot_info = ":user_defined_output_filename_jit_wrapper_aot_gen",
    header_file = "xls_ir_jit_wrapper_native_rule.h",
    jit_wrapper_args = {
        "namespace": "not_xls::test",
    },
    source_file = "xls_ir_jit_wrapper_native_rule.cc",
    wrapper_type = FUNCTION_WRAPPER_TYPE,
)

################################################################################
# Tests for same source file accessed by different targets.
################################################################################
# File accessed independently through targets
xls_dslx_library(
    name = "a_x_access_0_dslx",
    srcs = [
        ":a.x",
    ],
)

xls_dslx_library(
    name = "a_x_access_1_dslx",
    srcs = [
        ":a.x",
    ],
)

# File accessed through dependency
xls_dslx_library(
    name = "a_x_access_via_dependency_dslx",
    srcs = [
        ":a.x",
    ],
    deps = [
        ":a_dslx",
    ],
)

################################################################################
# Tests of eval_ir input validators.
################################################################################
xls_dslx_library(
    name = "die_if_odd_validator",
    srcs = ["die_if_odd_validator.x"],
)

xls_dslx_opt_ir(
    name = "die_if_odd",
    srcs = ["die_if_odd.x"],
    dslx_top = "die_if_odd",
    ir_file = "die_if_odd.ir",
    opt_ir_file = "die_if_odd.opt.ir",
)

# Check that 'input_validator_expr' works right.
xls_eval_ir_test(
    name = "die_if_odd_eval_ir_test_inline_validator",
    src = ":die_if_odd.ir",
    input_validator_expr = "fn validator(x: u32) -> bool { !((x & u32:1) as bool) }",
)

# Check that 'input_validator' works right.
xls_eval_ir_test(
    name = "die_if_odd_eval_ir_test_file_validator",
    src = ":die_if_odd.ir",
    input_validator = ":die_if_odd_validator",
)

################################################################################
# Verifies that xls_dslx_cpp_type_library macros work to (some) extent.
################################################################################
xls_dslx_cpp_type_library(
    name = "type_definitions_types_cpp",
    src = "type_definitions.x",
)

################################################################################
# Verifies that xls_ir_cc_library builds a library without exploding.
################################################################################
xls_ir_cc_library(
    name = "die_if_odd_cc",
    src = ":die_if_odd",
)

# Verifies that xls_cc_embed_data works correctly
xls_cc_embed_data(
    name = "arbitrary_data_file",
    data = "some_arbitrary_file.txt",
    namespace = "xls::build_rules::test",
)

cc_test(
    name = "embed_data_test",
    srcs = ["embed_data_test.cc"],
    deps = [
        ":arbitrary_data_file",
        "//xls/common:xls_gunit_main",
        "@googletest//:gtest",
    ],
)
