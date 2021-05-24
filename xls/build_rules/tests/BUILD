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

load(
    "//xls/build_rules:build_defs.bzl",
    "dslx_generated_rtl",
    "dslx_test",
)
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "cc_xls_ir_jit_wrapper",
    "xls_benchmark_ir",
    "xls_dslx_ir",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_opt_ir_test",
    "xls_dslx_test",
    "xls_dslx_verilog",
    "xls_eval_ir_test",
    "xls_ir_equivalence_test",
    "xls_ir_jit_wrapper",
    "xls_ir_opt_ir",
    "xls_ir_verilog",
)

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

# Ensure that we can take a dep on a genrule .x file and import it.
genrule(
    name = "generated_x",
    outs = ["generated.x"],
    cmd = "echo 'pub const VALUE = u32:42;' > $(OUTS) && " +
          "echo 'fn main() -> u32 { VALUE }' >> $(OUTS) && " +
          "echo '#![test]' >> $(OUTS) && " +
          "echo 'fn test_value() { assert_eq(VALUE, u32:42) }' >> $(OUTS)",
)

# Check that we can run the test inside of the generated file, convert it to
# IR, etc.
dslx_test(
    name = "generated",
    srcs = ["generated.x"],
)

# Check that we can use the generated file inside of a dslx_test rule (and that
# all resulting test targets it creates pass soundly).
dslx_test(
    name = "importer",
    srcs = ["importer.x"],
    deps = [
        "generated.x",
    ],
)

# Check that we can generate a block a couple different ways.
dslx_generated_rtl(
    name = "add_one_pipeline_rtl",
    srcs = ["add_one.x"],
    codegen_params = {
        "pipeline_stages": 2,
        "delay_model": "unit",
    },
)

sh_test(
    name = "add_one_pipeline_rtl_has_clk_test",
    srcs = ["has_clk.sh"],
    args = ["$(rootpath :add_one_pipeline_rtl.v)"],
    data = [":add_one_pipeline_rtl.v"],
)

dslx_generated_rtl(
    name = "add_one_combinational_rtl",
    srcs = ["add_one.x"],
    codegen_params = {
        "generator": "combinational",
    },
)

sh_test(
    name = "add_one_combinational_rtl_has_clk_test",
    srcs = ["not_has_clk.sh"],
    args = ["$(rootpath :add_one_combinational_rtl.v)"],
    data = [":add_one_combinational_rtl.v"],
)

xls_dslx_library(
    name = "a",
    srcs = [
        "a.x",
    ],
)

# xls_dslx_library target depends on xls_dslx_library target 'a'.
xls_dslx_library(
    name = "b",
    srcs = [
        "b.x",
    ],
    deps = [
        ":a",
    ],
)

xls_dslx_library(
    name = "c",
    srcs = [
        "c.x",
    ],
    deps = [
        ":a",
    ],
)

# xls_dslx_library target grouping DSLX files.
xls_dslx_library(
    name = "files_123",
    srcs = [
        "file_1.x",
        "file_2.x",
        "file_3.x",
    ],
)

xls_dslx_library(
    name = "file_4",
    srcs = [
        "file_4.x",
    ],
)

# xls_dslx_library target depends on xls_dslx_library target 'file_4'.
xls_dslx_library(
    name = "file_5",
    srcs = [
        "file_5.x",
    ],
    deps = [
        ":file_4",
    ],
)

# xls_dslx_test with no dependencies.
xls_dslx_test(
    name = "a_test",
    src = "a.x",
)

# xls_dslx_test with a dependency.
xls_dslx_test(
    name = "d_test",
    src = "d.x",
    deps = [
        ":c",
    ],
)

# xls_dslx_ir target depending on xls_dslx_library target 'c'.
xls_dslx_ir(
    name = "d_ir",
    src = "d.x",
    deps = [
        ":c",
    ],
)

# xls_dslx_ir target with an entry defined.
xls_dslx_ir(
    name = "a_ir",
    src = "a.x",
    ir_conv_args = {
        "entry": "a",
    },
)

# xls_ir_opt_ir target with an entry defined.
xls_ir_opt_ir(
    name = "a_opt_ir",
    src = "a_ir",
    opt_ir_args = {
        "entry": "__a__a",
    },
)

# xls_eval_ir_test target with targets as the sources.
xls_ir_equivalence_test(
    name = "a_equivalence_test",
    ir_equivalence_args = {
        "function": "__a__a",
    },
    src_0 = ":a_ir",
    src_1 = ":a_opt_ir",
)

# xls_eval_ir_test target with a target as a source.
xls_eval_ir_test(
    name = "a_eval_test",
    src = ":a_ir",
)

# xls_benchmark_ir target with a target as a source.
xls_benchmark_ir(
    name = "a_benchmark",
    src = ":a_ir",
    benchmark_ir_args = {
        "entry": "__a__a",
    },
)

# xls_ir_verilog target with a target as a source.
xls_ir_verilog(
    name = "a_verilog",
    src = ":a_ir",
    codegen_args = {
        "pipeline_stages": "1",
    },
)

xls_dslx_opt_ir(
    name = "file_5_opt_ir",
    src = "file_5.x",
    deps = [
        ":file_4",
    ],
)

xls_dslx_opt_ir_test(
    name = "file_5_opt_ir_test",
    dep = ":file_5_opt_ir",
)

xls_dslx_verilog(
    name = "e_verilog",
    src = "e.x",
    codegen_args = {
        "pipeline_stages": "1",
    },
)

# xls_dslx_ir target to test JIT wrapper rules
xls_dslx_ir(
    name = "jit_test_dslx_ir",
    src = "file_for_ir_to_jit_wrapper_test.x",
)

xls_ir_jit_wrapper(
    name = "jit_test_jit_wrapper",
    src = ":jit_test_dslx_ir",
)

cc_xls_ir_jit_wrapper(
    name = "jit_test_jit_wrapper_cc",
    src = ":jit_test_dslx_ir",
)

xls_dslx_library(
    name = "generated_dslx",
    srcs = ["generated.x"],
)

xls_dslx_ir(
    name = "with_dep_test",
    src = "importer.x",
    deps = [
        "generated_dslx",
    ],
)

xls_dslx_ir(
    name = "as_input_test",
    src = "generated.x",
)
