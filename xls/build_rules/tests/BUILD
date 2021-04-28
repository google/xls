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

load("//xls/build_rules:build_defs.bzl", "dslx_generated_rtl", "dslx_test")
load("//xls/build_rules:xls_dslx_rules.bzl", "dslx_library", dslx_ng_test = "dslx_test")
load("//xls/build_rules:xls_ir_rules.bzl", "dslx_to_ir", "ir_benchmark", "ir_equivalence_test", "ir_eval_test", "ir_opt")
load("//xls/build_rules:xls_codegen_rules.bzl", "ir_to_codegen")

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

dslx_library(
    name = "a",
    srcs = [
        "a.x",
    ],
)

# dslx_library target depends on dslx_library target 'a'.
dslx_library(
    name = "b",
    srcs = [
        "b.x",
    ],
    deps = [
        ":a",
    ],
)

dslx_library(
    name = "c",
    srcs = [
        "c.x",
    ],
    deps = [
        ":a",
    ],
)

# dslx_library target grouping DSLX files.
dslx_library(
    name = "files_123",
    srcs = [
        "file_1.x",
        "file_2.x",
        "file_3.x",
    ],
)

dslx_library(
    name = "file_4",
    srcs = [
        "file_4.x",
    ],
)

# dslx_library target depends on dslx_library target 'file_4'.
dslx_library(
    name = "file_5",
    srcs = [
        "file_5.x",
    ],
    deps = [
        ":file_4",
    ],
)

# dslx_test with no dependencies.
# TODO(vmirian) 04-13-21 after merge rename to dslx_test
dslx_ng_test(
    name = "a_test",
    src = "a.x",
)

# dslx_test with a dependency.
# TODO(vmirian) 04-13-21 after merge rename to dslx_test
dslx_ng_test(
    name = "d_test",
    src = "d.x",
    deps = [
        ":c",
    ],
)

# dslx_to_ir target depending on dslx_library target 'c'.
dslx_to_ir(
    name = "d_dslx_to_ir",
    src = "d.x",
    deps = [
        ":c",
    ],
)

# dslx_to_ir target with an entry defined.
dslx_to_ir(
    name = "a_dslx_to_ir",
    src = "a.x",
    ir_conv_args = {
        "entry": "a",
    },
)

# ir_opt target with an entry defined.
ir_opt(
    name = "a_ir_opt",
    src = "a_dslx_to_ir",
    ir_opt_args = {
        "entry": "__a__a",
    },
)

# ir_eval_test target with targets as the sources.
ir_equivalence_test(
    name = "a_equivalence_test",
    ir_equivalence_args = {
        "function": "__a__a",
    },
    src_0 = ":a_dslx_to_ir",
    src_1 = ":a_ir_opt",
)

# ir_eval_test target with a target as a source.
ir_eval_test(
    name = "a_eval_test",
    src = ":a_dslx_to_ir",
)

# ir_benchmark target with a target as a source.
ir_benchmark(
    name = "a_benchmark",
    src = ":a_dslx_to_ir",
    benchmark_args = {
        "entry": "__a__a",
    },
)

# ir_to_codegen target with a target as a source.
ir_to_codegen(
    name = "a_ir_to_codegen",
    src = ":a_dslx_to_ir",
    codegen_args = {
        "pipeline_stages": "1",
    },
)
