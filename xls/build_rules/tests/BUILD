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
