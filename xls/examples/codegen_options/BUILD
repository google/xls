# Copyright 2024 The XLS Authors
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

# Build rules demoing different codegen options.
load("@bazel_skylib//rules:build_test.bzl", "build_test")
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "xls_dslx_opt_ir",
    "xls_ir_verilog",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],
)

xls_dslx_opt_ir(
    name = "f_proc",
    srcs = ["f_proc.x"],
    dslx_top = "f_proc",
)

xls_ir_verilog(
    name = "f_proc_sv",
    src = ":f_proc.opt.ir",
    codegen_args = {
        "generator": "pipeline",
        "pipeline_stages": "4",
        "flop_inputs": "false",
        "flop_outputs": "false",
        "use_system_verilog": "true",
        "module_name": "f_proc",
        "codegen_version": "2",
    },
    enable_presubmit_generated_file = True,
    verilog_file = "f_proc.sv",
)

xls_ir_verilog(
    name = "f_proc_as_comb_sv",
    src = ":f_proc.opt.ir",
    codegen_args = {
        "generator": "combinational",
        "pipeline_stages": "1",
        "flop_inputs": "false",
        "flop_outputs": "false",
        "use_system_verilog": "true",
        "module_name": "f_proc_as_comb",
        "codegen_version": "2",
    },
    enable_presubmit_generated_file = True,
    verilog_file = "f_proc_as_comb.sv",
)

xls_dslx_opt_ir(
    name = "f_func",
    srcs = ["f_func.x"],
    dslx_top = "f_func",
)

xls_ir_verilog(
    name = "f_func_sv",
    src = ":f_func.opt.ir",
    codegen_args = {
        "generator": "pipeline",
        "pipeline_stages": "3",
        "flop_inputs": "false",
        "flop_outputs": "false",
        "use_system_verilog": "true",
        "module_name": "f_func",
        "codegen_version": "2",
    },
    enable_presubmit_generated_file = True,
    verilog_file = "f_func.sv",
)

xls_ir_verilog(
    name = "f_func_as_comb_sv",
    src = ":f_func.opt.ir",
    codegen_args = {
        "generator": "combinational",
        "pipeline_stages": "1",
        "flop_inputs": "false",
        "flop_outputs": "false",
        "use_system_verilog": "true",
        "module_name": "f_func_as_comb",
        "codegen_version": "2",
    },
    enable_presubmit_generated_file = True,
    verilog_file = "f_func_as_comb.sv",
)

build_test(
    name = "verilog_targets_build",
    targets = [
        ":f_proc_sv",
        ":f_proc_as_comb_sv",
        ":f_func_sv",
        ":f_func_as_comb_sv",
    ],
)
