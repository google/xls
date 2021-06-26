# Copyright 2021 The XLS Authors
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

"""
This module contains build rules for XLS.

This module is the primary module and **must** contain all the XLS build rules
exposed to the user. This module is created for convenience.
"""

load(
    "//xls/build_rules:xls_config_rules.bzl",
    _generated_file = "generated_file",
    _presubmit_generated_file = "presubmit_generated_file",
)
load(
    "//xls/build_rules:xls_dslx_rules.bzl",
    _xls_dslx_library = "xls_dslx_library",
    _xls_dslx_module_library = "xls_dslx_module_library",
    _xls_dslx_test = "xls_dslx_test",
)
load(
    "//xls/build_rules:xls_ir_rules.bzl",
    _get_mangled_ir_symbol = "get_mangled_ir_symbol",
    _xls_benchmark_ir = "xls_benchmark_ir",
    _xls_dslx_ir = "xls_dslx_ir",
    _xls_eval_ir_test = "xls_eval_ir_test",
    _xls_ir_equivalence_test = "xls_ir_equivalence_test",
    _xls_ir_opt_ir = "xls_ir_opt_ir",
)
load(
    "//xls/build_rules:xls_codegen_rules.bzl",
    _xls_ir_verilog = "xls_ir_verilog",
)
load(
    "//xls/build_rules:xls_jit_wrapper_rules.bzl",
    _cc_xls_ir_jit_wrapper = "cc_xls_ir_jit_wrapper",
    _xls_ir_jit_wrapper = "xls_ir_jit_wrapper",
)
load(
    "//xls/build_rules:xls_rules.bzl",
    _xls_dslx_opt_ir = "xls_dslx_opt_ir",
    _xls_dslx_opt_ir_test = "xls_dslx_opt_ir_test",
    _xls_dslx_verilog = "xls_dslx_verilog",
)
load(
    "//xls/build_rules:xls_codegen_macros.bzl",
    _xls_ir_verilog_macro = "xls_ir_verilog_macro",
)
load(
    "//xls/build_rules:xls_macros.bzl",
    _xls_dslx_verilog_macro = "xls_dslx_verilog_macro",
)

# General Functions
generated_file = _generated_file
presubmit_generated_file = _presubmit_generated_file

# XLS Rules
xls_dslx_library = _xls_dslx_library
xls_dslx_module_library = _xls_dslx_module_library
xls_dslx_test = _xls_dslx_test

xls_dslx_ir = _xls_dslx_ir
get_mangled_ir_symbol = _get_mangled_ir_symbol
xls_benchmark_ir = _xls_benchmark_ir
xls_ir_equivalence_test = _xls_ir_equivalence_test
xls_eval_ir_test = _xls_eval_ir_test
xls_ir_opt_ir = _xls_ir_opt_ir
xls_ir_verilog = _xls_ir_verilog

cc_xls_ir_jit_wrapper = _cc_xls_ir_jit_wrapper
xls_ir_jit_wrapper = _xls_ir_jit_wrapper

xls_dslx_verilog = _xls_dslx_verilog
xls_dslx_opt_ir = _xls_dslx_opt_ir
xls_dslx_opt_ir_test = _xls_dslx_opt_ir_test

# XLS Macros
xls_ir_verilog_macro = _xls_ir_verilog_macro
xls_dslx_verilog_macro = _xls_dslx_verilog_macro
