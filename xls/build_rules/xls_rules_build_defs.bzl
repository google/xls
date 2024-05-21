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
This module contains build rules/macros for XLS.

This module contain all the XLS build rules/macros that require testing but are
not exposed to the user. This module is created for convenience.
"""

load(
    "//xls/build_rules:xls_codegen_rules.bzl",
    _xls_ir_verilog = "xls_ir_verilog",
)
load(
    "//xls/build_rules:xls_ir_rules.bzl",
    _xls_dslx_ir = "xls_dslx_ir",
    _xls_ir_opt_ir = "xls_ir_opt_ir",
)
load(
    "//xls/build_rules:xls_jit_wrapper_rules.bzl",
    _BLOCK_WRAPPER_TYPE = "BLOCK_WRAPPER_TYPE",
    _FUNCTION_WRAPPER_TYPE = "FUNCTION_WRAPPER_TYPE",
    _PROC_WRAPPER_TYPE = "PROC_WRAPPER_TYPE",
    _xls_ir_jit_wrapper = "xls_ir_jit_wrapper",
    _xls_ir_jit_wrapper_macro = "xls_ir_jit_wrapper_macro",
)
load(
    "//xls/build_rules:xls_rules.bzl",
    _xls_dslx_opt_ir = "xls_dslx_opt_ir",
    _xls_dslx_verilog = "xls_dslx_verilog",
)

xls_dslx_ir = _xls_dslx_ir
xls_dslx_opt_ir = _xls_dslx_opt_ir
xls_dslx_verilog = _xls_dslx_verilog
xls_ir_opt_ir = _xls_ir_opt_ir
xls_ir_verilog = _xls_ir_verilog
xls_ir_jit_wrapper = _xls_ir_jit_wrapper
xls_ir_jit_wrapper_macro = _xls_ir_jit_wrapper_macro
FUNCTION_WRAPPER_TYPE = _FUNCTION_WRAPPER_TYPE
PROC_WRAPPER_TYPE = _PROC_WRAPPER_TYPE
BLOCK_WRAPPER_TYPE = _BLOCK_WRAPPER_TYPE
