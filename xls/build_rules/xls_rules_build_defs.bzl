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

This module contain all the XLS build rules that require testing but are not
exposed to the user. This module is created for convenience.
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
    "//xls/build_rules:xls_rules.bzl",
    _xls_dslx_opt_ir = "xls_dslx_opt_ir",
    _xls_dslx_verilog = "xls_dslx_verilog",
)

xls_dslx_ir = _xls_dslx_ir
xls_dslx_opt_ir = _xls_dslx_opt_ir
xls_dslx_verilog = _xls_dslx_verilog
xls_ir_opt_ir = _xls_ir_opt_ir
xls_ir_verilog = _xls_ir_verilog
