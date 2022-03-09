# Copyright 2022 The XLS Authors
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
This module contains build rules for XLScc.

This module is the primary module and **must** contain all the XLScc build rules
exposed to the user. This module is created for convenience.
"""

load(
    "//xls/contrib/xlscc/build_rules:xlscc_rules.bzl",
    _xls_cc_ir_macro = "xls_cc_ir_macro",
    _xls_cc_verilog_macro = "xls_cc_verilog_macro",
)

xls_cc_ir = _xls_cc_ir_macro
xls_cc_verilog = _xls_cc_verilog_macro
