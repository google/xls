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
    "//xls/build_rules:xls_dslx_rules.bzl",
    _dslx_library = "dslx_library",
    _dslx_test = "dslx_test",
)
load(
    "//xls/build_rules:xls_ir_rules.bzl",
    _dslx_to_ir = "dslx_to_ir",
    _ir_benchmark = "ir_benchmark",
    _ir_equivalence_test = "ir_equivalence_test",
    _ir_eval_test = "ir_eval_test",
    _ir_opt = "ir_opt",
)
load(
    "//xls/build_rules:xls_codegen_rules.bzl",
    _ir_to_codegen = "ir_to_codegen",
)
load(
    "//xls/build_rules:xls_jit_wrapper_rules.bzl",
    _cc_ir_to_jit_wrapper = "cc_ir_to_jit_wrapper",
    _ir_to_jit_wrapper = "ir_to_jit_wrapper",
)
load(
    "//xls/build_rules:xls_rules.bzl",
    _dslx_to_codegen = "dslx_to_codegen",
    _dslx_to_ir_opt = "dslx_to_ir_opt",
    _dslx_to_ir_opt_test = "dslx_to_ir_opt_test",
)

dslx_library = _dslx_library
dslx_test = _dslx_test

dslx_to_ir = _dslx_to_ir
ir_benchmark = _ir_benchmark
ir_equivalence_test = _ir_equivalence_test
ir_eval_test = _ir_eval_test
ir_opt = _ir_opt
ir_to_codegen = _ir_to_codegen

cc_ir_to_jit_wrapper = _cc_ir_to_jit_wrapper
ir_to_jit_wrapper = _ir_to_jit_wrapper

dslx_to_codegen = _dslx_to_codegen
dslx_to_ir_opt = _dslx_to_ir_opt
dslx_to_ir_opt_test = _dslx_to_ir_opt_test
