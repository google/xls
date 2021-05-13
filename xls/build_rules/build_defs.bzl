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

"""Contains macros for DSLX targets."""

load("//xls/build_rules:dslx_test.bzl", _dslx_mangle = "dslx_mangle", _dslx_test = "dslx_test")
load("//xls/build_rules:dslx_codegen.bzl", _dslx_codegen = "dslx_codegen")
load("//xls/build_rules:dslx_generated_rtl.bzl", _dslx_generated_rtl = "dslx_generated_rtl")
load("//xls/build_rules:dslx_jit_wrapper.bzl", _dslx_jit_wrapper = "dslx_jit_wrapper")

dslx_test = _dslx_test
dslx_mangle = _dslx_mangle
dslx_codegen = _dslx_codegen
dslx_generated_rtl = _dslx_generated_rtl
dslx_jit_wrapper = _dslx_jit_wrapper
