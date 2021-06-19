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

# Integration tests for the generated JIT wrapper being invoked from C++ code.

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "cc_xls_ir_jit_wrapper",
    "xls_dslx_module_library",
    "xls_dslx_opt_ir",
    "xls_dslx_test",
)

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_module_library(
    name = "simple_dslx_module",
    src = "simple.x",
)

xls_dslx_opt_ir(
    name = "simple",
    dep = ":simple_dslx_module",
    opt_ir_file = "simple.opt.ir",
)

xls_dslx_test(
    name = "simple_dslx_test",
    dep = ":simple_dslx_module",
)

cc_xls_ir_jit_wrapper(
    name = "identity",
    src = ":simple.opt.ir",
    jit_wrapper_args = {
        "class_name": "F32Identity",
        "function": "identity",
    },
)

cc_xls_ir_jit_wrapper(
    name = "is_inf",
    src = ":simple.opt.ir",
    jit_wrapper_args = {
        "class_name": "F32IsInf",
        "function": "is_inf",
    },
)

cc_xls_ir_jit_wrapper(
    name = "fail_on_42",
    src = ":simple.opt.ir",
    jit_wrapper_args = {
        "class_name": "FailOn42",
        "function": "fail_on_42",
    },
)

cc_test(
    name = "simple_jit_wrapper_test",
    srcs = ["simple_jit_wrapper_test.cc"],
    deps = [
        ":fail_on_42",
        ":identity",
        ":is_inf",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:value_helpers",
        "@com_google_googletest//:gtest",
    ],
)
