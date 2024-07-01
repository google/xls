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
    "xls_dslx_opt_ir",
    "xls_dslx_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    features = [
        "layering_check",
        "parse_headers",
    ],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_test(
    name = "simple_dslx_test",
    srcs = ["simple.x"],
)

xls_dslx_opt_ir(
    name = "simple_identity",
    srcs = ["simple.x"],
    dslx_top = "identity",
)

cc_xls_ir_jit_wrapper(
    name = "identity",
    src = ":simple_identity",
    jit_wrapper_args = {
        "class_name": "F32Identity",
        "function": "identity",
        "namespace": "xls::test",
    },
)

xls_dslx_opt_ir(
    name = "simple_wide_identity",
    srcs = ["simple.x"],
    dslx_top = "wide_identity",
)

cc_xls_ir_jit_wrapper(
    name = "wide_identity",
    src = ":simple_wide_identity",
    jit_wrapper_args = {
        "class_name": "WideIdentity",
        "function": "wide_identity",
        "namespace": "xls::test",
    },
)

xls_dslx_opt_ir(
    name = "simple_is_inf",
    srcs = ["simple.x"],
    dslx_top = "is_inf",
)

cc_xls_ir_jit_wrapper(
    name = "is_inf",
    src = ":simple_is_inf",
    jit_wrapper_args = {
        "class_name": "F32IsInf",
        "function": "is_inf",
        "namespace": "xls::test",
    },
)

xls_dslx_opt_ir(
    name = "simple_fail_on_42",
    srcs = ["simple.x"],
    dslx_top = "fail_on_42",
)

cc_xls_ir_jit_wrapper(
    name = "fail_on_42",
    src = ":simple_fail_on_42",
    jit_wrapper_args = {
        "class_name": "FailOn42",
        "function": "fail_on_42",
        "namespace": "xls::test",
    },
)

xls_dslx_opt_ir(
    name = "simple_make_tuple",
    srcs = ["simple.x"],
    dslx_top = "make_tuple",
)

cc_xls_ir_jit_wrapper(
    name = "make_tuple",
    src = ":simple_make_tuple",
    jit_wrapper_args = {
        "class_name": "MakeTuple",
        "function": "make_tuple",
        "namespace": "xls::test",
    },
)

cc_test(
    name = "simple_jit_wrapper_test",
    srcs = ["simple_jit_wrapper_test.cc"],
    deps = [
        ":fail_on_42",
        ":identity",
        ":is_inf",
        ":make_tuple",
        ":wide_identity",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "//xls/ir:value_view",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)
