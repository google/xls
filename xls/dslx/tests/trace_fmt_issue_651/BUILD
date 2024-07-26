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

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "cc_xls_ir_jit_wrapper",
    "xls_dslx_ir",
    "xls_dslx_library",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_library(
    name = "test_target_library",
    srcs = ["test_target.x"],
)

xls_dslx_ir(
    name = "test_target_u16",
    dslx_top = "trace_u16",
    library = ":test_target_library",
)

xls_dslx_ir(
    name = "test_target_u21",
    dslx_top = "trace_u21",
    library = ":test_target_library",
)

xls_dslx_ir(
    name = "test_target_s32",
    dslx_top = "trace_s32",
    library = ":test_target_library",
)

xls_dslx_ir(
    name = "test_target_enum",
    dslx_top = "trace_enum",
    library = ":test_target_library",
)

cc_xls_ir_jit_wrapper(
    name = "test_target_u16_wrapper",
    src = ":test_target_u16.ir",
    jit_wrapper_args = {
        "class_name": "Trace_u16",
        "namespace": "xls::wrapped",
    },
)

cc_xls_ir_jit_wrapper(
    name = "test_target_u21_wrapper",
    src = ":test_target_u21.ir",
    jit_wrapper_args = {
        "class_name": "Trace_u21",
        "namespace": "xls::wrapped",
    },
)

cc_xls_ir_jit_wrapper(
    name = "test_target_s32_wrapper",
    src = ":test_target_s32.ir",
    jit_wrapper_args = {
        "class_name": "Trace_s32",
        "namespace": "xls::wrapped",
    },
)

cc_xls_ir_jit_wrapper(
    name = "test_target_enum_wrapper",
    src = ":test_target_enum.ir",
    jit_wrapper_args = {
        "class_name": "TraceEnum",
        "namespace": "xls::wrapped",
    },
)

cc_test(
    name = "trace_fmt_test",
    srcs = ["trace_fmt_test.cc"],
    deps = [
        ":test_target_enum_wrapper",
        ":test_target_s32_wrapper",
        ":test_target_u16_wrapper",
        ":test_target_u21_wrapper",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/types:span",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "//xls/ir:events",
        "//xls/ir:value",
        "//xls/ir:value_builder",
    ],
)
