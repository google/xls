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
    "xls_dslx_library",
)
load("build_defs.bzl", "xls_dslx_ir_wrapper")

package(
    default_applicable_licenses = ["//:license"],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_library(
    name = "test_target_library",
    srcs = ["test_target.x"],
)

xls_dslx_ir_wrapper(
    name = "trace_u16",
)

xls_dslx_ir_wrapper(
    name = "trace_u21",
)

xls_dslx_ir_wrapper(
    name = "trace_s32",
)

xls_dslx_ir_wrapper(
    name = "trace_enum",
)

xls_dslx_ir_wrapper(
    name = "trace_u16_hex",
)

xls_dslx_ir_wrapper(
    name = "trace_u21_hex",
)

cc_test(
    name = "trace_fmt_test",
    srcs = ["trace_fmt_test.cc"],
    deps = [
        ":trace_enum_wrapper",
        ":trace_s32_wrapper",
        ":trace_u16_hex_wrapper",
        ":trace_u16_wrapper",
        ":trace_u21_hex_wrapper",
        ":trace_u21_wrapper",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "//xls/ir:events",
        "//xls/ir:value",
        "//xls/ir:value_builder",
        "@com_google_absl//absl/types:span",
        "@googletest//:gtest",
    ],
)
