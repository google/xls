# Copyright 2023 The XLS Authors
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

xls_dslx_opt_ir(
    name = "umul_with_overflow_21_21_18",
    srcs = ["umul_instances.x"],
    dslx_top = "umul_with_overflow_21_21_18",
)

cc_xls_ir_jit_wrapper(
    name = "umul_with_overflow_21_21_18_jit_wrapper",
    src = ":umul_with_overflow_21_21_18.opt.ir",
    jit_wrapper_args = {
        "class_name": "UmulWithOverflow212118",
        "function": "umul_with_overflow_21_21_18",
        "namespace": "xls::test",
    },
)

xls_dslx_opt_ir(
    name = "umul_with_overflow_35_32_18",
    srcs = ["umul_instances.x"],
    dslx_top = "umul_with_overflow_35_32_18",
)

cc_xls_ir_jit_wrapper(
    name = "umul_with_overflow_35_32_18_jit_wrapper",
    src = ":umul_with_overflow_35_32_18.opt.ir",
    jit_wrapper_args = {
        "class_name": "UmulWithOverflow353218",
        "function": "umul_with_overflow_35_32_18",
        "namespace": "xls::test",
    },
)

cc_test(
    name = "umul_with_overflow_test",
    srcs = ["umul_with_overflow_test.cc"],
    deps = [
        ":umul_with_overflow_21_21_18_jit_wrapper",
        ":umul_with_overflow_35_32_18_jit_wrapper",
        "@com_google_googletest//:gtest",
        "@com_google_fuzztest//fuzztest",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:value_view",
    ],
)

# Run localy with
#  bazel test -c opt ///xls/tests/umul_with_overflow:umul_with_overflow_long_test --test_timeout=20000
# Takes ~2hr to complete.
cc_test(
    name = "umul_with_overflow_long_test",
    srcs = ["umul_with_overflow_long_test.cc"],
    shard_count = 50,
    tags = [
        "local",
        "manual",
        "optonly",
    ],
    deps = [
        ":umul_with_overflow_21_21_18_jit_wrapper",
        ":umul_with_overflow_35_32_18_jit_wrapper",
        "@com_google_googletest//:gtest",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:value_view",
    ],
)
