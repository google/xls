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

# Build rules for DSLX modules.
load("//xls/build_rules:build_defs.bzl", "dslx_jit_wrapper", "dslx_test")

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "ir_examples",
    srcs = [
        ":fpadd_2x32_all_ir",
        ":fpmul_2x32_all_ir",
    ],
)

dslx_test(
    name = "fpadd_2x32",
    srcs = ["fpadd_2x32.x"],
    # TODO(b/152546795): Takes too long.
    prove_unopt_eq_opt = False,
)

dslx_jit_wrapper(
    name = "fpadd_2x32_jit_wrapper",
    dslx_name = "fpadd_2x32",
    deps = [":fpadd_2x32_opt_ir"],
)

# TODO(rspringer): Takes too long to run in normal testing.
cc_binary(
    name = "fpadd_2x32_bounds",
    srcs = ["fpadd_2x32_bounds.cc"],
    data = [":fpadd_2x32_all_ir"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/ir:ir_parser",
        "//xls/solvers:z3_ir_translator",
        "//xls/solvers:z3_utils",
        "@z3//:api",
    ],
)

cc_test(
    name = "fpadd_2x32_test",
    srcs = ["fpadd_2x32_test.cc"],
    data = [":fpadd_2x32_all_ir"],
    tags = ["optonly"],
    deps = [
        ":fpadd_2x32_jit_wrapper",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value_helpers",
        "//xls/ir:value_view_helpers",
        "//xls/tools:testbench",
    ],
)

dslx_test(
    name = "fpsub_2x32",
    srcs = ["fpsub_2x32.x"],
    deps = ["fpadd_2x32.x"],
    #TODO(b/152546795): Takes too long.
    prove_unopt_eq_opt = False,
)

dslx_test(
    name = "fpmul_2x32",
    srcs = ["fpmul_2x32.x"],
    # TODO(rspringer): Currently takes too long.
    prove_unopt_eq_opt = False,
)

dslx_jit_wrapper(
    name = "fpmul_2x32_jit_wrapper",
    dslx_name = "fpmul_2x32",
    deps = [":fpmul_2x32_opt_ir"],
)

cc_test(
    name = "fpmul_2x32_test",
    srcs = ["fpmul_2x32_test.cc"],
    data = [":fpmul_2x32_all_ir"],
    deps = [
        ":fpmul_2x32_jit_wrapper",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value_helpers",
        "//xls/ir:value_view_helpers",
        "//xls/tools:testbench",
    ],
)

cc_test(
    name = "fpadd_2x32_jit_wrapper_test",
    srcs = ["fpadd_2x32_jit_wrapper_test.cc"],
    deps = [
        ":fpadd_2x32_jit_wrapper",
        "//xls/common/status:matchers",
        "//xls/ir:bits",
        "//xls/ir:value",
        "//xls/ir:value_helpers",
        "@com_google_googletest//:gtest_main",
    ],
)
