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
load(
    "//xls/build_rules:xls_build_defs.bzl",
    "cc_xls_ir_jit_wrapper",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_opt_ir_test",
    "xls_dslx_test",
)

package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_library(
    name = "apfloat_add_2_dslx",
    srcs = ["apfloat_add_2.x"],
)

xls_dslx_test(
    name = "apfloat_sub_2",
    src = "apfloat_sub_2.x",
    dslx_test_args = {
        "compare": "none",
    },
    deps = ["apfloat_add_2_dslx"],
)

filegroup(
    name = "ir_examples",
    srcs = [
        ":fpadd_2x32.ir",
        ":fpadd_2x32.opt.ir",
        ":fpmul_2x32.ir",
        ":fpmul_2x32.opt.ir",
    ],
)

xls_dslx_library(
    name = "apfloat_fma_dslx",
    srcs = ["apfloat_fma.x"],
)

# TODO(rspringer): 2021-04-09: Make a macro to handle creating the three
# related rules. Consider also doing this for multiple widths (32, 64, bf16).
xls_dslx_opt_ir(
    name = "fma_32",
    src = "fma_32.x",
    ir_file = "fma_32.ir",
    opt_ir_file = "fma_32.opt.ir",
    deps = [":apfloat_fma_dslx"],
)

xls_dslx_opt_ir_test(
    name = "fma_32_xls_test",
    dep = ":fma_32",
    # TODO(b/152546795): 2021-03-08 Takes too long (~5 minutes).
    prove_unopt_eq_opt = False,
)

cc_xls_ir_jit_wrapper(
    name = "fma_32_jit_wrapper",
    src = "fma_32.opt.ir",
    jit_wrapper_args = {
        "class_name": "fma_32",
    },
)

cc_test(
    name = "fma_32_test",
    srcs = ["fma_32_test.cc"],
    data = [
        ":fma_32.ir",
        ":fma_32.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":fma_32_jit_wrapper",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/common/logging",
        "//xls/tools:testbench",
    ],
)

xls_dslx_opt_ir(
    name = "fma_64",
    src = "fma_64.x",
    ir_file = "fma_64.ir",
    opt_ir_file = "fma_64.opt.ir",
    deps = [":apfloat_fma_dslx"],
)

xls_dslx_opt_ir_test(
    name = "fma_64_xls_test",
    dep = ":fma_64",
    # TODO(b/152546795): 2021-03-08 Takes too long (~5 minutes).
    prove_unopt_eq_opt = False,
)

cc_xls_ir_jit_wrapper(
    name = "fma_64_jit_wrapper",
    src = "fma_64.opt.ir",
    jit_wrapper_args = {
        "class_name": "fma_64",
    },
)

cc_test(
    name = "fma_64_test",
    srcs = ["fma_64_test.cc"],
    data = [
        ":fma_64.ir",
        ":fma_64.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":fma_64_jit_wrapper",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/common/logging",
        "//xls/tools:testbench",
    ],
)

xls_dslx_library(
    name = "apfloat_sub_2_dslx",
    srcs = ["apfloat_sub_2.x"],
    deps = [":apfloat_add_2_dslx"],
)

xls_dslx_library(
    name = "fpadd_2x32_dslx",
    srcs = ["fpadd_2x32.x"],
    deps = [":apfloat_add_2_dslx"],
)

xls_dslx_opt_ir(
    name = "fpadd_2x32",
    src = "fpadd_2x32.x",
    ir_file = "fpadd_2x32.ir",
    opt_ir_file = "fpadd_2x32.opt.ir",
    deps = [":apfloat_add_2_dslx"],
)

xls_dslx_opt_ir_test(
    name = "fpadd_2x32_test",
    dep = ":fpadd_2x32",
    # TODO(b/152546795): 2021-03-08 Takes too long (~5 minutes).
    prove_unopt_eq_opt = False,
)

xls_dslx_opt_ir(
    name = "fpsub_2x32",
    src = "fpsub_2x32.x",
    deps = [
        ":apfloat_sub_2_dslx",
        ":fpadd_2x32_dslx",
    ],
)

xls_dslx_opt_ir_test(
    name = "fpsub_2x32_test",
    dep = ":fpsub_2x32",
    # TODO(b/152546795): 2021-03-08 Takes too long (~5 minutes).
    prove_unopt_eq_opt = False,
)

xls_dslx_library(
    name = "fpadd_2x64_dslx",
    srcs = ["fpadd_2x64.x"],
    deps = [
        ":apfloat_add_2_dslx",
        ":apfloat_sub_2_dslx",
    ],
)

xls_dslx_opt_ir(
    name = "fpsub_2x64",
    src = "fpsub_2x64.x",
    deps = [":fpadd_2x64_dslx"],
)

xls_dslx_opt_ir_test(
    name = "fpsub_2x64_test",
    dep = ":fpsub_2x64",
    # TODO(b/152546795): 2021-03-08 Takes too long (~5 minutes).
    prove_unopt_eq_opt = False,
)

cc_xls_ir_jit_wrapper(
    name = "fpadd_2x32_jit_wrapper",
    src = "fpadd_2x32.opt.ir",
    jit_wrapper_args = {
        "class_name": "fpadd_2x32",
    },
)

# TODO(rspringer): Takes too long to run in normal testing.
cc_binary(
    name = "fpadd_2x32_bounds",
    srcs = ["fpadd_2x32_bounds.cc"],
    data = [
        ":fpadd_2x32.ir",
        ":fpadd_2x32.opt.ir",
    ],
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
    name = "fpadd_2x32_test_cc",
    srcs = ["fpadd_2x32_test.cc"],
    tags = ["optonly"],
    deps = [
        ":fpadd_2x32_jit_wrapper",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value_helpers",
        "//xls/ir:value_view_helpers",
        "//xls/tools:testbench",
    ],
)

xls_dslx_opt_ir(
    name = "fpadd_2x64",
    src = "fpadd_2x64.x",
    ir_file = "fpadd_2x64.ir",
    opt_ir_file = "fpadd_2x64.opt.ir",
    deps = [
        ":apfloat_add_2_dslx",
    ],
)

xls_dslx_opt_ir_test(
    name = "fpadd_2x64_test",
    dep = ":fpadd_2x64",
    # 2021-03-08 Takes too long (> 15 minutes).
    prove_unopt_eq_opt = False,
)

cc_xls_ir_jit_wrapper(
    name = "fpadd_2x64_jit_wrapper",
    src = "fpadd_2x64.opt.ir",
    jit_wrapper_args = {
        "class_name": "fpadd_2x64",
    },
)

cc_test(
    name = "fpadd_2x64_test_cc",
    srcs = ["fpadd_2x64_test.cc"],
    data = [
        ":fpadd_2x64.ir",
        ":fpadd_2x64.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":fpadd_2x64_jit_wrapper",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value_helpers",
        "//xls/ir:value_view_helpers",
        "//xls/tools:testbench",
    ],
)

xls_dslx_opt_ir(
    name = "fpadd_2xbf16",
    src = "fpadd_2xbf16.x",
    deps = [
        ":apfloat_add_2_dslx",
    ],
)

xls_dslx_opt_ir_test(
    name = "fpadd_2xbf16_test",
    dep = ":fpadd_2xbf16",
    # 2021-03-08 Takes too long (> 15 minutes).
    prove_unopt_eq_opt = False,
)

xls_dslx_opt_ir(
    name = "fpldexp_32",
    src = "fpldexp_32.x",
    ir_file = "fpldexp_32.ir",
    opt_ir_file = "fpldexp_32.opt.ir",
)

xls_dslx_opt_ir_test(
    name = "fpldexp_32_test",
    dep = ":fpldexp_32",
)

cc_xls_ir_jit_wrapper(
    name = "fpldexp_32_jit_wrapper",
    src = "fpldexp_32.opt.ir",
    jit_wrapper_args = {
        "class_name": "fpldexp_32",
    },
)

cc_test(
    name = "fpldexp_32_test_cc",
    srcs = ["fpldexp_32_test.cc"],
    data = [
        ":fpldexp_32.ir",
        ":fpldexp_32.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":fpldexp_32_jit_wrapper",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value_helpers",
        "//xls/ir:value_view_helpers",
        "//xls/tools:testbench",
    ],
)

xls_dslx_library(
    name = "apfloat_mul_2_dslx",
    srcs = ["apfloat_mul_2.x"],
)

xls_dslx_opt_ir(
    name = "fpmul_2x32",
    src = "fpmul_2x32.x",
    ir_file = "fpmul_2x32.ir",
    opt_ir_file = "fpmul_2x32.opt.ir",
    deps = [":apfloat_mul_2_dslx"],
)

xls_dslx_opt_ir_test(
    name = "fpmul_2x32_test",
    dep = ":fpmul_2x32",
    # 2021-03-08: Takes too long (> 15 minutes).
    prove_unopt_eq_opt = False,
)

cc_xls_ir_jit_wrapper(
    name = "fpmul_2x32_jit_wrapper",
    src = "fpmul_2x32.opt.ir",
    jit_wrapper_args = {
        "class_name": "fpmul_2x32",
    },
)

cc_test(
    name = "fpmul_2x32_test_cc",
    srcs = ["fpmul_2x32_test.cc"],
    data = [
        ":fpmul_2x32.ir",
        ":fpmul_2x32.opt.ir",
    ],
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
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:value",
        "//xls/ir:value_helpers",
        "@com_google_googletest//:gtest",
    ],
)

xls_dslx_opt_ir(
    name = "fpmul_2x64",
    src = "fpmul_2x64.x",
    ir_file = "fpmul_2x64.ir",
    opt_ir_file = "fpmul_2x64.opt.ir",
    deps = [":apfloat_mul_2_dslx"],
)

xls_dslx_opt_ir_test(
    name = "fpmul_2x64_test",
    dep = ":fpmul_2x64",
    # TODO(rspringer): Currently takes too long.
    prove_unopt_eq_opt = False,
)

cc_xls_ir_jit_wrapper(
    name = "fpmul_2x64_jit_wrapper",
    src = "fpmul_2x64.opt.ir",
    jit_wrapper_args = {
        "class_name": "fpmul_2x64",
    },
)

cc_test(
    name = "fpmul_2x64_test_cc",
    srcs = ["fpmul_2x64_test.cc"],
    data = [
        ":fpmul_2x64.ir",
        ":fpmul_2x64.opt.ir",
    ],
    deps = [
        ":fpmul_2x64_jit_wrapper",
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

xls_dslx_library(
    name = "fpmul_2x32_dslx",
    srcs = ["fpmul_2x32.x"],
    deps = [
        ":apfloat_mul_2_dslx",
    ],
)

xls_dslx_opt_ir(
    name = "fp_fast_rsqrt_32",
    src = "fp_fast_rsqrt_32.x",
    ir_file = "fp_fast_rsqrt_32.ir",
    opt_ir_file = "fp_fast_rsqrt_32.opt.ir",
    deps = [
        ":apfloat_add_2_dslx",
        ":apfloat_mul_2_dslx",
        ":fpadd_2x32_dslx",
        ":fpmul_2x32_dslx",
    ],
)

xls_dslx_opt_ir_test(
    name = "fp_fast_rsqrt_32_test",
    dep = ":fp_fast_rsqrt_32",
    # TODO(rspringer): Currently takes too long.
    prove_unopt_eq_opt = False,
)

cc_xls_ir_jit_wrapper(
    name = "fp_fast_rsqrt_32_jit_wrapper",
    src = "fp_fast_rsqrt_32.opt.ir",
    jit_wrapper_args = {
        "class_name": "fp_fast_rsqrt_32",
    },
)

cc_test(
    name = "fp_fast_rsqrt_32_test_cc",
    srcs = ["fp_fast_rsqrt_32_test.cc"],
    data = [
        ":fp_fast_rsqrt_32.ir",
        ":fp_fast_rsqrt_32.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":fp_fast_rsqrt_32_jit_wrapper",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value_helpers",
        "//xls/ir:value_view_helpers",
        "//xls/tools:testbench",
    ],
)

exports_files(glob(["*.x"]))
