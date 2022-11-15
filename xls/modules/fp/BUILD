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
    "xls_benchmark_ir",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_opt_ir_test",
    "xls_dslx_test",
    "xls_eval_ir_test",
    "xls_ir_cc_library",
)

package(
    default_visibility = ["//xls:xls_users"],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_library(
    name = "apfloat_add_2_dslx",
    srcs = ["apfloat_add_2.x"],
)

xls_dslx_library(
    name = "apfloat_sub_2_dslx",
    srcs = ["apfloat_sub_2.x"],
    deps = [":apfloat_add_2_dslx"],
)

xls_dslx_library(
    name = "fp32_add_2_dslx",
    srcs = ["fp32_add_2.x"],
    deps = [":apfloat_add_2_dslx"],
)

xls_ir_cc_library(
    name = "fp32_add_2_cc",
    src = ":fp32_add_2.opt.ir",
    namespaces = "xls,fp",
    top = "fp32_add_2",
)

xls_dslx_library(
    name = "fp32_sub_2_dslx",
    srcs = ["fp32_sub_2.x"],
    deps = [
        ":apfloat_add_2_dslx",
        ":apfloat_sub_2_dslx",
        ":fp32_add_2_dslx",
    ],
)

xls_dslx_library(
    name = "fp32_ldexp_dslx",
    srcs = ["fp32_ldexp.x"],
)

xls_dslx_test(
    name = "apfloat_sub_2_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = ":apfloat_sub_2_dslx",
)

filegroup(
    name = "ir_examples",
    srcs = [
        ":fp32_add_2.ir",
        ":fp32_add_2.opt.ir",
        ":fp32_mul_2.ir",
        ":fp32_mul_2.opt.ir",
        ":fp64_add_2.ir",
        ":fp64_add_2.opt.ir",
        ":fp64_mul_2.ir",
        ":fp64_mul_2.opt.ir",
    ],
)

xls_dslx_library(
    name = "apfloat_fma_dslx",
    srcs = ["apfloat_fma.x"],
)

# TODO(rspringer): 2021-04-09: Make a macro to handle creating the three
# related rules. Consider also doing this for multiple widths (32, 64, bf16).
xls_dslx_library(
    name = "fp32_fma_dslx",
    srcs = ["fp32_fma.x"],
    deps = [":apfloat_fma_dslx"],
)

xls_dslx_opt_ir(
    name = "fp32_fma",
    dslx_top = "fp32_fma",
    library = ":fp32_fma_dslx",
)

xls_dslx_test(
    name = "fp32_fma_dslx_test",
    library = ":fp32_fma_dslx",
)

xls_ir_cc_library(
    name = "fp32_fma_cc",
    src = ":fp32_fma.opt.ir",
    namespaces = "xls,fp",
    top = "fp32_fma_cc",
)

xls_dslx_library(
    name = "apfloat_fmac_dslx",
    srcs = ["apfloat_fmac.x"],
    deps = [
        ":apfloat_fma_dslx",
    ],
)

xls_dslx_library(
    name = "fp32_fmac_dslx",
    srcs = ["fp32_fmac.x"],
    deps = [":apfloat_fmac_dslx"],
)

xls_dslx_opt_ir(
    name = "fp32_fmac",
    dslx_top = "fp32_fmac",
    library = ":fp32_fmac_dslx",
)

xls_dslx_test(
    name = "fp32_fmac_test",
    dslx_test_args = {
        "compare": "none",
    },
    library = ":fp32_fmac_dslx",
)

# TODO(b/152546795): 2021-03-08 Takes too long (~5 minutes). Add
# xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "fp32_fma_eval_ir_test",
    src = ":fp32_fma.ir",
)

xls_benchmark_ir(
    name = "fp32_fma_benchmark_ir",
    src = ":fp32_fma.ir",
)

cc_xls_ir_jit_wrapper(
    name = "fp32_fma_jit_wrapper",
    src = ":fp32_fma.opt.ir",
    jit_wrapper_args = {
        "class_name": "fp32_fma",
        "namespace": "xls::fp",
    },
)

cc_test(
    name = "fp32_fma_test",
    srcs = ["fp32_fma_test.cc"],
    data = [
        ":fp32_fma.ir",
        ":fp32_fma.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":fp32_fma_jit_wrapper",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/common/logging",
        "//xls/tools:testbench",
        "//xls/tools:testbench_builder",
    ],
)

xls_dslx_library(
    name = "fp64_fma_dslx",
    srcs = ["fp64_fma.x"],
    deps = [":apfloat_fma_dslx"],
)

xls_dslx_opt_ir(
    name = "fp64_fma",
    dslx_top = "fp64_fma",
    ir_file = "fp64_fma.ir",
    library = ":fp64_fma_dslx",
    opt_ir_file = "fp64_fma.opt.ir",
)

xls_dslx_test(
    name = "fp64_fma_dslx_test",
    library = ":fp64_fma_dslx",
)

# TODO(b/152546795): 2021-03-08 Takes too long (~5 minutes). Add
# xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "fp64_fma_eval_ir_test",
    src = ":fp64_fma.ir",
)

xls_benchmark_ir(
    name = "fp64_fma_benchmark_ir",
    src = ":fp64_fma.ir",
)

cc_xls_ir_jit_wrapper(
    name = "fp64_fma_jit_wrapper",
    src = ":fp64_fma.opt.ir",
    jit_wrapper_args = {
        "class_name": "fp64_fma",
        "namespace": "xls::fp",
    },
)

cc_test(
    name = "fp64_fma_test",
    srcs = ["fp64_fma_test.cc"],
    data = [
        ":fp64_fma.ir",
        ":fp64_fma.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":fp64_fma_jit_wrapper",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/common/logging",
        "//xls/tools:testbench",
        "//xls/tools:testbench_builder",
    ],
)

xls_dslx_opt_ir(
    name = "fp32_add_2",
    dslx_top = "fp32_add_2",
    ir_file = "fp32_add_2.ir",
    library = ":fp32_add_2_dslx",
    opt_ir_file = "fp32_add_2.opt.ir",
)

xls_dslx_test(
    name = "fp32_add_2_dslx_test",
    library = ":fp32_add_2_dslx",
)

# TODO(b/152546795): 2021-03-08 Takes too long (~5 minutes). Add
# xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "fp32_add_2_eval_ir_test",
    src = ":fp32_add_2.ir",
)

xls_benchmark_ir(
    name = "fp32_add_2_benchmark_ir",
    src = ":fp32_add_2.ir",
)

xls_dslx_opt_ir(
    name = "fp32_sub_2",
    dslx_top = "fp32_sub_2",
    ir_file = "fp32_sub_2.ir",
    library = ":fp32_sub_2_dslx",
)

xls_dslx_test(
    name = "fp32_sub_2_dslx_test",
    library = ":fp32_sub_2_dslx",
)

# TODO(b/152546795): 2021-03-08 Takes too long (~5 minutes). Add
# xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "fp32_sub_2_eval_ir_test",
    src = ":fp32_sub_2.ir",
)

xls_benchmark_ir(
    name = "fp32_sub_2_benchmark_ir",
    src = ":fp32_sub_2.ir",
)

xls_dslx_library(
    name = "fp64_sub_2_dslx",
    srcs = ["fp64_sub_2.x"],
    deps = [
        ":apfloat_sub_2_dslx",
        ":fp64_add_2_dslx",
    ],
)

xls_dslx_opt_ir(
    name = "fp64_sub_2",
    dslx_top = "fp64_sub_2",
    ir_file = "fp64_sub_2.ir",
    library = ":fp64_sub_2_dslx",
)

xls_dslx_test(
    name = "fp64_sub_2_dslx_test",
    library = ":fp64_sub_2_dslx",
)

# TODO(b/152546795): 2021-03-08 Takes too long (~5 minutes). Add
# xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "fp64_sub_2_eval_ir_test",
    src = ":fp64_sub_2.ir",
)

xls_benchmark_ir(
    name = "fp64_sub_2_benchmark_ir",
    src = ":fp64_sub_2.ir",
)

cc_xls_ir_jit_wrapper(
    name = "fp32_add_2_jit_wrapper",
    src = ":fp32_add_2.opt.ir",
    jit_wrapper_args = {
        "class_name": "fp32_add_2",
        "namespace": "xls::fp",
    },
)

# TODO(rspringer): Takes too long to run in normal testing.
cc_binary(
    name = "fp32_add_2_bounds",
    srcs = ["fp32_add_2_bounds.cc"],
    data = [
        ":fp32_add_2.ir",
        ":fp32_add_2.opt.ir",
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
    name = "fp32_add_2_test_cc",
    srcs = ["fp32_add_2_test.cc"],
    tags = ["optonly"],
    deps = [
        ":fp32_add_2_jit_wrapper",
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
        "//xls/tools:testbench_builder",
    ],
)

xls_dslx_library(
    name = "fp64_add_2_dslx",
    srcs = ["fp64_add_2.x"],
    deps = [":apfloat_add_2_dslx"],
)

xls_dslx_opt_ir(
    name = "fp64_add_2",
    dslx_top = "fp64_add_2",
    ir_file = "fp64_add_2.ir",
    library = "fp64_add_2_dslx",
    opt_ir_file = "fp64_add_2.opt.ir",
)

xls_dslx_test(
    name = "fp64_add_2_dslx_test",
    library = "fp64_add_2_dslx",
)

# TODO: 2021-03-08 Takes too long (> 15 minutes). Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "fp64_add_2_eval_ir_test",
    src = ":fp64_add_2.ir",
)

xls_benchmark_ir(
    name = "fp64_add_2_benchmark_ir",
    src = ":fp64_add_2.ir",
)

cc_xls_ir_jit_wrapper(
    name = "fp64_add_2_jit_wrapper",
    src = ":fp64_add_2.opt.ir",
    jit_wrapper_args = {
        "class_name": "fp64_add_2",
        "namespace": "xls::fp",
    },
)

cc_test(
    name = "fp64_add_2_test_cc",
    srcs = ["fp64_add_2_test.cc"],
    data = [
        ":fp64_add_2.ir",
        ":fp64_add_2.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":fp64_add_2_jit_wrapper",
        "@com_google_absl//absl/flags:flag",
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
        "//xls/tools:testbench_builder",
    ],
)

xls_dslx_library(
    name = "bf16_add_2_dslx",
    srcs = ["bf16_add_2.x"],
    deps = [
        ":apfloat_add_2_dslx",
    ],
)

xls_dslx_opt_ir(
    name = "bf16_add_2",
    dslx_top = "bf16_add_2",
    ir_file = "bf16_add_2.ir",
    library = "bf16_add_2_dslx",
)

xls_dslx_test(
    name = "bf16_add_2_dslx_test",
    library = "bf16_add_2_dslx",
)

# TODO: 2021-03-08 Takes too long (> 15 minutes). Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "bf16_add_2_eval_ir_test",
    src = ":bf16_add_2.ir",
)

xls_benchmark_ir(
    name = "bf16_add_2_benchmark_ir",
    src = ":bf16_add_2.ir",
)

xls_dslx_library(
    name = "bf16_mul_2_dslx",
    srcs = ["bf16_mul_2.x"],
    deps = [
        ":apfloat_mul_2_dslx",
    ],
)

xls_dslx_opt_ir(
    name = "bf16_mul_2",
    dslx_top = "bf16_mul_2",
    ir_file = "bf16_mul_2.ir",
    library = "bf16_mul_2_dslx",
)

xls_dslx_test(
    name = "bf16_mul_2_dslx_test",
    library = "bf16_mul_2_dslx",
)

xls_dslx_opt_ir(
    name = "fp32_ldexp",
    srcs = ["fp32_ldexp.x"],
    dslx_top = "fp32_ldexp",
    ir_file = "fp32_ldexp.ir",
    opt_ir_file = "fp32_ldexp.opt.ir",
)

xls_dslx_opt_ir_test(
    name = "fp32_ldexp_test",
    dep = ":fp32_ldexp",
)

cc_xls_ir_jit_wrapper(
    name = "fp32_ldexp_jit_wrapper",
    src = ":fp32_ldexp.opt.ir",
    jit_wrapper_args = {
        "class_name": "fp32_ldexp",
        "namespace": "xls::fp",
    },
)

cc_test(
    name = "fp32_ldexp_test_cc",
    srcs = ["fp32_ldexp_test.cc"],
    data = [
        ":fp32_ldexp.ir",
        ":fp32_ldexp.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":fp32_ldexp_jit_wrapper",
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
        "//xls/tools:testbench_builder",
    ],
)

xls_dslx_library(
    name = "apfloat_mul_2_dslx",
    srcs = ["apfloat_mul_2.x"],
)

xls_dslx_library(
    name = "fp32_mul_2_dslx",
    srcs = ["fp32_mul_2.x"],
    deps = [":apfloat_mul_2_dslx"],
)

xls_dslx_opt_ir(
    name = "fp32_mul_2",
    dslx_top = "fp32_mul_2",
    ir_file = "fp32_mul_2.ir",
    library = ":fp32_mul_2_dslx",
    opt_ir_file = "fp32_mul_2.opt.ir",
)

xls_dslx_test(
    name = "fp32_mul_2_dslx_test",
    library = ":fp32_mul_2_dslx",
)

# TODO: 2021-03-08 Takes too long (> 15 minutes). Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "fp32_mul_2_eval_ir_test",
    src = ":fp32_mul_2.ir",
)

xls_benchmark_ir(
    name = "fp32_mul_2_benchmark_ir",
    src = ":fp32_mul_2.ir",
)

cc_xls_ir_jit_wrapper(
    name = "fp32_mul_2_jit_wrapper",
    src = ":fp32_mul_2.opt.ir",
    jit_wrapper_args = {
        "class_name": "fp32_mul_2",
        "namespace": "xls::fp",
    },
)

cc_test(
    name = "fp32_mul_2_test_cc",
    srcs = ["fp32_mul_2_test.cc"],
    data = [
        ":fp32_mul_2.ir",
        ":fp32_mul_2.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":fp32_mul_2_jit_wrapper",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value_helpers",
        "//xls/ir:value_view_helpers",
        "//xls/tools:testbench",
        "//xls/tools:testbench_builder",
    ],
)

cc_test(
    name = "fp32_add_2_jit_wrapper_test",
    srcs = ["fp32_add_2_jit_wrapper_test.cc"],
    deps = [
        ":fp32_add_2_jit_wrapper",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:value",
        "//xls/ir:value_helpers",
        "@com_google_googletest//:gtest",
    ],
)

xls_dslx_library(
    name = "fp64_mul_2_dslx",
    srcs = ["fp64_mul_2.x"],
    deps = [":apfloat_mul_2_dslx"],
)

xls_dslx_opt_ir(
    name = "fp64_mul_2",
    dslx_top = "fp64_mul_2",
    ir_file = "fp64_mul_2.ir",
    library = ":fp64_mul_2_dslx",
    opt_ir_file = "fp64_mul_2.opt.ir",
)

xls_dslx_test(
    name = "fp64_mul_2_dslx_test",
    library = ":fp64_mul_2_dslx",
)

# TODO(rspringer): Currently takes too long. Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "fp64_mul_2_eval_ir_test",
    src = ":fp64_mul_2.ir",
)

xls_benchmark_ir(
    name = "fp64_mul_2_benchmark_ir",
    src = ":fp64_mul_2.ir",
)

cc_xls_ir_jit_wrapper(
    name = "fp64_mul_2_jit_wrapper",
    src = ":fp64_mul_2.opt.ir",
    jit_wrapper_args = {
        "class_name": "fp64_mul_2",
        "namespace": "xls::fp",
    },
)

cc_test(
    name = "fp64_mul_2_test",
    srcs = ["fp64_mul_2_test.cc"],
    data = [
        ":fp64_mul_2.ir",
        ":fp64_mul_2.opt.ir",
    ],
    # Flakes in fastbuild seen on 2021-06-04.
    tags = ["optonly"],
    deps = [
        ":fp64_mul_2_jit_wrapper",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value_helpers",
        "//xls/ir:value_view_helpers",
        "//xls/tools:testbench",
        "//xls/tools:testbench_builder",
    ],
)

xls_dslx_library(
    name = "fp32_fast_rsqrt_dslx",
    srcs = ["fp32_fast_rsqrt.x"],
    deps = [
        ":apfloat_add_2_dslx",
        ":apfloat_mul_2_dslx",
        ":fp32_add_2_dslx",
        ":fp32_mul_2_dslx",
    ],
)

xls_dslx_opt_ir(
    name = "fp32_fast_rsqrt",
    dslx_top = "fp32_fast_rsqrt",
    ir_file = "fp32_fast_rsqrt.ir",
    library = ":fp32_fast_rsqrt_dslx",
    opt_ir_file = "fp32_fast_rsqrt.opt.ir",
)

xls_dslx_test(
    name = "fp32_fast_rsqrt_dslx_test",
    library = ":fp32_fast_rsqrt_dslx",
)

# TODO(rspringer): Currently takes too long. Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "fp32_fast_rsqrt_eval_ir_test",
    src = ":fp32_fast_rsqrt.ir",
)

xls_benchmark_ir(
    name = "fp32_fast_rsqrt_benchmark_ir",
    src = ":fp32_fast_rsqrt.ir",
)

cc_xls_ir_jit_wrapper(
    name = "fp32_fast_rsqrt_jit_wrapper",
    src = ":fp32_fast_rsqrt.opt.ir",
    jit_wrapper_args = {
        "class_name": "fp32_fast_rsqrt",
        "namespace": "xls::fp",
    },
)

cc_test(
    name = "fp32_fast_rsqrt_test_cc",
    srcs = ["fp32_fast_rsqrt_test.cc"],
    data = [
        ":fp32_fast_rsqrt.ir",
        ":fp32_fast_rsqrt.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":fp32_fast_rsqrt_jit_wrapper",
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
        "//xls/tools:testbench_builder",
    ],
)
