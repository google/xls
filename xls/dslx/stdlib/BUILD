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

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "cc_xls_ir_jit_wrapper",
    "get_mangled_ir_symbol",
    "xls_benchmark_ir",
    "xls_dslx_fmt_test",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_prove_quickcheck_test",
    "xls_dslx_test",
    "xls_eval_ir_test",
    "xls_ir_cc_library",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_users"],
    licenses = ["notice"],  # Apache 2.0
)

xls_dslx_library(
    name = "std_dslx",
    srcs = ["std.x"],
)

xls_dslx_fmt_test(
    name = "std_dslx_fmt_test",
    src = "std.x",
    opportunistic_postcondition = True,
)

xls_dslx_test(
    name = "std_dslx_test",
    srcs = ["std.x"],
)

xls_dslx_prove_quickcheck_test(
    name = "std_dslx_prove_quickcheck_test",
    srcs = ["std.x"],
)

xls_dslx_test(
    name = "acm_random_dslx_test",
    srcs = ["acm_random.x"],
)

xls_dslx_fmt_test(
    name = "acm_random_dslx_fmt_test",
    src = "acm_random.x",
    opportunistic_postcondition = True,
)

xls_dslx_library(
    name = "apfloat_dslx",
    srcs = ["apfloat.x"],
    deps = [":std_dslx"],
)

xls_dslx_test(
    name = "apfloat_dslx_test",
    srcs = ["apfloat.x"],
    deps = [":apfloat_dslx"],
)

xls_dslx_prove_quickcheck_test(
    name = "apfloat_dslx_prove_quickcheck_test",
    srcs = ["apfloat.x"],
    deps = [":apfloat_dslx"],
)

xls_dslx_fmt_test(
    name = "apfloat_dslx_fmt_test",
    src = "apfloat.x",
    opportunistic_postcondition = True,
)

xls_dslx_library(
    name = "bfloat16_dslx",
    srcs = ["bfloat16.x"],
    deps = [":std_dslx"],
)

xls_dslx_fmt_test(
    name = "bfloat16_dslx_fmt_test",
    src = "bfloat16.x",
    opportunistic_postcondition = True,
)

xls_dslx_test(
    name = "bfloat16_dslx_test",
    srcs = ["bfloat16.x"],
    deps = [":bfloat16_dslx"],
)

xls_dslx_opt_ir(
    name = "bfloat16_add",
    dslx_top = "add",
    ir_file = "bfloat16_add.ir",
    library = "bfloat16_dslx",
)

# TODO: 2021-03-08 Takes too long (> 15 minutes). Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "bfloat16_add_eval_ir_test",
    src = ":bfloat16_add.ir",
)

xls_benchmark_ir(
    name = "bfloat16_add_benchmark_ir",
    src = ":bfloat16_add.ir",
)

xls_dslx_opt_ir(
    name = "bfloat16_sub",
    dslx_top = "sub",
    ir_file = "bfloat16_sub.ir",
    library = "bfloat16_dslx",
)

# TODO: 2021-03-08 Takes too long (> 15 minutes). Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "bfloat16_sub_eval_ir_test",
    src = ":bfloat16_sub.ir",
)

xls_benchmark_ir(
    name = "bfloat16_sub_benchmark_ir",
    src = ":bfloat16_sub.ir",
)

xls_dslx_opt_ir(
    name = "bfloat16_mul",
    dslx_top = "mul",
    ir_file = "bfloat16_mul.ir",
    library = "bfloat16_dslx",
)

xls_dslx_library(
    name = "float32_dslx",
    srcs = ["float32.x"],
    deps = [":std_dslx"],
)

xls_dslx_fmt_test(
    name = "float32_dslx_fmt_test",
    src = "float32.x",
    opportunistic_postcondition = True,
)

xls_dslx_test(
    name = "float32_dslx_test",
    srcs = ["float32.x"],
    deps = [":float32_dslx"],
)

xls_dslx_opt_ir(
    name = "float32_add",
    dslx_top = "add",
    ir_file = "float32_add.ir",
    library = ":float32_dslx",
    opt_ir_file = "float32_add.opt.ir",
)

xls_ir_cc_library(
    name = "float32_add_cc",
    src = ":float32_add.opt.ir",
    namespaces = "xls,fp",
)

# TODO(b/152546795): 2021-03-08 Takes too long (~5 minutes). Add
# xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "float32_add_eval_ir_test",
    src = ":float32_add.ir",
)

xls_benchmark_ir(
    name = "float32_add_benchmark_ir",
    src = ":float32_add.ir",
)

cc_xls_ir_jit_wrapper(
    name = "float32_add_jit_wrapper",
    src = ":float32_add.opt.ir",
    jit_wrapper_args = {
        "class_name": "Float32Add",
        "namespace": "xls::fp",
    },
)

cc_test(
    name = "float32_add_jit_wrapper_test",
    srcs = ["float32_add_jit_wrapper_test.cc"],
    deps = [
        ":float32_add_jit_wrapper",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "float32_add_test_cc",
    srcs = ["float32_add_test.cc"],
    tags = ["optonly"],
    deps = [
        ":float32_add_jit_wrapper",
        ":float32_test_utils",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/common/status:status_macros",
        "//xls/ir:value_utils",
        "//xls/ir:value_view_utils",
        "//xls/tools:testbench",
        "//xls/tools:testbench_builder",
    ],
)

xls_dslx_opt_ir(
    name = "float32",
    dslx_top = "sub",
    ir_file = "float32_sub.ir",
    library = ":float32_dslx",
    opt_ir_file = "float32_sub.opt.ir",
)

# TODO(b/152546795): 2021-03-08 Takes too long (~5 minutes). Add
# xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "float32_sub_eval_ir_test",
    src = ":float32_sub.ir",
)

xls_benchmark_ir(
    name = "float32_sub_benchmark_ir",
    src = ":float32_sub.ir",
)

xls_dslx_opt_ir(
    name = "float32_mul",
    dslx_top = "mul",
    ir_file = "float32_mul.ir",
    library = ":float32_dslx",
    opt_ir_file = "float32_mul.opt.ir",
)

# TODO: 2021-03-08 Takes too long (> 15 minutes). Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "float32_mul_eval_ir_test",
    src = ":float32_mul.ir",
)

xls_benchmark_ir(
    name = "float32_mul_benchmark_ir",
    src = ":float32_mul.ir",
)

cc_xls_ir_jit_wrapper(
    name = "float32_mul_jit_wrapper",
    src = ":float32_mul.opt.ir",
    jit_wrapper_args = {
        "class_name": "Float32Mul",
        "namespace": "xls::fp",
    },
)

cc_test(
    name = "float32_mul_test_cc",
    srcs = ["float32_mul_test.cc"],
    data = [
        ":float32_mul.ir",
        ":float32_mul.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":float32_mul_jit_wrapper",
        ":float32_test_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/tools:testbench",
        "//xls/tools:testbench_builder",
    ],
)

xls_dslx_test(
    name = "float32_upcast_test",
    srcs = ["float32_upcast_test.x"],
)

xls_dslx_opt_ir(
    name = "float32_upcast",
    srcs = ["float32_upcast_test.x"],
    dslx_top = "f32_to_f64",
    ir_file = "float32_upcast.ir",
    opt_ir_file = "float32_upcast.opt.ir",
)

cc_xls_ir_jit_wrapper(
    name = "float32_upcast_jit_wrapper",
    src = ":float32_upcast.opt.ir",
    jit_wrapper_args = {
        "class_name": "F32ToF64",
        "namespace": "xls::fp",
    },
)

cc_test(
    name = "float32_upcast_test_cc",
    srcs = ["float32_upcast_test.cc"],
    data = [
        ":float32_upcast.ir",
        ":float32_upcast.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":float32_test_utils",
        ":float32_upcast_jit_wrapper",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:status_macros",
        "//xls/ir:value_utils",
        "//xls/ir:value_view_utils",
        "//xls/tools:testbench",
        "//xls/tools:testbench_builder",
    ],
)

xls_dslx_opt_ir(
    name = "float32_to_int32",
    srcs = ["float32.x"],
    dslx_top = "to_int32",
    deps = [
        ":apfloat_dslx",
    ],
)

cc_xls_ir_jit_wrapper(
    name = "float32_to_int32_wrapper",
    src = ":float32_to_int32.opt.ir",
    jit_wrapper_args = {
        "class_name": "Float32ToInt32",
        "function": get_mangled_ir_symbol("float32", "to_int32"),
        "namespace": "xls::dslx",
    },
)

xls_dslx_opt_ir(
    name = "float32_from_int32",
    srcs = ["float32.x"],
    dslx_top = "from_int32",
    deps = [
        ":apfloat_dslx",
    ],
)

cc_xls_ir_jit_wrapper(
    name = "float32_from_int32_wrapper",
    src = ":float32_from_int32.opt.ir",
    jit_wrapper_args = {
        "class_name": "Float32FromInt32",
        "function": get_mangled_ir_symbol("float32", "from_int32"),
        "namespace": "xls::dslx",
    },
)

cc_test(
    name = "float32_test",
    srcs = ["float32_test.cc"],
    tags = ["optonly"],
    deps = [
        ":float32_from_int32_wrapper",
        ":float32_to_int32_wrapper",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "@com_google_googletest//:gtest",
    ],
)

# TODO(rspringer): Takes too long to run in normal testing.
cc_binary(
    name = "float32_add_bounds",
    srcs = ["float32_add_bounds.cc"],
    data = [
        ":float32_add.ir",
        ":float32_add.opt.ir",
    ],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:status_macros",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/solvers:z3_ir_translator",
        "//xls/solvers:z3_utils",
        "@z3//:api",
    ],
)

xls_dslx_opt_ir(
    name = "float32_fma",
    dslx_top = "fma",
    library = ":float32_dslx",
    opt_ir_file = "float32_fma.opt.ir",
)

xls_ir_cc_library(
    name = "float32_fma_cc",
    src = ":float32_fma.opt.ir",
    namespaces = "xls,fp",
)

# TODO(b/152546795): 2021-03-08 Takes too long (~5 minutes). Add
# xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "float32_fma_eval_ir_test",
    src = ":float32_fma.ir",
)

xls_benchmark_ir(
    name = "float32_fma_benchmark_ir",
    src = ":float32_fma.ir",
)

cc_xls_ir_jit_wrapper(
    name = "float32_fma_jit_wrapper",
    src = ":float32_fma.opt.ir",
    jit_wrapper_args = {
        "class_name": "Float32Fma",
        "namespace": "xls::fp",
    },
)

cc_test(
    name = "float32_fma_test",
    srcs = ["float32_fma_test.cc"],
    data = [
        ":float32_fma.ir",
        ":float32_fma.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":float32_fma_jit_wrapper",
        ":float32_test_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/tools:testbench",
        "//xls/tools:testbench_builder",
    ],
)

xls_dslx_opt_ir(
    name = "float32_ldexp",
    dslx_top = "ldexp",
    ir_file = "float32_ldexp.ir",
    library = ":float32_dslx",
    opt_ir_file = "float32_ldexp.opt.ir",
)

cc_xls_ir_jit_wrapper(
    name = "float32_ldexp_jit_wrapper",
    src = ":float32_ldexp.opt.ir",
    jit_wrapper_args = {
        "class_name": "Float32Ldexp",
        "namespace": "xls::fp",
    },
)

cc_test(
    name = "float32_ldexp_test_cc",
    srcs = ["float32_ldexp_test.cc"],
    data = [
        ":float32_ldexp.ir",
        ":float32_ldexp.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":float32_ldexp_jit_wrapper",
        ":float32_test_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/common/status:status_macros",
        "//xls/tools:testbench",
        "//xls/tools:testbench_builder",
    ],
)

xls_dslx_opt_ir(
    name = "float32_fast_rsqrt",
    dslx_top = "fast_rsqrt",
    ir_file = "float32_fast_rsqrt.ir",
    library = ":float32_dslx",
    opt_ir_file = "float32_fast_rsqrt.opt.ir",
)

# TODO(rspringer): Currently takes too long. Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "float32_fast_rsqrt_eval_ir_test",
    src = ":float32_fast_rsqrt.ir",
)

xls_benchmark_ir(
    name = "float32_fast_rsqrt_benchmark_ir",
    src = ":float32_fast_rsqrt.ir",
)

cc_xls_ir_jit_wrapper(
    name = "float32_fast_rsqrt_jit_wrapper",
    src = ":float32_fast_rsqrt.opt.ir",
    jit_wrapper_args = {
        "class_name": "Float32FastRsqrt",
        "namespace": "xls::fp",
    },
)

cc_test(
    name = "float32_fast_rsqrt_test_cc",
    srcs = ["float32_fast_rsqrt_test.cc"],
    data = [
        ":float32_fast_rsqrt.ir",
        ":float32_fast_rsqrt.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":float32_fast_rsqrt_jit_wrapper",
        ":float32_test_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/tools:testbench",
        "//xls/tools:testbench_builder",
    ],
)

cc_library(
    name = "float32_test_utils",
    testonly = True,
    hdrs = ["float32_test_utils.h"],
    deps = [
        "@com_google_absl//absl/base",
    ],
)

xls_dslx_library(
    name = "float64_dslx",
    srcs = ["float64.x"],
    deps = [":std_dslx"],
)

xls_dslx_fmt_test(
    name = "float64_dslx_fmt_test",
    src = "float64.x",
    opportunistic_postcondition = True,
)

xls_dslx_test(
    name = "float64_dslx_test",
    srcs = ["float64.x"],
    deps = [":float64_dslx"],
)

xls_dslx_opt_ir(
    name = "float64_add",
    dslx_top = "add",
    ir_file = "float64_add.ir",
    library = "float64_dslx",
    opt_ir_file = "float64_add.opt.ir",
)

# TODO: 2021-03-08 Takes too long (> 15 minutes). Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "float64_add_eval_ir_test",
    src = ":float64_add.ir",
)

xls_benchmark_ir(
    name = "float64_add_benchmark_ir",
    src = ":float64_add.ir",
)

cc_xls_ir_jit_wrapper(
    name = "float64_add_jit_wrapper",
    src = ":float64_add.opt.ir",
    jit_wrapper_args = {
        "class_name": "Float64Add",
        "namespace": "xls::fp",
    },
)

cc_test(
    name = "float64_add_test_cc",
    srcs = ["float64_add_test.cc"],
    data = [
        ":float64_add.ir",
        ":float64_add.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":float64_add_jit_wrapper",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/common/status:status_macros",
        "//xls/ir:value_utils",
        "//xls/ir:value_view_utils",
        "//xls/tools:testbench",
        "//xls/tools:testbench_builder",
    ],
)

xls_dslx_opt_ir(
    name = "float64_sub",
    dslx_top = "sub",
    ir_file = "float64_sub.ir",
    library = ":float64_dslx",
    opt_ir_file = "float64_sub.opt.ir",
)

# TODO(b/152546795): 2021-03-08 Takes too long (~5 minutes). Add
# xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "float64_sub_eval_ir_test",
    src = ":float64_sub.ir",
)

xls_benchmark_ir(
    name = "float64_sub_benchmark_ir",
    src = ":float64_sub.ir",
)

xls_dslx_opt_ir(
    name = "float64_mul",
    dslx_top = "mul",
    ir_file = "float64_mul.ir",
    library = ":float64_dslx",
    opt_ir_file = "float64_mul.opt.ir",
)

# TODO(rspringer): Currently takes too long. Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "float64_mul_eval_ir_test",
    src = ":float64_mul.ir",
)

xls_benchmark_ir(
    name = "float64_mul_benchmark_ir",
    src = ":float64_mul.ir",
)

cc_xls_ir_jit_wrapper(
    name = "float64_mul_jit_wrapper",
    src = ":float64_mul.opt.ir",
    jit_wrapper_args = {
        "class_name": "Float64Mul",
        "namespace": "xls::fp",
    },
)

cc_test(
    name = "float64_mul_test",
    srcs = ["float64_mul_test.cc"],
    data = [
        ":float64_mul.ir",
        ":float64_mul.opt.ir",
    ],
    # Flakes in fastbuild seen on 2021-06-04.
    tags = ["optonly"],
    deps = [
        ":float64_mul_jit_wrapper",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/tools:testbench",
        "//xls/tools:testbench_builder",
    ],
)

xls_dslx_opt_ir(
    name = "float64_fma",
    dslx_top = "fma",
    ir_file = "float64_fma.ir",
    library = ":float64_dslx",
    opt_ir_file = "float64_fma.opt.ir",
)

# TODO(b/152546795): 2021-03-08 Takes too long (~5 minutes). Add
# xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "float64_fma_eval_ir_test",
    src = ":float64_fma.ir",
)

xls_benchmark_ir(
    name = "float64_fma_benchmark_ir",
    src = ":float64_fma.ir",
)

cc_xls_ir_jit_wrapper(
    name = "float64_fma_jit_wrapper",
    src = ":float64_fma.opt.ir",
    jit_wrapper_args = {
        "class_name": "Float64Fma",
        "namespace": "xls::fp",
    },
)

cc_test(
    name = "float64_fma_test",
    srcs = ["float64_fma_test.cc"],
    data = [
        ":float64_fma.ir",
        ":float64_fma.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":float32_test_utils",
        ":float64_fma_jit_wrapper",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/tools:testbench",
        "//xls/tools:testbench_builder",
    ],
)

filegroup(
    name = "x_files",
    srcs = glob(["*.x"]),
    visibility = ["//xls:xls_users"],
)

filegroup(
    name = "ir_examples",
    srcs = [
        ":float32_add.ir",
        ":float32_add.opt.ir",
        ":float32_mul.ir",
        ":float32_mul.opt.ir",
        ":float32_sub.ir",
        ":float32_sub.opt.ir",
        ":float64_add.ir",
        ":float64_add.opt.ir",
        ":float64_mul.ir",
        ":float64_mul.opt.ir",
        ":float64_sub.ir",
        ":float64_sub.opt.ir",
    ],
)
