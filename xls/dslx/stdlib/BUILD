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
    "xls_ir_equivalence_test",
)

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_users"],
    features = [
        "layering_check",
        "parse_headers",
    ],
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
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_prove_quickcheck_test(
    name = "std_dslx_prove_quickcheck_test",
    srcs = ["std.x"],
)

xls_dslx_test(
    name = "acm_random_dslx_test",
    srcs = ["acm_random.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_fmt_test(
    name = "acm_random_dslx_fmt_test",
    src = "acm_random.x",
    opportunistic_postcondition = True,
)

xls_dslx_library(
    name = "apfloat_dslx",
    srcs = ["apfloat.x"],
)

xls_dslx_test(
    name = "apfloat_dslx_test",
    srcs = ["apfloat.x"],
    dslx_test_args = {"compare": "jit"},
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
)

xls_dslx_prove_quickcheck_test(
    name = "bfloat16_dslx_prove_quickcheck_test",
    srcs = ["bfloat16.x"],
)

xls_dslx_fmt_test(
    name = "bfloat16_dslx_fmt_test",
    src = "bfloat16.x",
    opportunistic_postcondition = True,
)

xls_dslx_test(
    name = "bfloat16_dslx_test",
    srcs = ["bfloat16.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_opt_ir(
    name = "bfloat16_add",
    dslx_top = "add",
    ir_file = "bfloat16_add.ir",
    library = "bfloat16_dslx",
)

# TODO: xls_ir_equivalence_test takes a long time. Tag enormous and manual.
xls_ir_equivalence_test(
    name = "bfloat16_add_ir_opt_equivalence_test",
    size = "enormous",
    src_0 = ":bfloat16_add.ir",
    src_1 = ":bfloat16_add.opt.ir",
    tags = ["manual"],
)

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

# TODO: xls_ir_equivalence_test takes a long time. Tag enormous and manual.
xls_ir_equivalence_test(
    name = "bfloat16_sub_ir_opt_equivalence_test",
    size = "enormous",
    src_0 = ":bfloat16_sub.ir",
    src_1 = ":bfloat16_sub.opt.ir",
    tags = ["manual"],
)

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
)

xls_dslx_fmt_test(
    name = "float32_dslx_fmt_test",
    src = "float32.x",
    opportunistic_postcondition = True,
)

xls_dslx_test(
    name = "float32_dslx_test",
    srcs = ["float32.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_opt_ir(
    name = "float32_add",
    dslx_top = "add",
    ir_file = "float32_add.ir",
    library = ":float32_dslx",
    opt_ir_file = "float32_add.opt.ir",
)

# TODO: xls_ir_equivalence_test takes a long time. Tag enormous and manual.
xls_ir_equivalence_test(
    name = "float32_add_ir_opt_equivalence_test",
    size = "enormous",
    src_0 = ":float32_add.ir",
    src_1 = ":float32_add.opt.ir",
    tags = ["manual"],
)

xls_ir_cc_library(
    name = "float32_add_cc",
    src = ":float32_add",
    namespaces = "xls,fp",
)

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
    src = ":float32_add",
    jit_wrapper_args = {
        "class_name": "Float32Add",
        "namespace": "xls::fp",
    },
)

xls_dslx_opt_ir(
    name = "float32",
    dslx_top = "sub",
    ir_file = "float32_sub.ir",
    library = ":float32_dslx",
    opt_ir_file = "float32_sub.opt.ir",
)

# TODO: xls_ir_equivalence_test takes a long time. Tag enormous and manual.
xls_ir_equivalence_test(
    name = "float32_sub_ir_opt_equivalence_test",
    size = "enormous",
    src_0 = ":float32_sub.ir",
    src_1 = ":float32_sub.opt.ir",
    tags = ["manual"],
)

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
    src = ":float32_mul",
    jit_wrapper_args = {
        "class_name": "Float32Mul",
        "namespace": "xls::fp",
    },
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
    src = ":float32_to_int32",
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
    src = ":float32_from_int32",
    jit_wrapper_args = {
        "class_name": "Float32FromInt32",
        "function": get_mangled_ir_symbol("float32", "from_int32"),
        "namespace": "xls::dslx",
    },
)

xls_dslx_opt_ir(
    name = "float32_fma",
    dslx_top = "fma",
    library = ":float32_dslx",
    opt_ir_file = "float32_fma.opt.ir",
)

xls_ir_cc_library(
    name = "float32_fma_cc",
    src = ":float32_fma",
    namespaces = "xls,fp",
)

# TODO(b/152546795): xls_ir_equivalence_test won't work because of
# https://github.com/google/xls/issues/365
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
    src = ":float32_fma",
    jit_wrapper_args = {
        "class_name": "Float32Fma",
        "namespace": "xls::fp",
    },
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
    src = ":float32_ldexp",
    jit_wrapper_args = {
        "class_name": "Float32Ldexp",
        "namespace": "xls::fp",
    },
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
    src = ":float32_fast_rsqrt",
    jit_wrapper_args = {
        "class_name": "Float32FastRsqrt",
        "namespace": "xls::fp",
    },
)

xls_dslx_library(
    name = "float64_dslx",
    srcs = ["float64.x"],
)

xls_dslx_fmt_test(
    name = "float64_dslx_fmt_test",
    src = "float64.x",
    opportunistic_postcondition = True,
)

xls_dslx_test(
    name = "float64_dslx_test",
    srcs = ["float64.x"],
    dslx_test_args = {"compare": "jit"},
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
    src = ":float64_add",
    jit_wrapper_args = {
        "class_name": "Float64Add",
        "namespace": "xls::fp",
    },
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
    src = ":float64_mul",
    jit_wrapper_args = {
        "class_name": "Float64Mul",
        "namespace": "xls::fp",
    },
)

xls_dslx_opt_ir(
    name = "float64_fma",
    dslx_top = "fma",
    ir_file = "float64_fma.ir",
    library = ":float64_dslx",
    opt_ir_file = "float64_fma.opt.ir",
)

# TODO(b/152546795): xls_ir_equivalence_test won't work because of
# https://github.com/google/xls/issues/365
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
    src = ":float64_fma",
    jit_wrapper_args = {
        "class_name": "Float64Fma",
        "namespace": "xls::fp",
    },
)

xls_dslx_opt_ir(
    name = "float32_ceil",
    dslx_top = "ceil",
    ir_file = "float32_ceil.ir",
    library = ":float32_dslx",
    opt_ir_file = "float32_ceil.opt.ir",
)

cc_xls_ir_jit_wrapper(
    name = "float32_ceil_jit_wrapper",
    src = ":float32_ceil.opt.ir",
    jit_wrapper_args = {
        "class_name": "F32Ceil",
        "namespace": "xls::fp",
    },
)

xls_dslx_opt_ir(
    name = "float32_floor",
    dslx_top = "floor",
    ir_file = "float32_floor.ir",
    library = ":float32_dslx",
    opt_ir_file = "float32_floor.opt.ir",
)

cc_xls_ir_jit_wrapper(
    name = "float32_floor_jit_wrapper",
    src = ":float32_floor.opt.ir",
    jit_wrapper_args = {
        "class_name": "F32Floor",
        "namespace": "xls::fp",
    },
)

xls_dslx_opt_ir(
    name = "float32_trunc",
    dslx_top = "trunc",
    ir_file = "float32_trunc.ir",
    library = ":float32_dslx",
    opt_ir_file = "float32_trunc.opt.ir",
)

cc_xls_ir_jit_wrapper(
    name = "float32_trunc_jit_wrapper",
    src = ":float32_trunc.opt.ir",
    jit_wrapper_args = {
        "class_name": "F32Trunc",
        "namespace": "xls::fp",
    },
)

xls_dslx_library(
    name = "abs_diff_dslx",
    srcs = ["abs_diff.x"],
)

xls_dslx_fmt_test(
    name = "abs_diff_dslx_fmt_test",
    src = "abs_diff.x",
    opportunistic_postcondition = True,
)

xls_dslx_test(
    name = "abs_diff_dslx_test",
    srcs = ["abs_diff.x"],
    dslx_test_args = {"compare": "jit"},
)

xls_dslx_prove_quickcheck_test(
    name = "abs_diff_dslx_prove_quickcheck_test",
    srcs = ["abs_diff.x"],
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
