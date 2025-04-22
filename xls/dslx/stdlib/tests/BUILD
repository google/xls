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

xls_dslx_opt_ir(
    name = "umul_with_overflow_21_21_18",
    srcs = ["umul_instances.x"],
    dslx_top = "umul_with_overflow_21_21_18",
)

cc_xls_ir_jit_wrapper(
    name = "umul_with_overflow_21_21_18_jit_wrapper",
    src = ":umul_with_overflow_21_21_18",
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
    src = ":umul_with_overflow_35_32_18",
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
        "//xls/common:xls_gunit_main",
        "//xls/common/fuzzing:fuzztest",
        "//xls/common/status:matchers",
        "//xls/ir:value_view",
        "@googletest//:gtest",
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
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/ir:value_view",
        "@com_google_absl//absl/strings:str_format",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "float32_add_jit_wrapper_test",
    srcs = ["float32_add_jit_wrapper_test.cc"],
    deps = [
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx/stdlib:float32_add_jit_wrapper",
        "//xls/ir:value",
        "//xls/ir:value_utils",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "float32_add_test_cc",
    srcs = ["float32_add_test.cc"],
    tags = ["optonly"],
    deps = [
        ":float32_test_utils",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/dslx/stdlib:float32_add_jit_wrapper",
        "//xls/tests:testbench",
        "//xls/tests:testbench_builder",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "float32_mul_test_cc",
    srcs = ["float32_mul_test.cc"],
    data = [
        "//xls/dslx/stdlib:float32_mul.ir",
        "//xls/dslx/stdlib:float32_mul.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":float32_test_utils",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/dslx/stdlib:float32_mul_jit_wrapper",
        "//xls/tests:testbench",
        "//xls/tests:testbench_builder",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
    ],
)

xls_dslx_test(
    name = "float32_downcast_test",
    srcs = ["float32_downcast_test.x"],
)

xls_dslx_opt_ir(
    name = "float32_downcast",
    srcs = ["float32_downcast_test.x"],
    dslx_top = "f64_to_f32",
    ir_file = "float32_downcast.ir",
    opt_ir_file = "float32_downcast.opt.ir",
)

cc_xls_ir_jit_wrapper(
    name = "float32_downcast_jit_wrapper",
    src = ":float32_downcast",
    jit_wrapper_args = {
        "class_name": "F64ToF32",
        "namespace": "xls::fp",
    },
)

cc_test(
    name = "float32_downcast_test_cc",
    srcs = ["float32_downcast_test.cc"],
    tags = ["optonly"],
    deps = [
        ":float32_downcast_jit_wrapper",
        "//xls/common:xls_gunit_main",
        "//xls/common/fuzzing:fuzztest",
        "@com_google_absl//absl/base",
        "@googletest//:gtest",
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
    src = ":float32_upcast",
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
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/tests:testbench",
        "//xls/tests:testbench_builder",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "float32_test",
    srcs = ["float32_test.cc"],
    tags = ["optonly"],
    deps = [
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx/stdlib:float32_from_int32_wrapper",
        "//xls/dslx/stdlib:float32_to_int32_wrapper",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@googletest//:gtest",
    ],
)

# TODO(rspringer): Takes too long to run in normal testing.
cc_binary(
    name = "float32_add_bounds",
    srcs = ["float32_add_bounds.cc"],
    data = [
        "//xls/dslx/stdlib:float32_add.ir",
        "//xls/dslx/stdlib:float32_add.opt.ir",
    ],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:get_runfile_path",
        "//xls/common/status:status_macros",
        "//xls/dslx:virtualizable_file_system",
        "//xls/ir",
        "//xls/ir:ir_parser",
        "//xls/solvers:z3_ir_translator",
        "//xls/solvers:z3_utils",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@z3//:api",
    ],
)

cc_test(
    name = "float32_fma_test",
    srcs = ["float32_fma_test.cc"],
    data = [
        "//xls/dslx/stdlib:float32_fma.ir",
        "//xls/dslx/stdlib:float32_fma.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":float32_test_utils",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/dslx/stdlib:float32_fma_jit_wrapper",
        "//xls/tests:testbench",
        "//xls/tests:testbench_builder",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "float32_ldexp_test_cc",
    srcs = ["float32_ldexp_test.cc"],
    data = [
        "//xls/dslx/stdlib:float32_ldexp.ir",
        "//xls/dslx/stdlib:float32_ldexp.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":float32_test_utils",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/dslx/stdlib:float32_ldexp_jit_wrapper",
        "//xls/tests:testbench",
        "//xls/tests:testbench_builder",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "float32_fast_rsqrt_test_cc",
    srcs = ["float32_fast_rsqrt_test.cc"],
    data = [
        "//xls/dslx/stdlib:float32_fast_rsqrt.ir",
        "//xls/dslx/stdlib:float32_fast_rsqrt.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":float32_test_utils",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/dslx/stdlib:float32_fast_rsqrt_jit_wrapper",
        "//xls/tests:testbench",
        "//xls/tests:testbench_builder",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
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

cc_test(
    name = "float64_add_test_cc",
    srcs = ["float64_add_test.cc"],
    data = [
        "//xls/dslx/stdlib:float64_add.ir",
        "//xls/dslx/stdlib:float64_add.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/dslx/stdlib:float64_add_jit_wrapper",
        "//xls/tests:testbench",
        "//xls/tests:testbench_builder",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "float64_mul_test",
    srcs = ["float64_mul_test.cc"],
    data = [
        "//xls/dslx/stdlib:float64_mul.ir",
        "//xls/dslx/stdlib:float64_mul.opt.ir",
    ],
    # Flakes in fastbuild seen on 2021-06-04.
    tags = ["optonly"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/dslx/stdlib:float64_mul_jit_wrapper",
        "//xls/tests:testbench",
        "//xls/tests:testbench_builder",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "float64_fma_test",
    srcs = ["float64_fma_test.cc"],
    data = [
        "//xls/dslx/stdlib:float64_fma.ir",
        "//xls/dslx/stdlib:float64_fma.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/dslx/stdlib:float64_fma_jit_wrapper",
        "//xls/tests:testbench",
        "//xls/tests:testbench_builder",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "float32_ceil_test_cc",
    srcs = ["float32_ceil_test.cc"],
    data = [
        "//xls/dslx/stdlib:float32_ceil.ir",
        "//xls/dslx/stdlib:float32_ceil.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":float32_test_utils",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/dslx/stdlib:float32_ceil_jit_wrapper",
        "//xls/tests:testbench",
        "//xls/tests:testbench_builder",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "float32_floor_test_cc",
    srcs = ["float32_floor_test.cc"],
    data = [
        "//xls/dslx/stdlib:float32_floor.ir",
        "//xls/dslx/stdlib:float32_floor.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":float32_test_utils",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/dslx/stdlib:float32_floor_jit_wrapper",
        "//xls/tests:testbench",
        "//xls/tests:testbench_builder",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "float32_trunc_test_cc",
    srcs = ["float32_trunc_test.cc"],
    data = [
        "//xls/dslx/stdlib:float32_trunc.ir",
        "//xls/dslx/stdlib:float32_trunc.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":float32_test_utils",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/dslx/stdlib:float32_trunc_jit_wrapper",
        "//xls/tests:testbench",
        "//xls/tests:testbench_builder",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
    ],
)
