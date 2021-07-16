# XLS libraries ported from Go math library.

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "cc_xls_ir_jit_wrapper",
    "xls_benchmark_ir",
    "xls_dslx_library",
    "xls_dslx_module_library",
    "xls_dslx_opt_ir",
    "xls_dslx_test",
    "xls_eval_ir_test",
)

package(
    default_visibility = ["//xls:xls_internal"],
)

licenses(["notice"])

exports_files(["LICENSE"])

xls_dslx_library(
    name = "fpexp_32_dslx",
    srcs = ["fpexp_32.x"],
    deps = [
        "//xls/modules:fpadd_2x32_dslx",
        "//xls/modules:fpldexp_32_dslx",
        "//xls/modules:fpmul_2x32_dslx",
        "//xls/modules:fpsub_2x32_dslx",
        "//third_party/xls_berkeley_softfloat:fpdiv_2x32_dslx",
    ],
)

xls_dslx_module_library(
    name = "fpexp_32_dslx_module",
    src = "fpexp_32.x",
    deps = [":fpexp_32_dslx"],
)

xls_dslx_opt_ir(
    name = "fpexp_32",
    dep = ":fpexp_32_dslx_module",
    ir_file = "fpexp_32.ir",
    opt_ir_file = "fpexp_32.opt.ir",
)

xls_dslx_test(
    name = "fpexp_32_dslx_test",
    dep = ":fpexp_32_dslx_module",
)

# TODO 2021-03-08: Takes too long (> 15 minutes). Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "fpexp_32_eval_ir_test",
    src = ":fpexp_32.ir",
)

xls_benchmark_ir(
    name = "fpexp_32_benchmark_ir",
    src = ":fpexp_32.ir",
)

cc_xls_ir_jit_wrapper(
    name = "fpexp_32_jit_wrapper",
    src = ":fpexp_32.opt.ir",
    jit_wrapper_args = {
        "class_name": "fpexp_32",
    },
)

cc_test(
    name = "fpexp_32_test",
    srcs = ["fpexp_32_test.cc"],
    data = [
        ":fpexp_32.ir",
        ":fpexp_32.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":fpexp_32_jit_wrapper",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value_helpers",
        "//xls/ir:value_view_helpers",
        "//xls/tools:testbench_builder",
    ],
)

xls_dslx_library(
    name = "fp_trig_reduce_dslx",
    srcs = ["fp_trig_reduce.x"],
    deps = [
        "//xls/modules:apfloat_add_2_dslx",
        "//xls/modules:apfloat_mul_2_dslx",
        "//xls/modules:apfloat_sub_2_dslx",
    ],
)

xls_dslx_library(
    name = "fp_sincos_32_dslx",
    srcs = ["fp_sincos_32.x"],
    deps = [
        ":fp_trig_reduce_dslx",
        "//xls/modules:fpadd_2x32_dslx",
        "//xls/modules:fpmul_2x32_dslx",
        "//xls/modules:fpsub_2x32_dslx",
    ],
)

xls_dslx_module_library(
    name = "fp_sincos_32_dslx_module",
    src = "fp_sincos_32.x",
    deps = [":fp_sincos_32_dslx"],
)

xls_dslx_opt_ir(
    name = "fp_sincos_32",
    dep = ":fp_sincos_32_dslx_module",
    ir_file = "fp_sincos_32.ir",
    opt_ir_file = "fp_sincos_32.opt.ir",
)

xls_dslx_test(
    name = "fp_sincos_32_dslx_test",
    dep = ":fp_sincos_32_dslx_module",
)

# TODO(b/152546795): 2021-03-08 Takes too long (~5 minutes). Add
# xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "fp_sincos_32_eval_ir_test",
    src = ":fp_sincos_32.ir",
)

xls_benchmark_ir(
    name = "fp_sincos_32_benchmark_ir",
    src = ":fp_sincos_32.ir",
)

cc_xls_ir_jit_wrapper(
    name = "fp_sincos_32_jit_wrapper",
    src = ":fp_sincos_32.opt.ir",
    jit_wrapper_args = {
        "class_name": "fp_sincos_32",
    },
)

cc_test(
    name = "fp_sincos_32_test",
    srcs = ["fp_sincos_32_test.cc"],
    data = [
        ":fp_sincos_32.ir",
        ":fp_sincos_32.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":fp_sincos_32_jit_wrapper",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value_helpers",
        "//xls/ir:value_view_helpers",
        "//xls/tools:testbench_builder",
    ],
)

xls_dslx_module_library(
    name = "fp_trig_reduce_dslx_module",
    src = "fp_trig_reduce.x",
    deps = [
        ":fp_trig_reduce_dslx",
    ],
)

xls_dslx_opt_ir(
    name = "fp_trig_reduce",
    dep = ":fp_trig_reduce_dslx_module",
    ir_file = "fp_trig_reduce.ir",
    opt_ir_file = "fp_trig_reduce.opt.ir",
)

xls_dslx_test(
    name = "fp_trig_reduce_dslx_test",
    dep = ":fp_trig_reduce_dslx_module",
)

# TODO(b/152546795): 2021-03-08 Takes too long (~5 minutes). Add
# xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "fp_trig_reduce_eval_ir_test",
    src = ":fp_trig_reduce.ir",
)

xls_benchmark_ir(
    name = "fp_trig_reduce_benchmark_ir",
    src = ":fp_trig_reduce.ir",
)

cc_xls_ir_jit_wrapper(
    name = "fp_trig_reduce_jit_wrapper",
    src = ":fp_trig_reduce.opt.ir",
    jit_wrapper_args = {
        "class_name": "fp_trig_reduce",
    },
)

cc_test(
    name = "fp_trig_reduce_64_test",
    srcs = ["fp_trig_reduce_64_test.cc"],
    data = [
        ":fp_trig_reduce.ir",
        ":fp_trig_reduce.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":fp_trig_reduce_jit_wrapper",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value_helpers",
        "//xls/ir:value_view_helpers",
        "//xls/tools:testbench_builder",
    ],
)

xls_dslx_library(
    name = "fpsqrt_32_dslx",
    srcs = ["fpsqrt_32.x"],
)

xls_dslx_module_library(
    name = "fpsqrt_32_dslx_module",
    src = "fpsqrt_32.x",
    deps = [
        ":fpsqrt_32_dslx",
    ],
)

xls_dslx_opt_ir(
    name = "fpsqrt_32",
    dep = ":fpsqrt_32_dslx_module",
    ir_file = "fpsqrt_32.ir",
    opt_ir_file = "fpsqrt_32.opt.ir",
)

xls_dslx_test(
    name = "fpsqrt_32_dslx_test",
    dep = ":fpsqrt_32_dslx_module",
)

# TODO: 2021-03-22 Takes too long. Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "fpsqrt_32_eval_ir_test",
    src = ":fpsqrt_32.ir",
)

xls_benchmark_ir(
    name = "fpsqrt_32_benchmark_ir",
    src = ":fpsqrt_32.ir",
)

cc_xls_ir_jit_wrapper(
    name = "fpsqrt_32_jit_wrapper",
    src = ":fpsqrt_32.opt.ir",
    jit_wrapper_args = {
        "class_name": "fpsqrt_32",
    },
)

cc_test(
    name = "fpsqrt_32_test",
    srcs = ["fpsqrt_32_test.cc"],
    data = [
        ":fpsqrt_32.ir",
        ":fpsqrt_32.opt.ir",
    ],
    tags = ["optonly"],
    deps = [
        ":fpsqrt_32_jit_wrapper",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value_helpers",
        "//xls/ir:value_view_helpers",
        "//xls/tools:testbench_builder",
    ],
)
