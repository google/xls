# XLS libraries ported from Go math library.

load(
    "//xls/build_rules:xls_build_defs.bzl",
    "cc_xls_ir_jit_wrapper",
    "xls_benchmark_ir",
    "xls_dslx_library",
    "xls_dslx_opt_ir",
    "xls_dslx_test",
    "xls_eval_ir_test",
)
load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//xls:xls_internal"],
)

license(
    name = "license",
    package_name = "xls_go_math",
)

licenses(["notice"])

exports_files(["LICENSE"])

xls_dslx_library(
    name = "fpexp_32_dslx",
    srcs = ["fpexp_32.x"],
    deps = [
        "//third_party/xls_berkeley_softfloat:fpdiv_2x32_dslx",
    ],
)

xls_dslx_opt_ir(
    name = "fpexp_32",
    dslx_top = "fpexp_32",
    ir_file = "fpexp_32.ir",
    library = ":fpexp_32_dslx",
    opt_ir_file = "fpexp_32.opt.ir",
)

xls_dslx_test(
    name = "fpexp_32_dslx_test",
    library = ":fpexp_32_dslx",
)

# TODO 2021-03-08: Takes too long (> 15 minutes). Add xls_ir_equivalence_test.
xls_eval_ir_test(
    name = "fpexp_32_eval_ir_test",
    src = ":fpexp_32.ir",
)

xls_benchmark_ir(
    name = "fpexp_32_benchmark_ir",
    src = ":fpexp_32.ir",
    # TODO: Takes >15m for synthesis; too long for routine benchmarks.
    synthesize = False,
)

cc_xls_ir_jit_wrapper(
    name = "fpexp_32_jit_wrapper",
    src = ":fpexp_32.opt.ir",
    jit_wrapper_args = {
        "class_name": "fpexp_32",
        "namespace": "xls::fp",
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
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value_utils",
        "//xls/ir:value_view_utils",
        "//xls/tools:testbench_builder",
    ],
)

xls_dslx_library(
    name = "fp_trig_reduce_dslx",
    srcs = ["fp_trig_reduce.x"],
)

xls_dslx_library(
    name = "fp_sincos_32_dslx",
    srcs = ["fp_sincos_32.x"],
    deps = [
        ":fp_trig_reduce_dslx",
    ],
)

xls_dslx_opt_ir(
    name = "fp_sincos_32",
    dslx_top = "fp_sincos_32",
    ir_file = "fp_sincos_32.ir",
    library = ":fp_sincos_32_dslx",
    opt_ir_file = "fp_sincos_32.opt.ir",
)

xls_dslx_test(
    name = "fp_sincos_32_dslx_test",
    library = ":fp_sincos_32_dslx",
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
    # TODO: Hits OOM issues during synthesis; should get this fixed.
    synthesize = False,
)

cc_xls_ir_jit_wrapper(
    name = "fp_sincos_32_jit_wrapper",
    src = ":fp_sincos_32.opt.ir",
    jit_wrapper_args = {
        "class_name": "fp_sincos_32",
        "namespace": "xls::fp",
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
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value_utils",
        "//xls/ir:value_view_utils",
        "//xls/tools:testbench_builder",
    ],
)

xls_dslx_library(
    name = "fp_trig_reduce_dslx_module",
    srcs = ["fp_trig_reduce.x"],
    deps = [":fp_trig_reduce_dslx"],
)

xls_dslx_opt_ir(
    name = "fp_trig_reduce",
    dslx_top = "main",
    ir_file = "fp_trig_reduce.ir",
    library = ":fp_trig_reduce_dslx_module",
    opt_ir_file = "fp_trig_reduce.opt.ir",
)

xls_dslx_test(
    name = "fp_trig_reduce_dslx_test",
    library = ":fp_trig_reduce_dslx_module",
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
    # TODO: Takes >15m for synthesis; too long for routine benchmarks.
    synthesize = False,
)

cc_xls_ir_jit_wrapper(
    name = "fp_trig_reduce_jit_wrapper",
    src = ":fp_trig_reduce.opt.ir",
    jit_wrapper_args = {
        "class_name": "fp_trig_reduce",
        "namespace": "xls::fp",
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
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value_utils",
        "//xls/ir:value_view_utils",
        "//xls/tools:testbench_builder",
    ],
)

xls_dslx_library(
    name = "fpsqrt_32_dslx",
    srcs = ["fpsqrt_32.x"],
)

xls_dslx_opt_ir(
    name = "fpsqrt_32",
    dslx_top = "fpsqrt_32",
    ir_file = "fpsqrt_32.ir",
    library = ":fpsqrt_32_dslx",
    opt_ir_file = "fpsqrt_32.opt.ir",
)

xls_dslx_test(
    name = "fpsqrt_32_dslx_test",
    library = ":fpsqrt_32_dslx",
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
        "namespace": "xls::fp",
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
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "//xls/common:init_xls",
        "//xls/common:math_util",
        "//xls/common/file:get_runfile_path",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/ir:value_utils",
        "//xls/ir:value_view_utils",
        "//xls/tools:testbench_builder",
    ],
)
